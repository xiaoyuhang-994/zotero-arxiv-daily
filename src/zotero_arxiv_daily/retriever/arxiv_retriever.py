from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        # Retry only here: avoid multiplying library and application retries.
        client = arxiv.Client(num_retries=0, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Keep RSS access at the original feedparser boundary so the existing
        # offline test fixture can intercept it without making network calls.
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if getattr(feed, "status", 200) >= 400:
            raise RuntimeError(f"arXiv RSS HTTP {feed.status}: {query}")
        title = feed.feed.get("title", "")
        if not title or 'Feed error for query' in title:
            raise RuntimeError(f"Invalid or unavailable arXiv RSS feed: {query}")

        allowed_types = {"new", "cross"} if include_cross_list else {"new"}
        all_paper_ids = list(dict.fromkeys(
            entry.id.removeprefix("oai:arXiv.org:")
            for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_types
        ))
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]
        logger.info(f"arXiv RSS candidates: {len(all_paper_ids)}")
        if not all_paper_ids:
            return []

        def fetch(ids: list[str]) -> list[ArxivResult]:
            for attempt in range(3):
                # Also separate RSS, batch and single-paper API requests.
                sleep(3)
                try:
                    return list(client.results(arxiv.Search(
                        id_list=ids, max_results=len(ids)
                    )))
                except arxiv.HTTPError as exc:
                    # 406 is handled by the caller. Do not retry it unchanged.
                    transient = exc.status == 429 or 500 <= exc.status < 600
                    if not transient or attempt == 2:
                        raise
                    delay = 30 * (2 ** attempt)
                    logger.warning(
                        f"arXiv HTTP {exc.status}; retry {attempt + 1}/2 "
                        f"after {delay}s"
                    )
                    sleep(delay)
                except (requests.exceptions.RequestException,
                        arxiv.UnexpectedEmptyPageError) as exc:
                    if attempt == 2:
                        raise
                    delay = 30 * (2 ** attempt)
                    logger.warning(
                        f"arXiv {type(exc).__name__}; retry after {delay}s"
                    )
                    sleep(delay)
            raise RuntimeError("arXiv retry loop exhausted")

        raw_papers = []
        skipped_ids = []
        with tqdm(total=len(all_paper_ids), desc="arXiv metadata IDs processed") as bar:
            for offset in range(0, len(all_paper_ids), 20):
                batch_ids = all_paper_ids[offset:offset + 20]
                try:
                    batch = fetch(batch_ids)
                except arxiv.HTTPError as exc:
                    if exc.status != 406:
                        # Persistent 429/5xx: stop, rather than fan out requests.
                        raise
                    logger.warning(
                        f"arXiv batch HTTP 406; falling back to single-paper "
                        f"requests for {len(batch_ids)} IDs"
                    )
                    batch = []

                # Recover omitted entries as well as rejected batches.
                by_id = {paper.entry_id.split("/abs/", 1)[-1]: paper for paper in batch}
                for paper_id in batch_ids:
                    if paper_id in by_id:
                        continue
                    try:
                        single = fetch([paper_id])
                    except arxiv.HTTPError as exc:
                        if exc.status not in (404, 406):
                            raise
                        logger.warning(
                            f"Skipping arXiv {paper_id}: HTTP {exc.status}"
                        )
                        single = []
                    by_id.update({
                        paper.entry_id.split("/abs/", 1)[-1]: paper for paper in single
                    })
                    if paper_id not in by_id:
                        skipped_ids.append(paper_id)
                        logger.warning(f"No usable metadata for arXiv {paper_id}")

                raw_papers.extend(by_id[pid] for pid in batch_ids if pid in by_id)
                bar.update(len(batch_ids))
                # A completely inaccessible first batch likely indicates a
                # wider service problem; do not send hundreds more requests.
                if not raw_papers:
                    raise RuntimeError(
                        "arXiv returned no usable metadata for the first batch, "
                        "including single-paper attempts. Stopping; this is "
                        "not a day with zero new papers."
                    )

        logger.info(
            f"arXiv metadata summary: candidates={len(all_paper_ids)}, "
            f"retrieved={len(raw_papers)}, skipped={len(skipped_ids)}"
        )
        if skipped_ids:
            logger.warning("Recommendation coverage is incomplete. Skipped IDs: "
                           + ", ".join(skipped_ids))
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
