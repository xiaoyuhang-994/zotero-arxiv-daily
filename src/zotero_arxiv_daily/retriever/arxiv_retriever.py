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
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import re

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



class _PlainText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in ("p", "br", "div", "li"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("p", "div", "li"):
            self.parts.append(" ")


def _rss_text(value) -> str:
    text = str(value or "")
    # Atom text abstracts may contain mathematical '<' characters, not HTML.
    if re.search(r"</?(?:p|div|span|a|br|b|i|em|strong|sub|sup)\b", text, re.I):
        parser = _PlainText()
        parser.feed(text)
        text = "".join(parser.parts)
    else:
        text = unescape(text)
    return " ".join(text.split())


def _paper_id(value: str) -> str:
    value = value.removeprefix("oai:arXiv.org:")
    return value.split("/abs/", 1)[-1].strip()


@dataclass
class _RSSMetadata:
    title: str
    authors: list[str]
    summary: str
    entry_id: str
    pdf_url: str


class _RSSSummaryPaper(Paper):
    def generate_tldr(self, openai_client, llm_params):
        text = super().generate_tldr(openai_client, llm_params)
        self.tldr = "【仅基于 RSS 摘要，未读取全文】" + (text or "")
        return self.tldr

    def generate_affiliations(self, openai_client, llm_params):
        # RSS does not provide reliable author affiliation information.
        self.affiliations = None
        return None


def _metadata_from_rss(entry, paper_id: str) -> _RSSMetadata | None:
    title = _rss_text(entry.get("title", ""))
    summary = _rss_text(entry.get("summary", "") or entry.get("description", ""))
    # Official Atom summary begins with an ID/announcement header.
    if summary.lower().startswith("arxiv:"):
        pieces = re.split(r"\bAbstract:\s*", summary, maxsplit=1, flags=re.I)
        summary = pieces[1].strip() if len(pieces) == 2 else ""
    if not title or not summary:
        return None
    author_values = [a.get("name", "") for a in entry.get("authors", [])]
    if not any(author_values):
        author_values = [entry.get("author", "") or entry.get("dc_creator", "")]
    authors = []
    for value in author_values:
        for name in _rss_text(value).split(","):
            name = name.strip()
            if name and name not in authors:
                authors.append(name)
    return _RSSMetadata(
        title=title, authors=authors, summary=summary,
        entry_id=f"https://arxiv.org/abs/{paper_id}",
        pdf_url=f"https://arxiv.org/pdf/{paper_id}",
    )


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult | _RSSMetadata]:
        # On API failure use the already-loaded RSS, without per-ID fan-out.
        client = arxiv.Client(num_retries=0, delay_seconds=10)
        query = '+'.join(self.config.source.arxiv.category)
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if getattr(feed, "status", 200) >= 400:
            raise RuntimeError(f"arXiv RSS HTTP {feed.status}: {query}")
        title = feed.feed.get("title", "")
        if not title or 'Feed error for query' in title or getattr(feed, "bozo", False):
            raise RuntimeError(f"Invalid or malformed arXiv RSS feed: {query}")
        allowed_types = {"new", "cross"} if self.config.source.arxiv.get(
            "include_cross_list", False
        ) else {"new"}
        entries_by_id = {}
        for entry in feed.entries:
            if entry.get("arxiv_announce_type", "new") not in allowed_types:
                continue
            pid = _paper_id(entry.get("id", ""))
            if not re.fullmatch(r"(?:[0-9]{4}\.[0-9]{4,5}|[A-Za-z.-]+/[0-9]{7})(?:v[0-9]+)?", pid):
                raise RuntimeError(f"Invalid arXiv ID in RSS: {pid!r}")
            if pid not in entries_by_id:
                entries_by_id[pid] = entry
        ids = list(entries_by_id)
        if self.config.executor.debug:
            ids = ids[:10]
        logger.info(f"arXiv RSS candidates: {len(ids)}")
        if not ids:
            return []

        raw_papers = []
        api_count = 0
        rss_count = 0
        skipped = []
        api_available = True
        with tqdm(total=len(ids), desc="arXiv metadata IDs processed") as bar:
            for offset in range(0, len(ids), 20):
                batch_ids = ids[offset:offset + 20]
                batch = []
                if api_available:
                    sleep(3)
                    try:
                        batch = list(client.results(arxiv.Search(
                            id_list=batch_ids, max_results=len(batch_ids)
                        )))
                    except (arxiv.HTTPError, arxiv.UnexpectedEmptyPageError,
                            requests.exceptions.RequestException) as exc:
                        api_available = False
                        logger.warning(
                            f"arXiv API unavailable ({type(exc).__name__}, "
                            f"status={getattr(exc, 'status', 'network/parse')}); "
                            "using cached RSS metadata for this and remaining "
                            "batches. No further API requests in this run."
                        )
                by_id = {_paper_id(p.entry_id): p for p in batch}
                for pid in batch_ids:
                    if pid in by_id:
                        raw_papers.append(by_id[pid])
                        api_count += 1
                    else:
                        paper = _metadata_from_rss(entries_by_id[pid], pid)
                        if paper is not None:
                            raw_papers.append(paper)
                            rss_count += 1
                        else:
                            skipped.append(pid)
                            logger.warning(f"Skipping {pid}: RSS title or abstract missing")
                bar.update(len(batch_ids))
        logger.info(
            f"arXiv metadata summary: candidates={len(ids)}, "
            f"api={api_count}, rss_fallback={rss_count}, skipped={len(skipped)}"
        )
        if not raw_papers:
            raise RuntimeError("RSS contains candidates but no usable metadata; no email generated.")
        if skipped:
            logger.warning("Recommendation coverage incomplete. Skipped IDs: " + ", ".join(skipped))
        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult | _RSSMetadata) -> Paper:
        if isinstance(raw_paper, _RSSMetadata):
            return _RSSSummaryPaper(
                source=self.name, title=raw_paper.title, authors=raw_paper.authors,
                abstract=raw_paper.summary, url=raw_paper.entry_id,
                pdf_url=raw_paper.pdf_url, full_text=None,
            )
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
