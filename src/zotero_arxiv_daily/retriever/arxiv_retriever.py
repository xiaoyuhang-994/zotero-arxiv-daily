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
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from time import monotonic

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


def _validated_id(value):
    pid = _paper_id(str(value))
    if not re.fullmatch(r"(?:[0-9]{4}\.[0-9]{4,5}|[A-Za-z.-]+/[0-9]{7})(?:v[0-9]+)?", pid):
        raise RuntimeError("Invalid arXiv ID: " + repr(pid))
    return re.sub(r"v\d+$", "", pid)


@dataclass
class _APIMetadata(_RSSMetadata):
    pass


class _APISummaryPaper(Paper):
    def generate_tldr(self, openai_client, llm_params):
        text = super().generate_tldr(openai_client, llm_params)
        self.tldr = "【基于 arXiv 摘要，未读取全文】" + (text or "")
        return self.tldr

    def generate_affiliations(self, openai_client, llm_params):
        self.affiliations = None
        return None


def _metadata_from_api(entry, categories, include_cross_list):
    pid = _validated_id(entry.get("id", ""))
    tags = {tag.get("term") for tag in entry.get("tags", [])}
    primary = entry.get("arxiv_primary_category", {}).get("term")
    if not tags or not primary:
        raise RuntimeError("API entry missing category metadata: " + pid)
    if not (set(categories) & tags if include_cross_list else primary in categories):
        return None
    title = _rss_text(entry.get("title", ""))
    summary = _rss_text(entry.get("summary", ""))
    if not title or not summary:
        raise RuntimeError("API entry missing title or abstract: " + pid)
    authors = [_rss_text(a.get("name", "")) for a in entry.get("authors", [])]
    return _APIMetadata(title, [a for a in authors if a], summary,
                        "https://arxiv.org/abs/" + pid, "https://arxiv.org/pdf/" + pid)


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self):
        mode = os.environ.get("ZAD_ARXIV_MODE") or self.retriever_config.get("retrieval_mode", "rss_first")
        if mode == "rss_first":
            return self._retrieve_rss_first()
        if mode != "api_first":
            raise ValueError("source.arxiv.retrieval_mode must be api_first or rss_first")

        self._last_request = None
        self._api_partial = {}
        self._report = {"mode": mode, "api_complete": False, "api_count": 0,
                        "rss_count": 0, "rss_raw_entries": 0, "candidates": 0}
        try:
            papers = self._retrieve_api_first()
            if self.config.executor.debug:
                papers = papers[:10]
            self._report["candidates"] = len(papers)
            logger.info("arXiv retrieval summary: {}", self._report)
            return papers
        except Exception as exc:
            self._report["error"] = str(exc)
            raise
        finally:
            report_path = os.environ.get("ZAD_DIAGNOSTICS_PATH")
            if report_path:
                path = Path(report_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(self._report, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_atom(self, url, params=None):
        # One connection at a time, at least three seconds between request starts.
        retries = int(self.retriever_config.get("api_retries", 1))
        if not 0 <= retries <= 3:
            raise ValueError("api_retries must be between 0 and 3")
        for attempt in range(retries + 1):
            if self._last_request is not None:
                sleep(max(0, 3.1 - (monotonic() - self._last_request)))
            self._last_request = monotonic()
            retry_delay = 30 * (attempt + 1)
            try:
                response = requests.get(
                    url, params=params, timeout=(10, 45),
                    headers={"User-Agent": "ZoteroArxivDaily/1.0 (metadata reader)",
                             "Accept": "application/atom+xml, application/xml;q=0.9"},
                )
                # A long server cooldown should not turn into rapid retries.
                retry_after = response.headers.get("Retry-After", "")
                try:
                    requested_wait = float(retry_after) if retry_after else 0
                except ValueError:
                    requested_wait = 121  # HTTP-date: do not retry this endpoint now.
                transient = response.status_code in (429, 500, 502, 503, 504)
                if transient and attempt < retries and requested_wait <= 120:
                    retry_delay = max(retry_delay, requested_wait)
                    logger.warning("arXiv HTTP {}; retrying in {}s", response.status_code, retry_delay)
                    response.close()
                    sleep(retry_delay)
                    continue
                response.raise_for_status()
                content = response.content
                response.close()
                feed = feedparser.parse(content)
                if feed.get("bozo") or not feed.get("version", "").startswith("atom"):
                    raise RuntimeError("arXiv returned malformed XML or a non-Atom response")
                if "error" in feed.feed.get("title", "").lower():
                    raise RuntimeError("arXiv feed reports an error: " + feed.feed.get("title", ""))
                return feed
            except requests.exceptions.HTTPError:
                raise
            except requests.exceptions.RequestException as exc:
                if attempt >= retries:
                    raise
                logger.warning("arXiv network error {}; retrying in {}s", type(exc).__name__, retry_delay)
                sleep(retry_delay)

    def _retrieve_api_first(self):
        categories = list(self.retriever_config.category)
        if not categories or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9.-]*", c) for c in categories):
            raise ValueError("Provide valid arXiv categories, e.g. quant-ph and physics.optics")
        days = int(self.retriever_config.get("lookback_days", 7))
        cap = int(self.retriever_config.get("max_candidates", 1000))
        page_size = int(self.retriever_config.get("api_page_size", 200))
        if not 1 <= days <= 31 or not 1 <= cap <= 5000 or not 1 <= page_size <= 1000:
            raise ValueError("lookback_days: 1..31; max_candidates: 1..5000; api_page_size: 1..1000")
        now = datetime.now(timezone.utc)
        start_time = (now - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)
        query = "(" + " OR ".join("cat:" + c for c in categories) + ")"
        query += f" AND submittedDate:[{start_time:%Y%m%d%H%M} TO {now:%Y%m%d%H%M}]"
        self._report.update(query=query, window_start_utc=start_time.isoformat(),
                            window_end_utc=now.isoformat())
        logger.info("arXiv API-first query: {}", query)
        api_error = None
        try:
            offset = 0
            observed_ids = set()
            expected_total = None
            while True:
                feed = self._read_atom("https://export.arxiv.org/api/query", params={
                    "search_query": query, "sortBy": "submittedDate", "sortOrder": "descending",
                    "start": offset, "max_results": min(page_size, cap - offset),
                })
                try:
                    total = int(feed.feed["opensearch_totalresults"])
                except (KeyError, ValueError, TypeError):
                    raise RuntimeError("arXiv API response is missing a valid totalResults")
                if total < 0:
                    raise RuntimeError("arXiv API returned a negative totalResults")
                if expected_total is not None and total != expected_total:
                    raise RuntimeError("arXiv result count changed during pagination; coverage uncertain")
                expected_total = total
                self._report["api_total"] = total
                entries = feed.entries
                if not entries and total > offset:
                    raise RuntimeError(f"arXiv API returned an empty page at {offset}, but totalResults={total}")
                if total == 0:
                    if entries:
                        raise RuntimeError("arXiv API returned entries with totalResults=0")
                    self._report["api_complete"] = True
                    logger.info("arXiv API explicitly confirmed 0 results for the logged date window")
                    return []
                for entry in entries:
                    pid = _validated_id(entry.get("id", ""))
                    if pid in observed_ids:
                        raise RuntimeError("arXiv API repeated IDs across pages; coverage uncertain")
                    observed_ids.add(pid)
                    paper = _metadata_from_api(entry, categories, self.retriever_config.get("include_cross_list", False))
                    if paper is not None:
                        self._api_partial[pid] = paper
                offset += len(entries)
                self._report["api_count"] = len(self._api_partial)
                logger.info("arXiv API page: seen={}/{}, accepted={}", offset, total, len(self._api_partial))
                if offset >= total:
                    self._report["api_complete"] = True
                    return list(self._api_partial.values())
                if offset >= cap:
                    self._report["truncated"] = True
                    logger.warning("API candidate limit {} reached (total {}); increase max_candidates for full coverage", cap, total)
                    return list(self._api_partial.values())
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            api_error = f"{type(exc).__name__}: {exc}"
            self._report["api_error"] = api_error
            logger.warning("arXiv API failed: {}. Trying official RSS fallback.", api_error)

        papers = dict(self._api_partial)
        try:
            feed = self._read_atom("https://rss.arxiv.org/atom/" + "+".join(categories))
            if not feed.feed.get("title"):
                raise RuntimeError("arXiv RSS has no feed title")
            self._report.update(rss_raw_entries=len(feed.entries), rss_updated=feed.feed.get("updated"))
            allowed = {"new", "cross"} if self.retriever_config.get("include_cross_list", False) else {"new"}
            counts = {}
            for entry in feed.entries:
                announcement = str(entry.get("arxiv_announce_type", "new")).strip().lower()
                counts[announcement] = counts.get(announcement, 0) + 1
                if announcement not in allowed:
                    continue
                pid = _validated_id(entry.get("id", ""))
                paper = _metadata_from_rss(entry, pid)
                if paper is None:
                    raise RuntimeError("RSS entry missing title or abstract: " + pid)
                if pid not in papers:
                    papers[pid] = paper
                    self._report["rss_count"] += 1
            self._report["rss_announce_types"] = counts
            logger.warning("RSS fallback: raw={}, types={}, accepted={}; RSS cannot replace the API date window",
                           len(feed.entries), counts, self._report["rss_count"])
        except (requests.exceptions.RequestException, RuntimeError) as exc:
            self._report["rss_error"] = f"{type(exc).__name__}: {exc}"
            logger.warning("arXiv RSS fallback failed: {}", exc)
        if not papers:
            raise RuntimeError(
                "arXiv API failed and RSS supplied no usable candidates. "
                "Retrieval failed; this is NOT a confirmed day with no new papers. " + str(api_error)
            )
        self._report["coverage_incomplete"] = True
        logger.warning("Proceeding with {} available papers; date-window coverage is incomplete", len(papers))
        return list(papers.values())

    def retrieve_papers(self):
        mode = os.environ.get("ZAD_ARXIV_MODE") or self.retriever_config.get("retrieval_mode", "rss_first")
        if mode != "api_first":
            return super().retrieve_papers()
        # Metadata conversion performs no network requests, so do not sleep per paper.
        return [self.convert_to_paper(p) for p in self._retrieve_raw_papers()]


    def _retrieve_rss_first(self) -> list[ArxivResult | _RSSMetadata]:
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
            cls = _APISummaryPaper if isinstance(raw_paper, _APIMetadata) else _RSSSummaryPaper
            return cls(
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
