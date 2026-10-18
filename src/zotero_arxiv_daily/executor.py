from loguru import logger
from pyzotero import zotero
from omegaconf import DictConfig, ListConfig
from .utils import glob_match
from .retriever import get_retriever_cls
from .protocol import CorpusPaper
import random
from datetime import datetime, timezone
import os
import re
import json
import hashlib
from pathlib import Path
from .reranker import get_reranker_cls
from .construct_email import render_email
from .utils import send_email
from openai import OpenAI
from tqdm import tqdm


def normalize_path_patterns(patterns: list[str] | ListConfig | None, config_key: str) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(f"config.zotero.{config_key} must contain only glob pattern strings.")

    return list(patterns)


def _delivery_key(paper):
    url = str(paper.url).strip()
    if paper.source == "arxiv":
        pid = url.split("/abs/", 1)[-1]
        pid = re.sub(r"v\d+$", "", pid)
        return "arxiv:" + pid
    return str(paper.source) + ":" + url


def _load_delivery_state(config):
    directory = os.environ.get("ZAD_STATE_DIR")
    if not directory:
        logger.warning("Sent-history persistence is disabled (ZAD_STATE_DIR is not set)")
        return None, None
    # Separate histories if this workflow is used with a different recipient.
    receiver = str(config.email.receiver).strip().lower()
    recipient = hashlib.sha256(receiver.encode("utf-8")).hexdigest()[:16]
    path = Path(directory) / "sent.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("version") != 1 or not isinstance(state.get("recipients"), dict):
                raise ValueError("unsupported history format")
            if any(not isinstance(items, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                   for k, v in items.items()) for items in state["recipients"].values()):
                raise ValueError("invalid recipient history")
        except (ValueError, AttributeError) as exc:
            raise RuntimeError("Invalid sent history; refusing to silently reset it") from exc
    else:
        logger.warning("No previous sent history restored; earlier emails cannot be deduplicated")
        state = {"version": 1, "recipients": {}}
    state["recipients"].setdefault(recipient, {})
    # Check the output directory is writable before contacting SMTP.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    temporary.unlink()
    return (path, state, recipient), state["recipients"][recipient]


def _record_delivery(context, papers):
    if context is None or not papers:
        return
    path, state, recipient = context
    delivered_at = datetime.now(timezone.utc).isoformat()
    for paper in papers:
        state["recipients"][recipient][_delivery_key(paper)] = delivered_at
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    logger.info("Recorded {} sent papers in {}", len(papers), path)


class Executor:
    def __init__(self, config:DictConfig):
        self.config = config
        self.include_path_patterns = normalize_path_patterns(config.zotero.include_path, "include_path")
        self.ignore_path_patterns = normalize_path_patterns(config.zotero.ignore_path, "ignore_path")
        self.retrievers = {
            source: get_retriever_cls(source)(config) for source in config.executor.source
        }
        self.reranker = get_reranker_cls(config.executor.reranker)(config)
        self.openai_client = OpenAI(api_key=config.llm.api.key, base_url=config.llm.api.base_url)
    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")
        zot = zotero.Zotero(self.config.zotero.user_id, 'user', self.config.zotero.api_key)
        collections = zot.everything(zot.collections())
        collections = {c['key']:c for c in collections}
        corpus = zot.everything(zot.items(itemType='conferencePaper || journalArticle || preprint'))
        corpus = [c for c in corpus if c['data']['abstractNote'] != '']
        def get_collection_path(col_key:str) -> str:
            if p := collections[col_key]['data']['parentCollection']:
                return get_collection_path(p) + '/' + collections[col_key]['data']['name']
            else:
                return collections[col_key]['data']['name']
        for c in corpus:
            paths = [get_collection_path(col) for col in c['data']['collections']]
            c['paths'] = paths
        logger.info(f"Fetched {len(corpus)} zotero papers")
        return [CorpusPaper(
            title=c['data']['title'],
            abstract=c['data']['abstractNote'],
            added_date=datetime.strptime(c['data']['dateAdded'], '%Y-%m-%dT%H:%M:%SZ'),
            paths=c['paths']
        ) for c in corpus]
    
    def filter_corpus(self, corpus:list[CorpusPaper]) -> list[CorpusPaper]:
        if self.include_path_patterns:
            logger.info(f"Selecting zotero papers matching include_path: {self.include_path_patterns}")
            corpus = [
                c for c in corpus
                if any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.include_path_patterns
                )
            ]
        if self.ignore_path_patterns:
            logger.info(f"Excluding zotero papers matching ignore_path: {self.ignore_path_patterns}")
            corpus = [
                c for c in corpus
                if not any(
                    glob_match(path, pattern)
                    for path in c.paths
                    for pattern in self.ignore_path_patterns
                )
            ]
        if self.include_path_patterns or self.ignore_path_patterns:
            samples = random.sample(corpus, min(5, len(corpus)))
            samples = '\n'.join([c.title + ' - ' + '\n'.join(c.paths) for c in samples])
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples}\n...")
        return corpus

    
    def run(self):
        delivery_context, sent = _load_delivery_state(self.config)
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)
        if len(corpus) == 0:
            raise RuntimeError("No Zotero papers with abstracts found; check library sync and filters")
        all_papers = []
        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")
            papers = retriever.retrieve_papers()
            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue
            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)
        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")
        unique_papers = {}
        for paper in all_papers:
            key = _delivery_key(paper)
            if sent is None or key not in sent:
                unique_papers.setdefault(key, paper)
        previous_count = len(all_papers)
        all_papers = list(unique_papers.values())
        logger.info("After sent-history filtering: fetched={}, eligible={}", previous_count, len(all_papers))
        if int(self.config.executor.max_paper_num) < 1:
            raise ValueError("executor.max_paper_num must be at least 1")
        reranked_papers = []
        if len(all_papers) > 0:
            logger.info("Reranking papers...")
            reranked_papers = self.reranker.rerank(all_papers, corpus)
            reranked_papers = reranked_papers[:self.config.executor.max_paper_num]
            logger.info("Generating TLDR and affiliations...")
            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)
        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return
        logger.info("Sending email...")
        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)
        logger.info("Email sent successfully")
        _record_delivery(delivery_context, reranked_papers)
