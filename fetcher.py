"""
fetcher.py — Semantic Scholar paper discovery and enrichment.

All API calls use SS batch/bulk endpoints for efficiency:
  - Seeds:     POST /paper/batch       (1 call for all seeds)
  - Discovery: GET  /paper/search/bulk (token-paginated, 1000/page)
  - Enrich:    POST /paper/batch       (references, 500/call)
               POST /author/batch      (h-index, 1000/call)

Typical run (4 seeds, 500 candidates): ~8 SS API calls total.
"""

import os
import re
import time
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import yaml
import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic Scholar config
# ---------------------------------------------------------------------------

_SS_BASE         = "https://api.semanticscholar.org/graph/v1"
_SS_BATCH_LIMIT  = 500    # max IDs per POST /paper/batch
_SS_AUTHOR_BATCH = 1000   # max IDs per POST /author/batch
_SS_LEEWAY       = 1.5    # seconds between API calls (extra safety margin)

_SS_API_KEY: str | None = os.environ.get("SS_API_KEY") or None


def configure_api_key(key: str) -> None:
    """Set the Semantic Scholar API key (or set SS_API_KEY env var)."""
    global _SS_API_KEY
    _SS_API_KEY = key.strip() or None
    log.info("SS API key configured")


def _ss_headers() -> dict:
    h: dict[str, str] = {}
    if _SS_API_KEY:
        h["x-api-key"] = _SS_API_KEY
    return h


# ---------------------------------------------------------------------------
# HTTP helpers with retry + 429 handling + leeway
# ---------------------------------------------------------------------------

def _ss_get(url: str, params: dict | None = None, retries: int = 3):
    """GET with retry, 429 back-off, and leeway sleep."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params,
                                headers=_ss_headers(), timeout=60)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10)) + _SS_LEEWAY
                log.warning("SS rate limited (GET); sleeping %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(_SS_LEEWAY)
            return resp.json()
        except requests.RequestException as exc:
            log.warning("SS GET error (attempt %d/%d): %s",
                        attempt + 1, retries, exc)
            time.sleep(2 ** attempt + _SS_LEEWAY)
    return None


def _ss_post(url: str, json: dict, params: dict | None = None, retries: int = 3):
    """POST with retry, 429 back-off, and leeway sleep."""
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=json, params=params,
                                 headers=_ss_headers(), timeout=60)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10)) + _SS_LEEWAY
                log.warning("SS rate limited (POST); sleeping %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(_SS_LEEWAY)
            return resp.json()
        except requests.RequestException as exc:
            log.warning("SS POST error (attempt %d/%d): %s",
                        attempt + 1, retries, exc)
            time.sleep(2 ** attempt + _SS_LEEWAY)
    return None


# ---------------------------------------------------------------------------
# SS field-of-study → arXiv category mapping (for scorer H6 compatibility)
# ---------------------------------------------------------------------------

_SS_TO_CATEGORY: dict[str, str] = {
    "Machine Learning":                          "cs.LG",
    "Artificial Intelligence":                   "cs.AI",
    "Computer Vision and Pattern Recognition":   "cs.CV",
    "Natural Language Processing":               "cs.CL",
    "Computation and Language":                  "cs.CL",
    "Databases":                                 "cs.DB",
    "Data Mining":                               "cs.DB",
    "Information Retrieval":                     "cs.IR",
    "Robotics":                                  "cs.RO",
    "Human-Computer Interaction":                "cs.HC",
    "Computer Science":                          "cs.AI",
    "Mathematics":                               "stat.ML",
}


def _ss_fields_to_categories(s2_fields: list[dict] | None) -> list[str]:
    """Map SS s2FieldsOfStudy entries to arXiv-like category strings."""
    cats: list[str] = []
    seen: set[str] = set()
    for f in (s2_fields or []):
        mapped = _SS_TO_CATEGORY.get(f.get("category", ""))
        if mapped and mapped not in seen:
            cats.append(mapped)
            seen.add(mapped)
    return cats


def _is_cs_paper(entry: dict) -> bool:
    """Return True if any s2FieldsOfStudy category maps to a CS/ML arXiv category."""
    for f in (entry.get("s2FieldsOfStudy") or []):
        if f.get("category", "") in _SS_TO_CATEGORY:
            return True
    return False


# ---------------------------------------------------------------------------
# Convert SS paper response → standard paper dict
# ---------------------------------------------------------------------------

def _ss_entry_to_paper(entry: dict, arxiv_id: str = "") -> dict:
    """Convert a raw SS API paper object to our standard dict format."""
    if not arxiv_id:
        arxiv_id = (entry.get("externalIds") or {}).get("ArXiv") or ""

    return {
        "source_id":               f"ARXIV:{arxiv_id}" if arxiv_id else entry.get("paperId", ""),
        "arxiv_id":                arxiv_id,
        "ss_id":                   entry.get("paperId") or "",
        "title":                   entry.get("title") or "",
        "abstract":                entry.get("abstract") or "",
        "published":               entry.get("publicationDate") or "",
        "authors": [
            {
                "name":         a.get("name", ""),
                "hIndex":       0,
                "affiliations": [],
                "authorId":     a.get("authorId") or "",
            }
            for a in (entry.get("authors") or [])
        ],
        "categories":              _ss_fields_to_categories(entry.get("s2FieldsOfStudy")),
        "references": [
            {
                "title":   r.get("title") or "",
                "authors": [{"name": a.get("name", "")} for a in (r.get("authors") or [])],
            }
            for r in (entry.get("references") or [])
        ],
        "citationCount":           entry.get("citationCount") or 0,
        "influentialCitationCount": entry.get("influentialCitationCount") or 0,
    }


# ---------------------------------------------------------------------------
# URL → arXiv ID extraction & seed loading
# ---------------------------------------------------------------------------

_ARXIV_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)"
    r"|(?:^|[/:\s])arxiv[:\s](\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


def parse_arxiv_id(url_or_id: str) -> Optional[str]:
    """Extract bare arXiv ID from a URL. Returns None if not arXiv."""
    m = _ARXIV_RE.search(url_or_id)
    if m:
        raw = m.group(1) or m.group(2)
        return raw.split("v")[0]
    return None


def load_seed_urls(filepath: str) -> list[str]:
    """
    Read paper URLs (one per line), extract arXiv IDs.
    Non-arXiv URLs are skipped with a warning.
    Returns deduplicated list of bare arXiv IDs.
    """
    ids: list[str] = []
    seen: set[str] = set()
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            arxiv_id = parse_arxiv_id(url)
            if arxiv_id and arxiv_id not in seen:
                ids.append(arxiv_id)
                seen.add(arxiv_id)
                log.info("Resolved  %-55s  →  %s", url[:55], arxiv_id)
            elif not arxiv_id:
                log.warning("Skipping non-arXiv URL: %s", url)
    return ids


load_seed_ids = load_seed_urls


# ---------------------------------------------------------------------------
# Seed paper fetching — POST /paper/batch (1 call)
# ---------------------------------------------------------------------------

_SEED_FIELDS = (
    "paperId,externalIds,title,abstract,publicationDate,"
    "s2FieldsOfStudy,authors,"
    "references.title,references.authors,"
    "citationCount,influentialCitationCount"
)


def fetch_seed_papers(arxiv_ids: list[str]) -> list[dict]:
    """Fetch full seed paper metadata via SS /paper/batch (one call)."""
    if not arxiv_ids:
        return []

    papers: list[dict] = []
    for i in range(0, len(arxiv_ids), _SS_BATCH_LIMIT):
        chunk = arxiv_ids[i: i + _SS_BATCH_LIMIT]
        data = _ss_post(
            f"{_SS_BASE}/paper/batch",
            json={"ids": [f"ARXIV:{aid}" for aid in chunk]},
            params={"fields": _SEED_FIELDS},
        )
        if not data:
            continue
        for entry, aid in zip(data, chunk):
            if entry:
                papers.append(_ss_entry_to_paper(entry, aid))

    log.info("Fetched %d/%d seed papers from SS", len(papers), len(arxiv_ids))
    return papers


# ---------------------------------------------------------------------------
# Discovery — GET /paper/search/bulk (token-paginated, 1000/page)
# ---------------------------------------------------------------------------

DEFAULT_CATEGORIES = ["cs.LG", "cs.DB", "stat.ML", "cs.AI"]

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _build_search_query() -> str:
    """Build search query from keyword_weights in config.yaml."""
    keywords: list[str] = []
    if _DEFAULT_CONFIG.exists():
        with open(_DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        keywords = list((data.get("keyword_weights") or {}).keys())
    query = " | ".join(keywords) if keywords else "machine learning"
    log.info("Search query: %s", query)
    return query


_SEARCH_FIELDS = (
    "paperId,externalIds,title,abstract,publicationDate,"
    "s2FieldsOfStudy,authors,citationCount,influentialCitationCount"
)


def _search_recent(hours: int) -> list[dict]:
    """Discover papers via SS bulk keyword search with date filter."""
    query = _build_search_query()
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")

    params: dict = {
        "query":                  query,
        "fields":                 _SEARCH_FIELDS,
        "publicationDateOrYear":  f"{cutoff}:",
        "fieldsOfStudy":          "Computer Science",
    }

    all_papers: list[dict] = []
    token: str | None = None

    while True:
        if token:
            params["token"] = token

        data = _ss_get(f"{_SS_BASE}/paper/search/bulk", params=params)
        if not data:
            break

        for entry in (data.get("data") or []):
            all_papers.append(_ss_entry_to_paper(entry))

        token = data.get("token")
        if not token:
            break

        log.info("SS search: %d papers so far...", len(all_papers))

    log.info("SS search: %d papers found", len(all_papers))
    return all_papers


# ---------------------------------------------------------------------------
# Citation crawl — recent papers citing any seed
# ---------------------------------------------------------------------------

_CITATIONS_MAX_OFFSET = 9999   # SS citations endpoint hard limit


def _fetch_recent_citers(seed_papers: list[dict], hours: int) -> list[dict]:
    """
    For each seed paper, fetch its citing papers via GET /paper/{id}/citations.
    Keep only citers published within the lookback window.
    Paginates (500/page), stops at SS offset limit (9999).
    """
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")
    seen_ids: set[str] = set()
    citers: list[dict] = []

    for seed in seed_papers:
        ss_id = seed.get("ss_id")
        if not ss_id:
            continue

        seed_citers = 0
        offset = 0

        while offset < _CITATIONS_MAX_OFFSET:
            data = _ss_get(
                f"{_SS_BASE}/paper/{ss_id}/citations",
                params={"fields": _SEARCH_FIELDS, "offset": offset, "limit": 500},
            )
            if not data:
                break

            page = data.get("data") or []
            if not page:
                break

            for item in page:
                citing = item.get("citingPaper") or {}
                pid = citing.get("paperId") or ""
                pub_date = citing.get("publicationDate") or ""

                if not pid or pid in seen_ids:
                    continue

                if pub_date >= cutoff and _is_cs_paper(citing):
                    seen_ids.add(pid)
                    citers.append(_ss_entry_to_paper(citing))
                    seed_citers += 1

            next_offset = data.get("next")
            if next_offset is None or next_offset >= _CITATIONS_MAX_OFFSET:
                break
            offset = next_offset

        log.info("Citations: %d recent citers for '%s'",
                 seed_citers, seed.get("title", "")[:60])

    log.info("Citations: %d recent citing papers total", len(citers))
    return citers


# ---------------------------------------------------------------------------
# Author crawl — recent papers by target authors (h-index > 50 from seeds)
# ---------------------------------------------------------------------------

def _fetch_target_author_papers(seed_papers: list[dict], hours: int) -> list[dict]:
    """
    Identify target authors (h-index > 50) from seed papers, then fetch
    their recent papers via GET /author/{id}/papers.
    """
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")

    # Collect unique target author IDs
    target_authors: list[tuple[str, str]] = []   # (authorId, name)
    seen_aids: set[str] = set()
    for paper in seed_papers:
        for a in (paper.get("authors") or []):
            aid = a.get("authorId", "")
            h = a.get("hIndex") or 0
            if aid and h > 50 and aid not in seen_aids:
                seen_aids.add(aid)
                target_authors.append((aid, a.get("name", "")))

    if not target_authors:
        log.info("Author crawl: no target authors (h-index > 50) found in seeds")
        return []

    log.info("Author crawl: checking %d target authors", len(target_authors))

    papers: list[dict] = []
    seen_pids: set[str] = set()

    for aid, name in target_authors:
        offset = 0
        author_count = 0

        while True:
            data = _ss_get(
                f"{_SS_BASE}/author/{aid}/papers",
                params={"fields": _SEARCH_FIELDS, "offset": offset, "limit": 100},
            )
            if not data:
                break

            page = data.get("data") or []
            if not page:
                break

            all_old = True
            for item in page:
                pub_date = item.get("publicationDate") or ""
                pid = item.get("paperId") or ""

                if not pid or pid in seen_pids:
                    continue

                if pub_date >= cutoff:
                    all_old = False
                    if _is_cs_paper(item):
                        seen_pids.add(pid)
                        papers.append(_ss_entry_to_paper(item))
                        author_count += 1

            # Stop if all papers on this page are older than cutoff
            if all_old:
                break

            next_offset = data.get("next")
            if next_offset is None:
                break
            offset = next_offset

        log.info("Author crawl: %d recent papers by %s", author_count, name)

    log.info("Author crawl: %d recent papers total from %d target authors",
             len(papers), len(target_authors))
    return papers


# ---------------------------------------------------------------------------
# Combined discovery: search + citations + author crawl
# ---------------------------------------------------------------------------

def fetch_recent_papers(
    seed_papers: list[dict],
    hours: int = 24,
) -> list[dict]:
    """
    Discover recent papers via three channels:
      1. SS bulk keyword search (broad discovery).
      2. SS citation crawl (papers citing any seed, within time window).
      3. SS author crawl (recent papers by target authors with h-index > 50).
    Results are merged and deduplicated.
    """
    # Channel 1: keyword search from config.yaml
    search_papers = _search_recent(hours)

    # Channel 2: citation crawl
    citer_papers = _fetch_recent_citers(seed_papers, hours)

    # Channel 3: target author recent papers
    author_papers = _fetch_target_author_papers(seed_papers, hours)

    # Merge + deduplicate (prefer search version if duplicate)
    seen: set[str] = set()
    merged: list[dict] = []

    for p in search_papers:
        key = p.get("ss_id") or p.get("arxiv_id") or ""
        if key and key not in seen:
            seen.add(key)
            merged.append(p)

    added_from_citations = 0
    for p in citer_papers:
        key = p.get("ss_id") or p.get("arxiv_id") or ""
        if key and key not in seen:
            seen.add(key)
            merged.append(p)
            added_from_citations += 1

    added_from_authors = 0
    for p in author_papers:
        key = p.get("ss_id") or p.get("arxiv_id") or ""
        if key and key not in seen:
            seen.add(key)
            merged.append(p)
            added_from_authors += 1

    # Remove seeds
    seed_arxiv = {p.get("arxiv_id") for p in seed_papers if p.get("arxiv_id")}
    seed_ss    = {p.get("ss_id") for p in seed_papers if p.get("ss_id")}
    candidates = [
        p for p in merged
        if p.get("arxiv_id", "") not in seed_arxiv
        and p.get("ss_id", "") not in seed_ss
    ]

    log.info("Discovery: %d search + %d citations + %d authors → %d candidates",
             len(search_papers), added_from_citations, added_from_authors,
             len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Batch enrichment — references + author h-index/affiliations
# ---------------------------------------------------------------------------

def enrich_with_ss(papers: list[dict]) -> None:
    """
    Batch-enrich papers in place:
      - POST /paper/batch  → references (500 IDs/call)
      - POST /author/batch → h-index + affiliations (1000 IDs/call)

    Papers must have either 'ss_id' or 'arxiv_id' set.
    """
    # Build query ID list (prefer ss_id, fall back to ARXIV:id)
    query_ids: list[tuple[int, str]] = []   # (index, query_id)
    for i, p in enumerate(papers):
        qid = p.get("ss_id") or (f"ARXIV:{p['arxiv_id']}" if p.get("arxiv_id") else "")
        if qid:
            query_ids.append((i, qid))

    if not query_ids:
        return

    # --- Step 1: batch fetch references ---
    ref_fields = "paperId,references.title,references.authors"
    ref_map: dict[str, dict] = {}
    id_list = [qid for _, qid in query_ids]

    for i in range(0, len(id_list), _SS_BATCH_LIMIT):
        chunk = id_list[i: i + _SS_BATCH_LIMIT]
        data = _ss_post(
            f"{_SS_BASE}/paper/batch",
            json={"ids": chunk},
            params={"fields": ref_fields},
        )
        if not data:
            continue
        for entry, qid in zip(data, chunk):
            if entry:
                ref_map[qid] = entry

    for idx, qid in query_ids:
        entry = ref_map.get(qid)
        if not entry:
            continue
        papers[idx]["references"] = [
            {
                "title":   r.get("title") or "",
                "authors": [{"name": a.get("name", "")} for a in (r.get("authors") or [])],
            }
            for r in (entry.get("references") or [])
        ]

    # --- Step 2: batch fetch author details ---
    all_author_ids: set[str] = set()
    for p in papers:
        for a in (p.get("authors") or []):
            if a.get("authorId"):
                all_author_ids.add(a["authorId"])

    author_map: dict[str, dict] = {}
    if all_author_ids:
        aid_list = list(all_author_ids)
        for i in range(0, len(aid_list), _SS_AUTHOR_BATCH):
            chunk = aid_list[i: i + _SS_AUTHOR_BATCH]
            data = _ss_post(
                f"{_SS_BASE}/author/batch",
                json={"ids": chunk},
                params={"fields": "authorId,hIndex,affiliations"},
            )
            if not data:
                continue
            for entry in data:
                if entry and entry.get("authorId"):
                    author_map[entry["authorId"]] = {
                        "hIndex":       entry.get("hIndex") or 0,
                        "affiliations": entry.get("affiliations") or [],
                    }

    for p in papers:
        for a in (p.get("authors") or []):
            details = author_map.get(a.get("authorId", ""))
            if details:
                a["hIndex"]       = details["hIndex"]
                a["affiliations"] = details["affiliations"]

    log.info("Enriched %d papers: %d with refs, %d authors resolved",
             len(papers), len(ref_map), len(author_map))
