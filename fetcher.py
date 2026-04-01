"""
fetcher.py — Semantic Scholar paper discovery and enrichment.

All API calls use SS batch/bulk endpoints for efficiency:
  - Seeds:     POST /paper/batch       (1 call for all seeds)
  - Discovery: GET  /paper/search/bulk (token-paginated, 1000/page)
  - Enrich:    POST /paper/batch       (references, 500/call)
               POST /author/batch      (h-index, 1000/call)

The three discovery channels (keyword search, citation crawl, author crawl)
run concurrently via ThreadPoolExecutor.  A shared rate limiter serialises
the actual HTTP calls to stay within Semantic Scholar's rate limits while
overlapping I/O wait with result processing to minimise total wall time.

Typical run (4 seeds, 500 candidates): ~8 SS API calls total.
"""

import os
import re
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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
_SS_LEEWAY_KEY   = 1.0    # seconds between calls (with API key)
_SS_LEEWAY_FREE  = 3.5    # seconds between calls (no API key — stricter limits)
_SS_MAX_429      = 8       # max consecutive 429s before giving up

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


def _leeway() -> float:
    return _SS_LEEWAY_KEY if _SS_API_KEY else _SS_LEEWAY_FREE


# ---------------------------------------------------------------------------
# Thread-safe rate limiter — only sleeps the *remaining* gap since last call.
# When running discovery channels concurrently this eliminates wasted idle
# time: while one thread processes results, another can issue its request.
# ---------------------------------------------------------------------------

class _RateLimiter:
    __slots__ = ("_lock", "_last_call")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        gap = _leeway()
        with self._lock:
            now = time.monotonic()
            remaining = gap - (now - self._last_call)
            if remaining > 0:
                time.sleep(remaining)
            self._last_call = time.monotonic()

_rate_limiter = _RateLimiter()

# Per-thread sessions — requests.Session is NOT thread-safe, so each
# thread gets its own session (still benefits from connection pooling
# within that thread's sequence of calls).
_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


# ---------------------------------------------------------------------------
# HTTP helpers with retry + 429 handling + leeway
#
# 429 rate limits do NOT consume error retries — they are expected behaviour
# and the code will keep retrying with increasing backoff up to _SS_MAX_429
# consecutive 429s before giving up.
# ---------------------------------------------------------------------------

def _ss_get(url: str, params: dict | None = None, retries: int = 4):
    """GET with retry, 429 back-off, and rate-limited pacing."""
    gap = _leeway()
    attempt = 0
    rate_hits = 0
    while attempt < retries:
        try:
            _rate_limiter.wait()
            resp = _get_session().get(
                url, params=params, headers=_ss_headers(),
                timeout=(10, 90),   # (connect, read)
            )
            if resp.status_code == 429:
                rate_hits += 1
                if rate_hits > _SS_MAX_429:
                    log.error("SS rate limit: %d consecutive 429s, giving up", _SS_MAX_429)
                    return None
                wait = int(resp.headers.get("Retry-After", 5)) + gap * rate_hits
                log.warning("SS rate limited (GET); waiting %ds (%d/%d)",
                            int(wait), rate_hits, _SS_MAX_429)
                time.sleep(wait)
                continue   # retry WITHOUT incrementing attempt
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            attempt += 1
            log.warning("SS GET error (attempt %d/%d): %s",
                        attempt, retries, exc)
            time.sleep(2 ** attempt + gap)
    return None


def _ss_post(url: str, json: dict, params: dict | None = None, retries: int = 4):
    """POST with retry, 429 back-off, and rate-limited pacing."""
    gap = _leeway()
    attempt = 0
    rate_hits = 0
    while attempt < retries:
        try:
            _rate_limiter.wait()
            resp = _get_session().post(
                url, json=json, params=params, headers=_ss_headers(),
                timeout=(10, 180),   # (connect, read) — batch POSTs need more time
            )
            if resp.status_code == 429:
                rate_hits += 1
                if rate_hits > _SS_MAX_429:
                    log.error("SS rate limit: %d consecutive 429s, giving up", _SS_MAX_429)
                    return None
                wait = int(resp.headers.get("Retry-After", 5)) + gap * rate_hits
                log.warning("SS rate limited (POST); waiting %ds (%d/%d)",
                            int(wait), rate_hits, _SS_MAX_429)
                time.sleep(wait)
                continue   # retry WITHOUT incrementing attempt
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            attempt += 1
            log.warning("SS POST error (attempt %d/%d): %s",
                        attempt, retries, exc)
            time.sleep(2 ** attempt + gap)
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

    # Extract venue name from publicationVenue (structured) or venue (string)
    pub_venue = entry.get("publicationVenue") or {}
    venue = pub_venue.get("name") or entry.get("venue") or ""

    return {
        "source_id":               f"ARXIV:{arxiv_id}" if arxiv_id else entry.get("paperId", ""),
        "arxiv_id":                arxiv_id,
        "ss_id":                   entry.get("paperId") or "",
        "title":                   entry.get("title") or "",
        "abstract":                entry.get("abstract") or "",
        "published":               entry.get("publicationDate") or "",
        "venue":                   venue,
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
                "paperId": r.get("paperId") or "",
                "arxivId": (r.get("externalIds") or {}).get("ArXiv") or "",
                "title":   r.get("title") or "",
                "authors": [
                    {"name": a.get("name", ""), "authorId": a.get("authorId") or ""}
                    for a in (r.get("authors") or [])
                ],
            }
            for r in (entry.get("references") or [])
        ],
        "citationCount":           entry.get("citationCount") or 0,
        "influentialCitationCount": entry.get("influentialCitationCount") or 0,
    }


# ---------------------------------------------------------------------------
# URL → SS paper ID resolution & seed loading
# ---------------------------------------------------------------------------

_ARXIV_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)"
    r"|(?:^|[/:\s])arxiv[:\s](\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)

_DOI_RE = re.compile(r"(10\.\d{4,}/[^\s?#]+)")


def parse_arxiv_id(url_or_id: str) -> Optional[str]:
    """Extract bare arXiv ID from a URL. Returns None if not arXiv."""
    m = _ARXIV_RE.search(url_or_id)
    if m:
        raw = m.group(1) or m.group(2)
        return raw.split("v")[0]
    return None


def _try_arxiv(url: str) -> Optional[str]:
    """Try to extract ARXIV:id from an arXiv URL or arxiv: prefix."""
    m = _ARXIV_RE.search(url)
    if m:
        raw = m.group(1) or m.group(2)
        return f"ARXIV:{raw.split('v')[0]}"
    return None


def _try_doi_in_url(url: str) -> Optional[str]:
    """Try to extract DOI from a URL that contains a 10.xxxx/ pattern."""
    m = _DOI_RE.search(url)
    if m:
        doi = m.group(1).rstrip("/")
        return f"DOI:{doi}"
    return None


def _try_publisher_url(url: str) -> Optional[str]:
    """
    Construct a DOI from known publisher URL patterns where the DOI
    is NOT explicitly present in the URL.
    """
    # Nature: nature.com/articles/{articleId} → DOI:10.1038/{articleId}
    m = re.search(r"nature\.com/articles/([\w.-]+)", url, re.I)
    if m:
        return f"DOI:10.1038/{m.group(1)}"

    # Science.org: science.org/doi/{doi}
    m = re.search(r"science\.org/doi/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        return f"DOI:{m.group(1)}"

    # PNAS: pnas.org/doi/{abs|full}/{doi}
    m = re.search(r"pnas\.org/doi/(?:abs|full)/(10\.\d{4,}/[^\s?#]+)", url, re.I)
    if m:
        return f"DOI:{m.group(1)}"

    # Cell Press: cell.com/{journal}/fulltext/{PII} — resolve via DOI prefix
    m = re.search(r"cell\.com/[^/]+/fulltext/(S[\d-]+)", url, re.I)
    if m:
        return f"DOI:10.1016/{m.group(1)}"

    return None


def _try_pubmed(url: str) -> Optional[str]:
    """Try to extract PMID or PMCID from PubMed/PMC URLs."""
    m = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url, re.I)
    if m:
        return f"PMID:{m.group(1)}"
    m = re.search(r"ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)", url, re.I)
    if m:
        return f"PMCID:{m.group(1)}"
    return None


def resolve_url_to_ss_paper_id(url: str) -> Optional[str]:
    """
    Resolve a URL to a Semantic Scholar paper ID.

    Strategies (in priority order):
      1. arXiv URL/prefix   → ARXIV:id
      2. DOI in URL          → DOI:10.xxxx/...
      3. Publisher pattern   → DOI:10.xxxx/...  (Nature, Science, PNAS, Cell)
      4. PubMed/PMC URL      → PMID:xxx / PMCID:PMCxxx
    """
    url = url.strip()
    for strategy in (_try_arxiv, _try_doi_in_url, _try_publisher_url, _try_pubmed):
        result = strategy(url)
        if result:
            return result
    return None


def _is_recent(date_str: str, cutoff: datetime) -> bool:
    """Return True if date_str (YYYY-MM-DD) is at or after the cutoff datetime."""
    if not date_str:
        return False
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt >= cutoff
    except ValueError:
        return False


def load_seed_urls(filepath: str) -> list[str]:
    """
    Read paper URLs/IDs (one per line), resolve to SS paper IDs.
    Supports arXiv, DOI, Nature, PubMed, PMC, Science.org, PNAS, Cell, and
    any URL containing a DOI (e.g. Springer, Wiley, IEEE, ACM).
    Returns deduplicated list of SS paper IDs.
    """
    ids: list[str] = []
    seen: set[str] = set()
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            paper_id = resolve_url_to_ss_paper_id(url)
            if paper_id and paper_id not in seen:
                ids.append(paper_id)
                seen.add(paper_id)
                log.info("Resolved  %-55s  →  %s", url[:55], paper_id)
            elif not paper_id:
                log.warning("Could not resolve URL: %s", url)
    return ids


load_seed_ids = load_seed_urls


# ---------------------------------------------------------------------------
# Seed paper fetching — POST /paper/batch (1 call)
# ---------------------------------------------------------------------------

_SEED_FIELDS = (
    "paperId,externalIds,title,abstract,publicationDate,"
    "s2FieldsOfStudy,authors,venue,publicationVenue,"
    "references.paperId,references.title,references.authors,references.externalIds,"
    "citationCount,influentialCitationCount"
)


def fetch_seed_papers(paper_ids: list[str]) -> list[dict]:
    """
    Fetch full seed paper metadata via SS /paper/batch.
    Accepts any SS-compatible paper IDs (ARXIV:, DOI:, PMID:, PMCID:, etc.).
    """
    if not paper_ids:
        return []

    papers: list[dict] = []
    for i in range(0, len(paper_ids), _SS_BATCH_LIMIT):
        chunk = paper_ids[i: i + _SS_BATCH_LIMIT]
        data = _ss_post(
            f"{_SS_BASE}/paper/batch",
            json={"ids": chunk},
            params={"fields": _SEED_FIELDS},
        )
        if not data:
            continue
        for entry in data:
            if entry:
                papers.append(_ss_entry_to_paper(entry))

    log.info("Fetched %d/%d seed papers from SS", len(papers), len(paper_ids))
    return papers


# ---------------------------------------------------------------------------
# Discovery — GET /paper/search/bulk (token-paginated, 1000/page)
# ---------------------------------------------------------------------------

DEFAULT_CATEGORIES = ["cs.LG", "cs.DB", "stat.ML", "cs.AI"]

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _build_search_query(config_path: str | None = None) -> str:
    """Build search query from keywords list in config.yaml."""
    cfg_path = Path(config_path) if config_path else _DEFAULT_CONFIG
    keywords: list[str] = []
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        keywords = data.get("keywords") or []
    query = " | ".join(keywords) if keywords else "machine learning"
    log.info("Search query: %s", query)
    return query


_SEARCH_FIELDS = (
    "paperId,externalIds,title,abstract,publicationDate,"
    "s2FieldsOfStudy,authors,venue,publicationVenue,"
    "citationCount,influentialCitationCount"
)


def _search_recent(hours: int, config_path: str | None = None) -> list[dict]:
    """Discover papers via SS bulk keyword search with date filter."""
    query = _build_search_query(config_path)
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


def _fetch_recent_citers(
    seed_papers: list[dict], hours: int,
) -> tuple[list[dict], dict[str, int]]:
    """
    For each seed paper, fetch its citing papers via GET /paper/{id}/citations.
    Keep only citers published within the lookback window.

    Returns:
      citers      — deduplicated list of citing paper dicts
      cites_count — mapping paperId → number of distinct seeds cited
    """
    cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%d")
    seen_ids: set[str] = set()
    citers: list[dict] = []
    cites_count: dict[str, int] = {}

    for seed in seed_papers:
        ss_id = seed.get("ss_id")
        if not ss_id:
            continue

        seed_citers = 0
        offset = 0

        while offset < _CITATIONS_MAX_OFFSET:
            remaining = _CITATIONS_MAX_OFFSET - offset
            page_limit = min(500, remaining)
            data = _ss_get(
                f"{_SS_BASE}/paper/{ss_id}/citations",
                params={"fields": _SEARCH_FIELDS, "offset": offset, "limit": page_limit},
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

                if not pid:
                    continue

                if pub_date >= cutoff and _is_cs_paper(citing):
                    # Track how many seeds this paper cites (before dedup)
                    cites_count[pid] = cites_count.get(pid, 0) + 1

                    if pid not in seen_ids:
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
    return citers, cites_count


# ---------------------------------------------------------------------------
# Author crawl — recent papers by target authors (h-index > 20 from seeds)
# ---------------------------------------------------------------------------

def _fetch_target_author_papers(seed_papers: list[dict], hours: int) -> list[dict]:
    """
    Identify target authors (h-index > 20) from seed papers, then fetch
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
            if aid and h > 20 and aid not in seen_aids:
                seen_aids.add(aid)
                target_authors.append((aid, a.get("name", "")))

    if not target_authors:
        log.info("Author crawl: no target authors (h-index > 20) found in seeds")
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
    config_path: str | None = None,
) -> list[dict]:
    """
    Discover recent papers via three channels:
      1. SS bulk keyword search (broad discovery).
      2. SS citation crawl (papers citing any seed, within time window).
      3. SS author crawl (recent papers by target authors with h-index > 20).
    Results are merged and deduplicated.
    """
    # Run all three discovery channels concurrently.
    # The shared _rate_limiter serialises HTTP calls to respect the API
    # rate limit, while threads overlap I/O wait with result processing.
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="discovery") as pool:
        f_search = pool.submit(_search_recent, hours, config_path)
        f_cite   = pool.submit(_fetch_recent_citers, seed_papers, hours)
        f_author = pool.submit(_fetch_target_author_papers, seed_papers, hours)

        search_papers = f_search.result()
        citer_papers, cites_count = f_cite.result()
        author_papers = f_author.result()

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

    # Annotate papers with seed citation counts from the citation crawl
    for p in candidates:
        pid = p.get("ss_id") or ""
        if pid in cites_count:
            p["_cites_seed_count"] = cites_count[pid]

    log.info("Discovery: %d search + %d citations + %d authors → %d candidates",
             len(search_papers), added_from_citations, added_from_authors,
             len(candidates))
    return candidates


# ---------------------------------------------------------------------------
# Batch enrichment — references + author h-index/affiliations
# ---------------------------------------------------------------------------

def enrich_with_ss(papers: list[dict], skip_references: bool = False) -> None:
    """
    Batch-enrich papers in place:
      - POST /paper/batch  → references (500 IDs/call)  [unless skip_references]
      - POST /author/batch → h-index + affiliations (1000 IDs/call)

    Papers must have either 'ss_id' or 'arxiv_id' set.
    Use skip_references=True for candidates (references are not used in scoring).
    """
    # Build query ID list (prefer ss_id, fall back to ARXIV:id)
    query_ids: list[tuple[int, str]] = []   # (index, query_id)
    for i, p in enumerate(papers):
        qid = p.get("ss_id") or (f"ARXIV:{p['arxiv_id']}" if p.get("arxiv_id") else "")
        if qid:
            query_ids.append((i, qid))

    if not query_ids:
        return

    # --- Step 1: batch fetch references (skippable) ---
    ref_count = 0
    if not skip_references:
        ref_fields = "paperId,references.paperId,references.title,references.authors,references.externalIds"
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
                    "paperId": r.get("paperId") or "",
                    "arxivId": (r.get("externalIds") or {}).get("ArXiv") or "",
                    "title":   r.get("title") or "",
                    "authors": [
                        {"name": a.get("name", ""), "authorId": a.get("authorId") or ""}
                        for a in (r.get("authors") or [])
                    ],
                }
                for r in (entry.get("references") or [])
            ]
        ref_count = len(ref_map)

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
             len(papers), ref_count, len(author_map))
