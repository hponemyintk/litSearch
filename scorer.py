"""
scorer.py — semantic similarity + metadata-driven paper scoring engine.

Scores a paper using 8 heuristics (point values configurable via config.yaml):

  Relevance:
    1. Semantic Similarity          — embedding cosine sim to seeds      (35)
    2. Author in Seed Refs          — candidate authorId in seed refs    (10)
    3. Cites Seed                   — paper directly cites seed papers   (15)

  Quality:
    4. Author Authority             — logarithmic h-index               (9)
    5. Citation Velocity            — citations/day + recency prior      (12)

  Bonuses:
    6. Institutional Authority      — affiliation check                  (5)
    7. Category Intersection        — dynamic seed-category overlap      (5)
    8. Benchmark Specificity        — dataset names in abstract          (9)

Call build_scoring_config(seed_papers, window_hours) once per session,
then precompute_similarities(candidates, config) before scoring,
then pass the config to score_paper() / rank_papers().
"""

import re
import math
import copy
import pickle
import logging
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

import yaml
import numpy as np
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridden by config.yaml scoring/institutions/benchmarks)
# ---------------------------------------------------------------------------

_DEFAULT_SCORING = {
    "max_semantic": 35,
    "max_seed_ref_author": 10,
    "seed_ref_per_author": 4,
    "max_cites_seed": 15,
    "cites_seed_per_seed": 5,
    "max_authority": 9,
    "max_citation_vel": 12,
    "flat_institution": 5,
    "flat_cross_cat": 5,
    "max_benchmark": 9,
}

_DEFAULT_INSTITUTIONS: list[str] = [
    "Stanford", "MIT", "Berkeley", "Carnegie Mellon", "CMU",
    "Princeton", "Harvard", "Cornell", "Yale", "Columbia",
    "University of Washington", "Georgia Tech", "UCLA", "NYU",
    "University of Michigan", "UIUC", "University of Illinois",
    "Caltech",
    "Oxford", "Cambridge", "UCL", "Imperial College", "Edinburgh",
    "ETH", "EPFL", "Tübingen", "Max Planck",
    "TU Munich", "University of Amsterdam", "INRIA",
    "Mila", "University of Toronto", "University of Montreal", "McGill",
    "Tsinghua", "Peking University", "KAIST", "University of Tokyo",
    "NUS", "National University of Singapore", "HKUST",
    "Google", "DeepMind", "Meta", "FAIR",
    "Microsoft Research", "Microsoft",
    "OpenAI", "Anthropic", "Apple",
    "NVIDIA", "Amazon", "IBM Research",
    "Allen Institute", "AI2",
    "Hugging Face", "HuggingFace",
    "Alibaba", "DAMO", "Baidu", "ByteDance", "Samsung",
    "Tesla",
]

_DEFAULT_BENCHMARKS: list[str] = [
    "RelBench", "OGB", "Open Graph Benchmark", "OpenML",
    "TabZilla", "WILDS",
]

_EMBED_MODEL = "all-MiniLM-L12-v2"
_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"
_CACHE_DIR = Path(__file__).parent / "cache"
_CACHE_FILE = _CACHE_DIR / "embedding_cache.pkl"

# Increment when embedding model or text format changes to invalidate stale cache
_CACHE_VERSION = 3

# Papers with fewer abstract words get a neutral similarity score
_MIN_ABSTRACT_WORDS = 10


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

def _load_config_file(path: Path | str | None = None) -> dict:
    p = Path(path) if path else _DEFAULT_CONFIG
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _load_embedding_cache() -> dict[str, np.ndarray]:
    if _CACHE_FILE.exists():
        try:
            with open(_CACHE_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and data.get("_version") == _CACHE_VERSION:
                return data.get("embeddings", {})
            log.info("Embedding cache version mismatch, rebuilding")
        except Exception:
            log.warning("Failed to load embedding cache, starting fresh")
    return {}


def _save_embedding_cache(cache: dict[str, np.ndarray]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "wb") as f:
        pickle.dump({"_version": _CACHE_VERSION, "embeddings": cache}, f)


# ---------------------------------------------------------------------------
# Institution regex builder (word-boundary matching)
# ---------------------------------------------------------------------------

def _build_institution_pattern(institutions: list[str]) -> re.Pattern:
    escaped = [re.escape(inst) for inst in institutions]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Embedding text builder
# ---------------------------------------------------------------------------

def _build_embed_text(paper: dict) -> str:
    """Title repeated 3x for emphasis, followed by abstract."""
    title = paper.get("title") or ""
    abstract = paper.get("abstract") or ""
    return f"{title}. {title}. {title}. {abstract}"


# ---------------------------------------------------------------------------
# Config builder — call once per session
# ---------------------------------------------------------------------------

def build_scoring_config(
    seed_papers: list[dict[str, Any]],
    window_hours: int = 24,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Build dynamic scoring config from seed papers + config.yaml.

    Loads the embedding model, encodes seeds, auto-calibrates the
    similarity threshold from inter-seed cosine similarity, and
    extracts author IDs from seed references.
    """
    file_cfg = _load_config_file(config_path)

    # Scoring constants: config.yaml overrides defaults
    sc = {**_DEFAULT_SCORING, **file_cfg.get("scoring", {})}

    institutions = file_cfg.get("institutions", _DEFAULT_INSTITUTIONS)
    benchmarks = file_cfg.get("benchmarks", _DEFAULT_BENCHMARKS)

    # Load embedding model
    model = SentenceTransformer(_EMBED_MODEL)

    # Encode seed papers (title 3x)
    seed_texts = [_build_embed_text(p) for p in seed_papers]
    seed_embeddings = (
        model.encode(seed_texts, normalize_embeddings=True, show_progress_bar=False)
        if seed_texts
        else np.empty((0, 0))
    )

    # Auto-calibrate similarity threshold from inter-seed similarity
    if len(seed_embeddings) >= 2:
        seed_sims = seed_embeddings @ seed_embeddings.T
        mask = np.triu(np.ones(seed_sims.shape, dtype=bool), k=1)
        sim_threshold = float(seed_sims[mask].mean())
    else:
        sim_threshold = 0.5

    log.info(
        "Similarity threshold auto-calibrated: %.3f (from %d seeds)",
        sim_threshold, len(seed_embeddings),
    )

    # Compute max total dynamically from scoring constants
    max_total = (
        sc["max_semantic"] + sc["max_seed_ref_author"] + sc["max_cites_seed"]
        + sc["max_authority"] + sc["max_citation_vel"]
        + sc["flat_institution"] + sc["flat_cross_cat"] + sc["max_benchmark"]
    )

    # Extract distinct category groups from seeds for dynamic intersection
    seed_cat_groups = _extract_seed_category_groups(seed_papers)
    log.info(
        "Category groups from seeds: %d distinct (%s)",
        len(seed_cat_groups),
        ", ".join(str(g) for g in seed_cat_groups),
    )

    return {
        # model + embeddings
        "_model":            model,
        "_seed_embeddings":  seed_embeddings,
        "_sim_threshold":    sim_threshold,
        "_institution_re":   _build_institution_pattern(institutions),
        # author matching (by authorId)
        "seed_ref_author_ids": _extract_seed_ref_author_ids(seed_papers),
        # category intersection (dynamic)
        "_seed_category_groups": seed_cat_groups,
        # window
        "window_hours":      window_hours,
        # scoring constants (from config.yaml with defaults)
        **sc,
        "max_total":         max_total,
        # lists
        "benchmarks":        benchmarks,
    }


# -- private helpers ---------------------------------------------------------

def _extract_seed_ref_author_ids(seed_papers: list[dict]) -> set[str]:
    """Collect unique authorIds from all seed papers' reference lists."""
    ids: set[str] = set()
    for paper in seed_papers:
        for ref in (paper.get("references") or []):
            for author in (ref.get("authors") or []):
                aid = (author.get("authorId") or "").strip()
                if aid:
                    ids.add(aid)
    return ids


def _extract_seed_category_groups(seed_papers: list[dict]) -> list[set[str]]:
    """
    Extract distinct category sets from seed papers.

    Each seed contributes its set of arXiv-like categories.
    Duplicate category sets are collapsed so that e.g. four seeds all tagged
    {cs.LG, cs.AI} count as one group, not four.

    Used by score_category_intersection to reward candidates that bridge
    2+ distinct seed category groups.
    """
    groups: list[set[str]] = []
    seen: set[frozenset[str]] = set()
    for p in seed_papers:
        cats = frozenset(p.get("categories") or [])
        if cats and cats not in seen:
            seen.add(cats)
            groups.append(set(cats))
    return groups


# ---------------------------------------------------------------------------
# Pre-computation — call once before scoring
# ---------------------------------------------------------------------------

def precompute_similarities(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """
    Batch-encode candidate papers and compute max cosine similarity
    to any seed paper.  Uses embedding cache for previously seen papers.
    Papers with no/very short abstract get a neutral similarity value.
    Stores '_semantic_sim' in each paper dict.
    """
    model = config.get("_model")
    seed_emb = config.get("_seed_embeddings")
    threshold = config.get("_sim_threshold", 0.5)

    if model is None or seed_emb is None or seed_emb.size == 0 or not candidates:
        return

    dim = seed_emb.shape[1]
    cache = _load_embedding_cache()

    # Split into cached vs needs-encoding
    need_encoding: list[int] = []
    cached_embs: dict[int, np.ndarray] = {}

    for i, p in enumerate(candidates):
        pid = p.get("ss_id") or p.get("arxiv_id") or ""
        if pid and pid in cache:
            cached_embs[i] = cache[pid]
        else:
            need_encoding.append(i)

    log.info("Embeddings: %d cached, %d to encode", len(cached_embs), len(need_encoding))

    # Encode uncached papers
    new_embs: dict[int, np.ndarray] = {}
    if need_encoding:
        texts = [_build_embed_text(candidates[i]) for i in need_encoding]
        encoded = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        for idx, emb in zip(need_encoding, encoded):
            new_embs[idx] = emb
            pid = candidates[idx].get("ss_id") or candidates[idx].get("arxiv_id") or ""
            if pid:
                cache[pid] = emb

    # Save updated cache
    if need_encoding:
        _save_embedding_cache(cache)

    # Build full embedding matrix and compute similarities
    cand_emb = np.zeros((len(candidates), dim), dtype=np.float32)
    for i in range(len(candidates)):
        if i in cached_embs:
            cand_emb[i] = cached_embs[i]
        elif i in new_embs:
            cand_emb[i] = new_embs[i]

    sims = cand_emb @ seed_emb.T
    max_sims = sims.max(axis=1)

    for paper, sim in zip(candidates, max_sims):
        abstract = paper.get("abstract") or ""
        if len(abstract.split()) < _MIN_ABSTRACT_WORDS:
            # Unreliable embedding — assign neutral score (half threshold)
            paper["_semantic_sim"] = threshold * 0.5
        else:
            paper["_semantic_sim"] = float(sim)


# ---------------------------------------------------------------------------
# Heuristic 1: Semantic Similarity
# ---------------------------------------------------------------------------

def score_semantic_similarity(paper: dict[str, Any], config: dict[str, Any]) -> int:
    """
    Score from pre-computed cosine similarity to seed papers.
    Uses a softer clip: papers must reach 1.3x the inter-seed mean
    similarity to earn full score, giving headroom to differentiate
    above-threshold papers.
    """
    sim = paper.get("_semantic_sim", 0.0)
    if sim <= 0:
        return 0
    threshold = config.get("_sim_threshold", 0.5)
    max_pts = config.get("max_semantic", _DEFAULT_SCORING["max_semantic"])
    normalized = min(sim / (threshold * 1.3), 1.0)
    return round(normalized * max_pts)


# ---------------------------------------------------------------------------
# Heuristic 2: Author in Seed References (by authorId)
# ---------------------------------------------------------------------------

def score_author_in_seed_refs(
    authors: list[dict[str, Any]],
    config: dict[str, Any],
) -> int:
    """
    Award points per candidate author whose SS authorId appears
    in the seed papers' reference author lists.
    """
    seed_ref_ids = config.get("seed_ref_author_ids") or set()
    if not seed_ref_ids:
        return 0
    per_author = config.get("seed_ref_per_author", _DEFAULT_SCORING["seed_ref_per_author"])
    max_pts = config.get("max_seed_ref_author", _DEFAULT_SCORING["max_seed_ref_author"])
    total = 0
    for author in authors:
        aid = (author.get("authorId") or "").strip()
        if aid and aid in seed_ref_ids:
            total += per_author
    return min(total, max_pts)


# ---------------------------------------------------------------------------
# Heuristic 3: Cites Seed Papers
# ---------------------------------------------------------------------------

def score_cites_seed(paper: dict[str, Any], config: dict[str, Any]) -> int:
    """
    Award points if this paper directly cites one or more seed papers.
    Detected during the citation crawl discovery channel.
    """
    count = paper.get("_cites_seed_count", 0)
    if count <= 0:
        return 0
    per_seed = config.get("cites_seed_per_seed", _DEFAULT_SCORING["cites_seed_per_seed"])
    max_pts = config.get("max_cites_seed", _DEFAULT_SCORING["max_cites_seed"])
    return min(count * per_seed, max_pts)


# ---------------------------------------------------------------------------
# Heuristic 4: Author Authority (logarithmic h-index)
# ---------------------------------------------------------------------------

def score_author_authority(
    authors: list[dict[str, Any]],
    config: dict[str, Any],
) -> int:
    """
    score = round(max_pts * log(max_h + 1) / log(101)), capped.
    Calibration at max_pts=10: h=10->5, h=25->7, h=50->9, h=100->10.
    """
    if not authors:
        return 0
    max_h = max((a.get("hIndex") or 0) for a in authors)
    if max_h <= 0:
        return 0
    max_pts = config.get("max_authority", _DEFAULT_SCORING["max_authority"])
    return min(round(max_pts * math.log(max_h + 1) / math.log(101)), max_pts)


# ---------------------------------------------------------------------------
# Heuristic 5: Citation Velocity (with window-scaled recency prior)
# ---------------------------------------------------------------------------

def score_citation_velocity(
    paper: dict[str, Any],
    config: dict[str, Any],
    _now: Optional[datetime] = None,
) -> int:
    """
    Combines citation velocity with a recency prior scaled to the
    lookback window.

    Recency prior (benefit of the doubt for new papers):
      < 15% of window  ->  base 5 pts
      < 35% of window  ->  base 3 pts
      >= 35%            ->  base 0 pts (must earn via citations)

    Citation velocity bins (effective citations/day):
      >= 5.0 -> 10    >= 0.5 -> 4
      >= 2.0 -> 8     >= 0.2 -> 3
      >= 1.0 -> 6     >= 0.1 -> 2

    Final score = max(base, velocity_score), capped.
    Influential citations count 3x (2x bonus on top of base count).
    """
    published   = paper.get("published") or ""
    citations   = paper.get("citationCount") or 0
    influential = paper.get("influentialCitationCount") or 0
    effective   = citations + influential * 2
    max_pts     = config.get("max_citation_vel", _DEFAULT_SCORING["max_citation_vel"])
    window_hours = config.get("window_hours", 168)

    if not published:
        return 0

    try:
        pub_dt = datetime.strptime(published[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        now = _now or datetime.now(tz=timezone.utc)
        age_days = (now - pub_dt).total_seconds() / 86400
    except ValueError:
        return 0

    if age_days < 0:
        return 0

    # Scale recency thresholds to window size
    window_days = window_hours / 24
    fresh_cutoff = window_days * 0.15
    mid_cutoff = window_days * 0.35

    if age_days < fresh_cutoff:
        base = 5
    elif age_days < mid_cutoff:
        base = 3
    else:
        base = 0

    # Citation velocity
    vel_score = 0
    if age_days >= 1 and effective > 0:
        velocity = effective / age_days
        if velocity >= 5.0:
            vel_score = 10
        elif velocity >= 2.0:
            vel_score = 8
        elif velocity >= 1.0:
            vel_score = 6
        elif velocity >= 0.5:
            vel_score = 4
        elif velocity >= 0.2:
            vel_score = 3
        elif velocity >= 0.1:
            vel_score = 2

    return min(max(base, vel_score), max_pts)


# ---------------------------------------------------------------------------
# Heuristic 6: Institutional Authority (word-boundary matching)
# ---------------------------------------------------------------------------

def score_institutional_authority(
    authors: list[dict[str, Any]],
    config: dict[str, Any],
) -> int:
    """Flat points if any author affiliated with a target institution."""
    pattern = config.get("_institution_re")
    if pattern is None:
        return 0
    flat_pts = config.get("flat_institution", _DEFAULT_SCORING["flat_institution"])
    for author in authors:
        affiliations = author.get("affiliations") or []
        if isinstance(affiliations, str):
            affiliations = [affiliations]
        affil_str = " ".join(affiliations)
        if pattern.search(affil_str):
            return flat_pts
    return 0


# ---------------------------------------------------------------------------
# Heuristic 7: Category Intersection (dynamic seed-category overlap)
# ---------------------------------------------------------------------------

def score_category_intersection(
    categories: list[str],
    config: dict[str, Any],
) -> int:
    """
    Award points if the paper's categories overlap with 2+ distinct
    seed-paper category groups (i.e. it bridges multiple research areas
    represented by the seeds).

    Category groups are extracted from seed papers at config time and
    deduplicated, so identical seeds don't inflate the count.
    """
    seed_groups = config.get("_seed_category_groups", [])
    if len(seed_groups) < 2 or not categories:
        return 0
    max_pts = config.get("flat_cross_cat", _DEFAULT_SCORING["flat_cross_cat"])
    cat_set = set(categories)
    groups_hit = sum(1 for group in seed_groups if cat_set & group)
    if groups_hit < 2:
        return 0
    return max_pts


# ---------------------------------------------------------------------------
# Heuristic 8: Benchmark Specificity (dataset names in abstract)
# ---------------------------------------------------------------------------

def score_benchmark_specificity(
    abstract: str,
    config: dict[str, Any],
) -> int:
    """Full points for any benchmark/dataset name in abstract."""
    benchmarks = config.get("benchmarks", _DEFAULT_BENCHMARKS)
    max_pts = config.get("max_benchmark", _DEFAULT_SCORING["max_benchmark"])
    abstract_lower = abstract.lower()
    for b in benchmarks:
        if b.lower() in abstract_lower:
            return max_pts
    return 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def score_paper(paper: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Score a single paper using all 8 heuristics.
    Returns a shallow copy of the paper with 'score_breakdown' injected.
    """
    authors    = paper.get("authors") or []
    abstract   = paper.get("abstract") or ""
    categories = paper.get("categories") or []
    max_total  = config.get("max_total", 100)

    sem   = score_semantic_similarity(paper, config)
    sref  = score_author_in_seed_refs(authors, config)
    cseed = score_cites_seed(paper, config)
    auth  = score_author_authority(authors, config)
    vel   = score_citation_velocity(paper, config)
    inst  = score_institutional_authority(authors, config)
    cat   = score_category_intersection(categories, config)
    bench = score_benchmark_specificity(abstract, config)

    total = min(sem + sref + cseed + auth + vel + inst + cat + bench, max_total)

    summary = (
        f"Total: {total}/{max_total} | "
        f"Sem: {sem}, SRef: {sref}, Cite: {cseed}, Auth: {auth}, "
        f"Vel: {vel}, Inst: {inst}, Cat: {cat}, Bench: {bench}"
    )

    result = copy.copy(paper)
    result["score_breakdown"] = {
        "total":         total,
        "max_total":     max_total,
        "semantic":      sem,
        "seed_ref":      sref,
        "cites_seed":    cseed,
        "authority":     auth,
        "velocity":      vel,
        "institution":   inst,
        "cross_cat":     cat,
        "benchmark":     bench,
        "summary":       summary,
    }
    return result


def rank_papers(
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score all candidates and return sorted by total score descending."""
    scored = [score_paper(p, config) for p in candidates]
    scored.sort(key=lambda p: p["score_breakdown"]["total"], reverse=True)
    return scored
