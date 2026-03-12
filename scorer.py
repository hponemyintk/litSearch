"""
scorer.py — metadata & keyword-driven paper scoring engine.

Scores a paper out of 100 points using 9 heuristics:

  Relevance (65 pts):
    1. Keyword Match              (max 30)  — config.yaml weighted keywords
    2. Reference Overlap          (max 15)  — seed authors (h>50) + ref topics
    3. Keyword Coverage           (max 10)  — fraction of config keywords hit
    4. Author in Seed Refs        (max 10)  — candidate author cited by seeds

  Quality (20 pts):
    5. Author Authority           (max 10)  — logarithmic h-index
    6. Citation Velocity          (max 10)  — effective citations / day

  Bonuses (15 pts):
    7. Institutional Authority    (flat  5) — affiliation check
    8. Category Intersection      (flat  5) — arXiv cs.DB ∩ cs.LG/stat.ML
    9. Benchmark Specificity      (max   5) — dataset/tool names in abstract

Call build_scoring_config(seed_papers, window_hours) once per session,
then pass the returned config to score_paper() / rank_papers().
"""

import math
import copy
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timezone

import yaml
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------------
# Constants (static heuristics 4–7)
# ---------------------------------------------------------------------------

MAX_KEYWORDS         = 30
MAX_REFERENCES       = 15
MAX_KW_COVERAGE      = 10
MAX_SEED_REF_AUTHOR  = 10
MAX_AUTHORITY        = 10
MAX_CITATION_VEL     = 10
FLAT_INSTITUTION     = 5
FLAT_CROSS_CAT       = 5
MAX_BENCHMARK        = 5
MAX_TOTAL            = 100

TARGET_INSTITUTIONS: list[str] = [
    "Stanford", "Max Planck", "DeepMind", "Meta", "FAIR",
    "MIT", "Berkeley", "Tübingen",
]

TARGET_BENCHMARKS: list[str] = [
    "RelBench", "XGBoost", "CatBoost", "LightGBM", "AutoGluon",
]


# ---------------------------------------------------------------------------
# Config builder — call once per session
# ---------------------------------------------------------------------------

def build_scoring_config(
    seed_papers: list[dict[str, Any]],
    window_hours: int = 24,
    config_path: str | None = None,
) -> dict[str, Any]:
    """
    Build dynamic scoring config from seed papers.

    Returns a config dict with keys:
      keyword_weights   dict[str, int]       from config.yaml              (H1)
      target_authors    list[str]            authors with h-index > 50    (H2)
      target_topics     list[str]            frequent ref-title phrases    (H2)
      seed_ref_authors  set[str]             author names from seed refs   (H4)
      window_hours      int
    """
    return {
        "keyword_weights":  _load_keyword_weights(config_path),
        "target_authors":   _extract_target_authors(seed_papers),
        "target_topics":    _extract_reference_topics(seed_papers),
        "seed_ref_authors": _extract_seed_ref_authors(seed_papers),
        "window_hours":     window_hours,
    }


# -- private helpers ---------------------------------------------------------

_DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _load_keyword_weights(config_path: Path | str | None = None) -> dict[str, int]:
    """Load keyword_weights from config.yaml."""
    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("keyword_weights") or {}


def _extract_target_authors(seed_papers: list[dict]) -> list[str]:
    """Unique authors with h-index > 50 across all seed papers."""
    seen: set[str] = set()
    result: list[str] = []
    for paper in seed_papers:
        for author in (paper.get("authors") or []):
            name = (author.get("name") or "").strip()
            h = author.get("hIndex") or 0
            if h > 50 and name and name not in seen:
                result.append(name)
                seen.add(name)
    return result


def _extract_reference_topics(
    seed_papers: list[dict],
    top_n: int = 10,
) -> list[str]:
    """
    Most frequent n-gram phrases from reference titles of all seed papers.
    These represent the canonical works in the research area.
    """
    ref_titles = [
        ref.get("title") or ""
        for paper in seed_papers
        for ref in (paper.get("references") or [])
        if ref.get("title")
    ]
    if not ref_titles:
        return []

    vec = CountVectorizer(
        ngram_range=(1, 3),
        stop_words="english",
        max_features=200,
        min_df=1,
    )
    try:
        X = vec.fit_transform(ref_titles)
    except ValueError:
        return []

    counts = np.asarray(X.sum(axis=0)).flatten()
    names = vec.get_feature_names_out()
    top_idx = counts.argsort()[-top_n:][::-1]
    return [names[i] for i in top_idx if counts[i] > 0]


def _extract_seed_ref_authors(seed_papers: list[dict]) -> set[str]:
    """Collect unique lowercase author names from all seed papers' reference lists."""
    names: set[str] = set()
    for paper in seed_papers:
        for ref in (paper.get("references") or []):
            for author in (ref.get("authors") or []):
                name = (author.get("name") or "").strip().lower()
                if name:
                    names.add(name)
    return names



# ---------------------------------------------------------------------------
# Heuristic 1: Dynamic Weighted Keywords
# ---------------------------------------------------------------------------

def score_weighted_keywords(
    title: str,
    abstract: str,
    config: dict[str, Any],
) -> int:
    """
    Scan title (2× weight) and abstract (1× weight) for config keywords.
    Each keyword counted at most once per field.
    Capped at MAX_KEYWORDS (30).
    """
    keyword_weights = config.get("keyword_weights") or {}
    if not keyword_weights:
        return 0
    title_lower    = title.lower()
    abstract_lower = abstract.lower()
    total = 0
    for keyword, weight in keyword_weights.items():
        kw = keyword.lower()
        if kw in title_lower:
            total += weight * 2
        if kw in abstract_lower:
            total += weight * 1
    return min(total, MAX_KEYWORDS)


# ---------------------------------------------------------------------------
# Heuristic 2: Author Authority (logarithmic h-index)
# ---------------------------------------------------------------------------

def score_author_authority(authors: list[dict[str, Any]]) -> int:
    """
    score = round(10 × log(max_h + 1) / log(101)), capped at 10.
    Calibration: h=10→5, h=25→7, h=50→9, h=100→10.
    """
    if not authors:
        return 0
    max_h = max((a.get("hIndex") or 0) for a in authors)
    if max_h <= 0:
        return 0
    return min(round(10 * math.log(max_h + 1) / math.log(101)), MAX_AUTHORITY)


# ---------------------------------------------------------------------------
# Heuristic 3: Dynamic Reference Overlap
# ---------------------------------------------------------------------------

def score_reference_overlap(
    references: list[dict[str, Any]],
    config: dict[str, Any],
) -> int:
    """
    5 pts per unique match of:
      - target_authors (seed authors with h-index > 50, exact full-name match)
      - target_topics  (frequent phrases from seed reference titles, substring)
    Capped at MAX_REFERENCES (15).
    """
    target_authors = config.get("target_authors") or []
    target_topics  = config.get("target_topics") or []
    matched: set[str] = set()

    for ref in references:
        ref_title   = (ref.get("title") or "").lower()
        ref_authors = ref.get("authors") or []

        for topic in target_topics:
            if topic not in matched and topic.lower() in ref_title:
                matched.add(topic)

        for target in target_authors:
            if target not in matched:
                for author in ref_authors:
                    if target.lower() == (author.get("name") or "").lower():
                        matched.add(target)
                        break

    return min(len(matched) * 5, MAX_REFERENCES)



# ---------------------------------------------------------------------------
# Heuristic 5: Institutional Authority
# ---------------------------------------------------------------------------

def score_institutional_authority(authors: list[dict[str, Any]]) -> int:
    """Flat 5 if any author is affiliated with a target institution."""
    for author in authors:
        affiliations = author.get("affiliations") or []
        if isinstance(affiliations, str):
            affiliations = [affiliations]
        affil_str = " ".join(affiliations).lower()
        for inst in TARGET_INSTITUTIONS:
            if inst.lower() in affil_str:
                return FLAT_INSTITUTION
    return 0


# ---------------------------------------------------------------------------
# Heuristic 6: Category Intersection
# ---------------------------------------------------------------------------

def score_category_intersection(categories: list[str]) -> int:
    """Flat 5 if paper has cs.DB AND (cs.LG OR stat.ML). Case-sensitive."""
    if "cs.DB" in categories and (
        "cs.LG" in categories or "stat.ML" in categories
    ):
        return FLAT_CROSS_CAT
    return 0


# ---------------------------------------------------------------------------
# Heuristic 7: Benchmark / Baseline Specificity
# ---------------------------------------------------------------------------

def score_benchmark_specificity(abstract: str) -> int:
    """5 pts for any benchmark match in abstract, capped at 5."""
    abstract_lower = abstract.lower()
    matched = sum(1 for b in TARGET_BENCHMARKS if b.lower() in abstract_lower)
    return min(matched * 5, MAX_BENCHMARK)


# ---------------------------------------------------------------------------
# Heuristic 3b: Keyword Coverage (fraction of config keywords present)
# ---------------------------------------------------------------------------

def score_keyword_coverage(
    title: str,
    abstract: str,
    config: dict[str, Any],
) -> int:
    """
    Fraction of distinct config keywords that appear in title or abstract.
    Score = round(keywords_hit / total_keywords × MAX_KW_COVERAGE), capped at 10.
    """
    keyword_weights = config.get("keyword_weights") or {}
    if not keyword_weights:
        return 0
    candidate = (title + " " + abstract).lower()
    hits = sum(1 for kw in keyword_weights if kw.lower() in candidate)
    return min(round(hits / len(keyword_weights) * MAX_KW_COVERAGE), MAX_KW_COVERAGE)


# ---------------------------------------------------------------------------
# Heuristic 9: Citation Velocity (citations / day)
# ---------------------------------------------------------------------------

def score_recency_momentum(
    paper: dict[str, Any],
    config: dict[str, Any],
    _now: Optional[datetime] = None,
) -> int:
    """
    Citation velocity = effective_citations / days_since_publication.
    Influential citations count 3× (they are already included once in
    citationCount, so we add 2× influentialCitationCount on top).
    Capped at MAX_CITATION_VEL (10).

    Bins (effective citations/day):
      ≥ 5.0 → 10
      ≥ 2.0 → 8
      ≥ 1.0 → 6
      ≥ 0.5 → 4
      ≥ 0.2 → 3
      ≥ 0.1 → 2
      < 0.1 → 0

    Papers with no publication date or published today get 0.
    _now: injectable datetime for testing (defaults to datetime.now(utc)).
    """
    published    = paper.get("published") or ""
    citations    = paper.get("citationCount") or 0
    influential  = paper.get("influentialCitationCount") or 0
    effective    = citations + influential * 2

    if not published or effective <= 0:
        return 0

    try:
        pub_dt = datetime.strptime(published[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        now = _now or datetime.now(tz=timezone.utc)
        age_days = (now - pub_dt).total_seconds() / 86400
    except ValueError:
        return 0

    if age_days < 1:
        return 0

    velocity = effective / age_days

    if velocity >= 5.0:
        return 10
    elif velocity >= 2.0:
        return 8
    elif velocity >= 1.0:
        return 6
    elif velocity >= 0.5:
        return 4
    elif velocity >= 0.2:
        return 3
    elif velocity >= 0.1:
        return 2
    else:
        return 0


# ---------------------------------------------------------------------------
# Heuristic 4: Author in Seed References
# ---------------------------------------------------------------------------

def score_author_in_seed_refs(
    authors: list[dict[str, Any]],
    config: dict[str, Any],
) -> int:
    """
    Award points if any of the candidate's authors appear as authors
    in the seed papers' reference lists (i.e., this person's work is
    cited by the seeds — they are part of the research community).
    Flat MAX_SEED_REF_AUTHOR (10) for any match.
    """
    seed_ref_authors = config.get("seed_ref_authors") or set()
    if not seed_ref_authors:
        return 0
    for author in authors:
        name = (author.get("name") or "").strip().lower()
        if name and name in seed_ref_authors:
            return MAX_SEED_REF_AUTHOR
    return 0


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def score_paper(paper: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Score a single paper using all 9 heuristics.
    Returns a shallow copy of the paper with 'score_breakdown' injected.
    """
    title      = paper.get("title") or ""
    abstract   = paper.get("abstract") or ""
    authors    = paper.get("authors") or []
    references = paper.get("references") or []
    categories = paper.get("categories") or []

    kw    = score_weighted_keywords(title, abstract, config)
    refs  = score_reference_overlap(references, config)
    cov   = score_keyword_coverage(title, abstract, config)
    sref  = score_author_in_seed_refs(authors, config)
    auth  = score_author_authority(authors)
    vel   = score_recency_momentum(paper, config)
    inst  = score_institutional_authority(authors)
    cat   = score_category_intersection(categories)
    bench = score_benchmark_specificity(abstract)

    total = min(kw + refs + cov + sref + auth + vel + inst + cat + bench, MAX_TOTAL)

    summary = (
        f"Total: {total}/{MAX_TOTAL} | "
        f"KW: {kw}, Ref: {refs}, Cov: {cov}, SRef: {sref}, "
        f"Auth: {auth}, Vel: {vel}, "
        f"Inst: {inst}, Cat: {cat}, Bench: {bench}"
    )

    result = copy.copy(paper)
    result["score_breakdown"] = {
        "total":       total,
        "keywords":    kw,
        "references":  refs,
        "kw_coverage": cov,
        "seed_ref":    sref,
        "authority":   auth,
        "velocity":    vel,
        "institution": inst,
        "cross_cat":   cat,
        "benchmark":   bench,
        "summary":     summary,
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
