"""
Unit tests for scorer.py (9 heuristics + config builder + aggregation).
Run with: python -m pytest test_scorer.py -v
"""
import pytest
from datetime import datetime, timedelta, timezone

from scorer import (
    # config
    build_scoring_config,
    _build_recency_bins,
    _extract_keyword_weights,
    _extract_target_authors,
    _extract_reference_topics,
    _extract_area_signatures,
    # heuristics
    score_weighted_keywords,
    score_author_authority,
    score_reference_overlap,
    score_open_source,
    score_institutional_authority,
    score_category_intersection,
    score_benchmark_specificity,
    score_multi_area_intersection,
    score_recency_momentum,
    # aggregation
    score_paper,
    rank_papers,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Three seed papers covering distinct sub-areas: graph ML, tabular ML, relational
SEED_PAPERS = [
    {
        "title": "Relational Deep Learning on Heterogeneous Graphs",
        "abstract": (
            "We propose a relational deep learning framework for heterogeneous "
            "graph neural networks using message passing aggregation."
        ),
        "authors": [{"name": "Jure Leskovec", "hIndex": 120}],
        "references": [
            {"title": "GraphSAGE: Inductive Representation Learning on Large Graphs",
             "authors": [{"name": "Jure Leskovec"}]},
        ],
    },
    {
        "title": "Tabular Foundation Model with In-Context Learning",
        "abstract": (
            "We introduce a tabular foundation model that uses in-context learning "
            "for cross-table prediction on structured data."
        ),
        "authors": [{"name": "Frank Hutter", "hIndex": 65}],
        "references": [
            {"title": "TabPFN: Prior-Fitted Networks for Tabular Data",
             "authors": [{"name": "Frank Hutter"}]},
        ],
    },
    {
        "title": "Graph Transformer for Knowledge Graph Completion",
        "abstract": (
            "We present a graph transformer architecture with heterogeneous "
            "graph attention for knowledge graph completion tasks."
        ),
        "authors": [{"name": "Matthias Fey", "hIndex": 55}],
        "references": [],
    },
]

# A fixed 'now' for all recency tests: 2024-01-15 06:00 UTC
# Papers published on 2024-01-15 → pub_dt = midnight UTC → age = 6h
# Papers published on 2024-01-14 → age = 30h
TEST_NOW = datetime(2024, 1, 15, 6, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def seed_papers():
    return SEED_PAPERS


@pytest.fixture(scope="module")
def config(seed_papers):
    return build_scoring_config(seed_papers, window_hours=24)


@pytest.fixture
def minimal_config():
    """Deterministic config for tests that don't need real TF-IDF extraction."""
    return {
        "keyword_weights": {"graph transformer": 10, "message passing": 5},
        "target_authors": ["Jure Leskovec"],
        "target_topics": ["graphsage"],
        "area_signatures": [
            {"graph transformer", "message passing"},
            {"tabular", "foundation model"},
        ],
        "recency_bins": _build_recency_bins(24),
        "window_hours": 24,
    }


# ---------------------------------------------------------------------------
# build_scoring_config / helpers
# ---------------------------------------------------------------------------

class TestBuildScoringConfig:
    def test_returns_all_keys(self, config):
        assert "keyword_weights"  in config
        assert "target_authors"   in config
        assert "target_topics"    in config
        assert "area_signatures"  in config
        assert "recency_bins"     in config
        assert "window_hours"     in config

    def test_keyword_weights_non_empty(self, config):
        assert len(config["keyword_weights"]) > 0

    def test_keyword_weights_values_in_range(self, config):
        for w in config["keyword_weights"].values():
            assert 3 <= w <= 15

    def test_target_authors_only_high_hindex(self, config):
        # All three seed authors have h > 50 so all three should appear
        assert "Jure Leskovec" in config["target_authors"]
        assert "Frank Hutter"  in config["target_authors"]
        assert "Matthias Fey"  in config["target_authors"]

    def test_low_hindex_author_excluded(self):
        papers = [{"title": "T", "abstract": "A",
                   "authors": [{"name": "Nobody", "hIndex": 10}],
                   "references": []}]
        cfg = build_scoring_config(papers)
        assert "Nobody" not in cfg["target_authors"]

    def test_area_signatures_one_per_seed(self, config, seed_papers):
        assert len(config["area_signatures"]) == len(seed_papers)

    def test_area_signatures_are_sets(self, config):
        for sig in config["area_signatures"]:
            assert isinstance(sig, set)

    def test_recency_bins_count(self, config):
        assert len(config["recency_bins"]) == 4

    def test_window_hours_stored(self):
        cfg = build_scoring_config(SEED_PAPERS, window_hours=48)
        assert cfg["window_hours"] == 48

    def test_empty_seed_papers(self):
        cfg = build_scoring_config([])
        assert cfg["keyword_weights"] == {}
        assert cfg["target_authors"]  == []
        assert cfg["target_topics"]   == []
        assert cfg["area_signatures"] == []


class TestBuildRecencyBins:
    def test_24h_window(self):
        bins = _build_recency_bins(24)
        assert bins == [(6.0, 8), (12.0, 6), (18.0, 4), (24.0, 2)]

    def test_48h_window(self):
        bins = _build_recency_bins(48)
        assert bins == [(12.0, 8), (24.0, 6), (36.0, 4), (48.0, 2)]

    def test_72h_window(self):
        bins = _build_recency_bins(72)
        assert bins == [(18.0, 8), (36.0, 6), (54.0, 4), (72.0, 2)]

    def test_6h_window(self):
        bins = _build_recency_bins(6)
        assert bins == [(1.5, 8), (3.0, 6), (4.5, 4), (6.0, 2)]

    def test_bins_sorted_ascending_age(self):
        bins = _build_recency_bins(24)
        ages = [age for age, _ in bins]
        assert ages == sorted(ages)

    def test_points_decrease_with_age(self):
        bins = _build_recency_bins(24)
        pts = [p for _, p in bins]
        assert pts == sorted(pts, reverse=True)


class TestExtractTargetAuthors:
    def test_filters_above_50(self):
        papers = [{"authors": [
            {"name": "High H", "hIndex": 80},
            {"name": "Low H",  "hIndex": 30},
            {"name": "Border", "hIndex": 50},   # exactly 50 — excluded
        ], "references": []}]
        result = _extract_target_authors(papers)
        assert "High H"  in result
        assert "Low H"   not in result
        assert "Border"  not in result

    def test_deduplicates_across_papers(self):
        papers = [
            {"authors": [{"name": "Alice", "hIndex": 60}], "references": []},
            {"authors": [{"name": "Alice", "hIndex": 60}], "references": []},
        ]
        result = _extract_target_authors(papers)
        assert result.count("Alice") == 1

    def test_empty_papers(self):
        assert _extract_target_authors([]) == []

    def test_missing_hindex_treated_as_zero(self):
        papers = [{"authors": [{"name": "Ghost"}], "references": []}]
        assert _extract_target_authors(papers) == []


class TestExtractReferenceTopics:
    def test_returns_list_of_strings(self):
        topics = _extract_reference_topics(SEED_PAPERS)
        assert isinstance(topics, list)
        assert all(isinstance(t, str) for t in topics)

    def test_empty_references(self):
        papers = [{"authors": [], "references": [], "title": "T", "abstract": "A"}]
        assert _extract_reference_topics(papers) == []

    def test_frequent_term_appears(self):
        # "learning" appears in both ref titles → should show up
        papers = [{"authors": [], "references": [
            {"title": "Deep Learning for Graphs"},
            {"title": "Representation Learning"},
        ], "title": "", "abstract": ""}]
        topics = _extract_reference_topics(papers, top_n=5)
        combined = " ".join(topics)
        assert "learning" in combined


# ---------------------------------------------------------------------------
# 1. Dynamic Weighted Keywords
# ---------------------------------------------------------------------------

class TestWeightedKeywords:
    def test_empty_inputs_give_zero(self, minimal_config):
        assert score_weighted_keywords("", "", minimal_config) == 0

    def test_empty_config_gives_zero(self):
        cfg = {"keyword_weights": {}}
        assert score_weighted_keywords("graph transformer", "abstract", cfg) == 0

    def test_title_match_double_weight(self, minimal_config):
        # "graph transformer" in title only → 10 * 2 = 20
        assert score_weighted_keywords("Graph Transformer Networks", "", minimal_config) == 20

    def test_abstract_match_base_weight(self, minimal_config):
        # "message passing" in abstract only → 5 * 1 = 5
        assert score_weighted_keywords("", "We use message passing.", minimal_config) == 5

    def test_title_and_abstract_both_count(self, minimal_config):
        # "graph transformer" title (20) + abstract (10) = 30
        score = score_weighted_keywords(
            "Graph Transformer Networks",
            "We propose a graph transformer.",
            minimal_config,
        )
        assert score == 30

    def test_cap_at_40(self, minimal_config):
        # Overload all keywords to force cap
        cfg = {"keyword_weights": {f"kw{i}": 15 for i in range(10)}}
        title = " ".join(f"kw{i}" for i in range(10))
        assert score_weighted_keywords(title, title, cfg) == 40

    def test_case_insensitive(self, minimal_config):
        assert score_weighted_keywords("GRAPH TRANSFORMER", "", minimal_config) == 20

    def test_partial_phrase_no_match(self, minimal_config):
        # "graph" alone should not match "graph transformer"
        assert score_weighted_keywords("Graph Networks", "graph only", minimal_config) == 0

    def test_each_keyword_counted_once_per_field(self, minimal_config):
        # Repeated occurrence of a keyword in the abstract should only count once
        score = score_weighted_keywords(
            "message passing",
            "message passing message passing",
            minimal_config,
        )
        # title: 5*2=10, abstract: 5*1=5 (counted once per field) = 15
        assert score == 15


# ---------------------------------------------------------------------------
# 2. Author Authority
# ---------------------------------------------------------------------------

class TestAuthorAuthority:
    def test_empty_authors(self):
        assert score_author_authority([]) == 0

    def test_zero_hindex(self):
        assert score_author_authority([{"hIndex": 0}]) == 0

    def test_single_author_h25(self):
        assert score_author_authority([{"hIndex": 25}]) == 28

    def test_takes_maximum(self):
        authors = [{"hIndex": 10}, {"hIndex": 35}, {"hIndex": 5}]
        assert score_author_authority(authors) == 31  # h=35 → 31

    def test_cap_at_40(self):
        assert score_author_authority([{"hIndex": 100}]) == 40
        assert score_author_authority([{"hIndex": 80}, {"hIndex": 100}]) == 40

    def test_h40_gives_32(self):
        assert score_author_authority([{"hIndex": 40}]) == 32

    def test_h100_gives_exactly_40(self):
        assert score_author_authority([{"hIndex": 100}]) == 40

    def test_missing_hindex_key(self):
        authors = [{"name": "Alice"}, {"hIndex": 20}]
        assert score_author_authority(authors) == 26  # h=20 → 26

    def test_none_hindex_treated_as_zero(self):
        authors = [{"hIndex": None}, {"hIndex": 15}]
        assert score_author_authority(authors) == 24  # h=15 → 24

    def test_monotone_increasing(self):
        scores = [score_author_authority([{"hIndex": h}]) for h in [1, 10, 25, 50, 75, 100]]
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# 3. Dynamic Reference Overlap
# ---------------------------------------------------------------------------

class TestReferenceOverlap:
    def test_no_references(self, minimal_config):
        assert score_reference_overlap([], minimal_config) == 0

    def test_matching_target_author(self, minimal_config):
        refs = [{"title": "Some Paper", "authors": [{"name": "Jure Leskovec"}]}]
        assert score_reference_overlap(refs, minimal_config) == 10

    def test_matching_target_topic(self, minimal_config):
        refs = [{"title": "GraphSAGE: Inductive Representation", "authors": []}]
        assert score_reference_overlap(refs, minimal_config) == 10

    def test_two_distinct_matches(self, minimal_config):
        refs = [
            {"title": "GraphSAGE paper", "authors": []},
            {"title": "Another paper", "authors": [{"name": "Jure Leskovec"}]},
        ]
        assert score_reference_overlap(refs, minimal_config) == 20

    def test_cap_at_20(self, minimal_config):
        cfg = dict(minimal_config)
        cfg["target_authors"] = ["A", "B", "C"]
        cfg["target_topics"]  = ["x", "y"]
        refs = [
            {"title": "x topic", "authors": [{"name": "A"}]},
            {"title": "y topic", "authors": [{"name": "B"}]},
            {"title": "z topic", "authors": [{"name": "C"}]},
        ]
        assert score_reference_overlap(refs, cfg) == 20

    def test_same_target_matched_multiple_refs_counted_once(self, minimal_config):
        # "graphsage" appears in two refs — should only count once
        refs = [
            {"title": "GraphSAGE intro", "authors": []},
            {"title": "GraphSAGE follow-up", "authors": []},
        ]
        assert score_reference_overlap(refs, minimal_config) == 10

    def test_partial_author_name_no_match(self, minimal_config):
        refs = [{"title": "Paper", "authors": [{"name": "Jure"}]}]
        assert score_reference_overlap(refs, minimal_config) == 0

    def test_empty_config_targets(self):
        cfg = {"target_authors": [], "target_topics": []}
        refs = [{"title": "GraphSAGE", "authors": [{"name": "Jure Leskovec"}]}]
        assert score_reference_overlap(refs, cfg) == 0


# ---------------------------------------------------------------------------
# 4. Open Source Bonus
# ---------------------------------------------------------------------------

class TestOpenSource:
    def test_empty_abstract(self):
        assert score_open_source("") == 0

    def test_github_link(self):
        assert score_open_source("Code at https://github.com/user/repo.") == 15

    def test_huggingface_link(self):
        assert score_open_source("Model at huggingface.co/models/foo.") == 15

    def test_code_is_available(self):
        assert score_open_source("The code is available upon request.") == 15

    def test_open_source_phrase(self):
        assert score_open_source("This is an open-source implementation.") == 15

    def test_we_release(self):
        assert score_open_source("We release our weights and code publicly.") == 15

    def test_multiple_matches_still_15(self):
        assert score_open_source("github.com/x and we release open-source code.") == 15

    def test_no_match(self):
        assert score_open_source("We present a novel method without releasing code.") == 0

    def test_case_insensitive(self):
        assert score_open_source("Code Is Available on our website.") == 15


# ---------------------------------------------------------------------------
# 5. Institutional Authority
# ---------------------------------------------------------------------------

class TestInstitutionalAuthority:
    def test_empty_authors(self):
        assert score_institutional_authority([]) == 0

    def test_stanford(self):
        assert score_institutional_authority([{"affiliations": ["Stanford University"]}]) == 15

    def test_deepmind(self):
        assert score_institutional_authority([{"affiliations": ["DeepMind, London"]}]) == 15

    def test_fair(self):
        assert score_institutional_authority([{"affiliations": ["Meta FAIR"]}]) == 15

    def test_max_planck(self):
        assert score_institutional_authority(
            [{"affiliations": ["Max Planck Institute for Intelligent Systems"]}]
        ) == 15

    def test_tubingen(self):
        assert score_institutional_authority(
            [{"affiliations": ["University of Tübingen"]}]
        ) == 15

    def test_multiple_matches_still_15(self):
        authors = [
            {"affiliations": ["Stanford University"]},
            {"affiliations": ["MIT CSAIL"]},
        ]
        assert score_institutional_authority(authors) == 15

    def test_no_match(self):
        assert score_institutional_authority([{"affiliations": ["University of Nowhere"]}]) == 0

    def test_missing_affiliations_key(self):
        authors = [{"name": "Alice"}, {"affiliations": ["Berkeley"]}]
        assert score_institutional_authority(authors) == 15

    def test_affiliations_as_plain_string(self):
        assert score_institutional_authority([{"affiliations": "MIT, Cambridge"}]) == 15


# ---------------------------------------------------------------------------
# 6. Category Intersection
# ---------------------------------------------------------------------------

class TestCategoryIntersection:
    def test_empty(self):
        assert score_category_intersection([]) == 0

    def test_cs_db_and_cs_lg(self):
        assert score_category_intersection(["cs.DB", "cs.LG"]) == 10

    def test_cs_db_and_stat_ml(self):
        assert score_category_intersection(["cs.DB", "stat.ML"]) == 10

    def test_cs_db_both(self):
        assert score_category_intersection(["cs.DB", "cs.LG", "stat.ML"]) == 10

    def test_cs_db_only(self):
        assert score_category_intersection(["cs.DB"]) == 0

    def test_cs_lg_only(self):
        assert score_category_intersection(["cs.LG", "stat.ML"]) == 0

    def test_unrelated(self):
        assert score_category_intersection(["cs.CV", "cs.NLP"]) == 0

    def test_case_sensitive(self):
        assert score_category_intersection(["CS.DB", "cs.LG"]) == 0


# ---------------------------------------------------------------------------
# 7. Benchmark Specificity
# ---------------------------------------------------------------------------

class TestBenchmarkSpecificity:
    def test_empty(self):
        assert score_benchmark_specificity("") == 0

    def test_single(self):
        assert score_benchmark_specificity("We compare against XBoost.") == 5

    def test_two(self):
        assert score_benchmark_specificity("We use CatBoost and LightGBM.") == 10

    def test_relbench(self):
        assert score_benchmark_specificity("Evaluated on RelBench.") == 5

    def test_autogluon(self):
        assert score_benchmark_specificity("AutoGluon is a strong baseline.") == 5

    def test_cap_at_10(self):
        abstract = "RelBench XBoost CatBoost LightGBM AutoGluon"
        assert score_benchmark_specificity(abstract) == 10

    def test_no_match(self):
        assert score_benchmark_specificity("We compare against several baselines.") == 0


# ---------------------------------------------------------------------------
# 8. Dynamic Multi-area Intersection
# ---------------------------------------------------------------------------

class TestMultiAreaIntersection:
    def test_empty_signatures(self):
        cfg = {"area_signatures": []}
        assert score_multi_area_intersection("anything", "anything", cfg) == 0

    def test_no_match(self, minimal_config):
        assert score_multi_area_intersection(
            "Image Segmentation", "convolutional features pixel-wise", minimal_config
        ) == 0

    def test_one_of_two_areas_hit(self, minimal_config):
        # minimal_config has 2 areas; hitting 1 → round(1/2 * 10) = 5
        score = score_multi_area_intersection(
            "Graph Transformer Networks",
            "We use graph transformer and message passing.",
            minimal_config,
        )
        assert score == 5

    def test_both_areas_hit(self, minimal_config):
        # hitting both areas → round(2/2 * 10) = 10
        score = score_multi_area_intersection(
            "Graph Transformer for Tabular Foundation Model",
            "message passing with foundation model",
            minimal_config,
        )
        assert score == 10

    def test_cap_at_10(self):
        # 5 areas, all hit → 10
        cfg = {"area_signatures": [{"kw" + str(i)} for i in range(5)]}
        text = "kw0 kw1 kw2 kw3 kw4"
        assert score_multi_area_intersection(text, text, cfg) == 10

    def test_empty_signature_sets_not_counted(self):
        # An empty signature set should not count as a 'hit'
        cfg = {"area_signatures": [set(), {"graph"}, set()]}
        assert score_multi_area_intersection("graph paper", "graph", cfg) == 3  # 1/3*10

    def test_proportional_scoring(self, config, seed_papers):
        # A paper touching all three seed areas should score higher
        # than one touching only one
        broad_paper = (
            "Relational Deep Learning Tabular Foundation Model Graph Transformer",
            "heterogeneous graph message passing in-context learning tabular structured"
        )
        narrow_paper = (
            "Graph Transformer Networks",
            "message passing aggregation graph neural network"
        )
        broad_score  = score_multi_area_intersection(*broad_paper, config)
        narrow_score = score_multi_area_intersection(*narrow_paper, config)
        assert broad_score >= narrow_score


# ---------------------------------------------------------------------------
# 9. Recency + Citation Momentum
# ---------------------------------------------------------------------------

class TestRecencyMomentum:
    # TEST_NOW = 2024-01-15 06:00 UTC
    # Paper published 2024-01-15 → pub_dt = midnight → age = 6h exactly
    # → within first quartile of 24h window (≤6h) → 8 pts

    def _paper(self, pub_date: str, citations: int = 0, influential: int = 0) -> dict:
        return {
            "published": pub_date,
            "citationCount": citations,
            "influentialCitationCount": influential,
        }

    def test_freshest_quartile_24h_window(self, minimal_config):
        p = self._paper("2024-01-15")  # age = 6h at TEST_NOW → Q1 → 8pts
        assert score_recency_momentum(p, minimal_config, _now=TEST_NOW) == 8

    def test_second_quartile_24h_window(self):
        # Need age in (6h, 12h] → paper published 2024-01-15, _now at +9h
        now = datetime(2024, 1, 15, 9, 0, 0, tzinfo=timezone.utc)
        cfg = {"recency_bins": _build_recency_bins(24)}
        p = self._paper("2024-01-15")  # age = 9h → Q2 → 6pts
        assert score_recency_momentum(p, cfg, _now=now) == 6

    def test_third_quartile_24h_window(self):
        now = datetime(2024, 1, 15, 15, 0, 0, tzinfo=timezone.utc)
        cfg = {"recency_bins": _build_recency_bins(24)}
        p = self._paper("2024-01-15")  # age = 15h → Q3 → 4pts
        assert score_recency_momentum(p, cfg, _now=now) == 4

    def test_oldest_quartile_24h_window(self):
        now = datetime(2024, 1, 15, 21, 0, 0, tzinfo=timezone.utc)
        cfg = {"recency_bins": _build_recency_bins(24)}
        p = self._paper("2024-01-15")  # age = 21h → Q4 → 2pts
        assert score_recency_momentum(p, cfg, _now=now) == 2

    def test_outside_window_zero_recency(self, minimal_config):
        # Paper published 2 days ago — outside 24h window
        p = self._paper("2024-01-13")  # age = 54h → 0 recency pts
        score = score_recency_momentum(p, minimal_config, _now=TEST_NOW)
        assert score == 0  # no recency + no citations = 0

    def test_bins_scale_with_window_48h(self):
        cfg = {"recency_bins": _build_recency_bins(48)}
        # age = 30h → 48h bins: [12,24,36,48] → 30h in (24,36] → Q3 → 4pts
        now = datetime(2024, 1, 16, 6, 0, 0, tzinfo=timezone.utc)
        p = self._paper("2024-01-15")  # age=30h at this now
        assert score_recency_momentum(p, cfg, _now=now) == 4

    def test_influential_citation_3_gives_7(self, minimal_config):
        p = self._paper("2000-01-01", influential=3)
        # no recency, momentum = 7
        assert score_recency_momentum(p, minimal_config, _now=TEST_NOW) == 7

    def test_influential_citation_1_gives_5(self, minimal_config):
        p = self._paper("2000-01-01", influential=1)
        assert score_recency_momentum(p, minimal_config, _now=TEST_NOW) == 5

    def test_citation_10_gives_3(self, minimal_config):
        p = self._paper("2000-01-01", citations=10)
        assert score_recency_momentum(p, minimal_config, _now=TEST_NOW) == 3

    def test_citation_3_gives_1(self, minimal_config):
        p = self._paper("2000-01-01", citations=3)
        assert score_recency_momentum(p, minimal_config, _now=TEST_NOW) == 1

    def test_combined_recency_and_momentum_capped_at_15(self, minimal_config):
        p = self._paper("2024-01-15", influential=3)  # 8 + 7 = 15
        assert score_recency_momentum(p, minimal_config, _now=TEST_NOW) == 15

    def test_missing_published_date(self, minimal_config):
        p = {"published": "", "citationCount": 0, "influentialCitationCount": 0}
        assert score_recency_momentum(p, minimal_config, _now=TEST_NOW) == 0

    def test_invalid_published_date(self, minimal_config):
        p = self._paper("not-a-date")
        assert score_recency_momentum(p, minimal_config, _now=TEST_NOW) == 0

    def test_no_citations_no_recency_gives_zero(self, minimal_config):
        p = self._paper("2000-01-01")
        assert score_recency_momentum(p, minimal_config, _now=TEST_NOW) == 0


# ---------------------------------------------------------------------------
# score_paper — integration
# ---------------------------------------------------------------------------

class TestScorePaper:
    def _blank(self, **kw) -> dict:
        base = {
            "title": "", "abstract": "", "authors": [],
            "references": [], "categories": [],
            "published": "", "citationCount": 0, "influentialCitationCount": 0,
        }
        base.update(kw)
        return base

    def test_zero_score_blank_paper(self, minimal_config):
        assert score_paper(self._blank(), minimal_config)["score_breakdown"]["total"] == 0

    def test_all_breakdown_keys_present(self, minimal_config):
        bd = score_paper(self._blank(), minimal_config)["score_breakdown"]
        for key in ("total", "keywords", "authority", "references", "code",
                    "institution", "cross_cat", "benchmark", "multi_area",
                    "recency", "summary"):
            assert key in bd

    def test_summary_format(self, minimal_config):
        bd = score_paper(self._blank(), minimal_config)["score_breakdown"]
        assert "Total:" in bd["summary"]
        assert "/175" in bd["summary"]

    def test_total_equals_component_sum(self, minimal_config):
        paper = self._blank(
            title="Graph Transformer",
            abstract="github.com/foo. We use message passing. CatBoost LightGBM.",
            authors=[{"hIndex": 30, "affiliations": ["Stanford University"]}],
            categories=["cs.DB", "cs.LG"],
            references=[{"title": "graphsage network", "authors": []}],
            published="2024-01-15",
            citationCount=0,
            influentialCitationCount=0,
        )
        result = score_paper(paper, minimal_config)
        bd = result["score_breakdown"]
        component_sum = (
            bd["keywords"] + bd["authority"] + bd["references"] + bd["code"]
            + bd["institution"] + bd["cross_cat"] + bd["benchmark"]
            + bd["multi_area"] + bd["recency"]
        )
        assert bd["total"] == min(component_sum, 175)

    def test_total_capped_at_175(self, config):
        paper = self._blank(
            title="Relational Deep Learning Tabular Foundation Model Graph Transformer",
            abstract=(
                "in-context learning heterogeneous graph message passing "
                "github.com/x CatBoost LightGBM RelBench AutoGluon XBoost"
            ),
            authors=[{"hIndex": 100, "affiliations": ["Stanford University"]}],
            categories=["cs.DB", "cs.LG", "stat.ML"],
            references=[
                {"title": "graphsage paper", "authors": [{"name": "Jure Leskovec"}]},
                {"title": "tabpfn paper",    "authors": [{"name": "Frank Hutter"}]},
            ],
            published="2024-01-15",
            citationCount=0,
            influentialCitationCount=3,
        )
        result = score_paper(paper, config)
        assert result["score_breakdown"]["total"] <= 175

    def test_preserves_original_keys(self, minimal_config):
        paper = self._blank(title="Test", abstract="nothing")
        result = score_paper(paper, minimal_config)
        assert result["title"] == "Test"
        assert "score_breakdown" in result

    def test_does_not_mutate_original(self, minimal_config):
        paper = self._blank(title="Original")
        score_paper(paper, minimal_config)
        assert "score_breakdown" not in paper


# ---------------------------------------------------------------------------
# rank_papers — ordering
# ---------------------------------------------------------------------------

class TestRankPapers:
    def _paper(self, title: str, h: int = 0) -> dict:
        return {
            "title": title, "abstract": "", "authors": [{"hIndex": h}],
            "references": [], "categories": [],
            "published": "", "citationCount": 0, "influentialCitationCount": 0,
        }

    def test_empty_list(self, minimal_config):
        assert rank_papers([], minimal_config) == []

    def test_single_paper_returned(self, minimal_config):
        assert len(rank_papers([self._paper("A")], minimal_config)) == 1

    def test_sorted_descending_by_hindex(self, minimal_config):
        papers = [self._paper("Low", 5), self._paper("High", 80), self._paper("Mid", 30)]
        result = rank_papers(papers, minimal_config)
        titles = [p["title"] for p in result]
        assert titles == ["High", "Mid", "Low"]

    def test_score_breakdown_injected(self, minimal_config):
        result = rank_papers([self._paper("X", h=20)], minimal_config)
        assert "score_breakdown" in result[0]

    def test_all_zero_scores_all_returned(self, minimal_config):
        papers = [self._paper(t) for t in ["A", "B", "C"]]
        assert len(rank_papers(papers, minimal_config)) == 3

    def test_higher_score_ranks_first(self, minimal_config):
        low  = {"title": "L", "abstract": "",
                "authors": [{"hIndex": 5}], "references": [],
                "categories": [], "published": "", "citationCount": 0,
                "influentialCitationCount": 0}
        high = {"title": "H",
                "abstract": "github.com/foo CatBoost LightGBM",
                "authors": [{"hIndex": 80, "affiliations": ["Stanford"]}],
                "references": [], "categories": ["cs.DB", "cs.LG"],
                "published": "", "citationCount": 0, "influentialCitationCount": 0}
        result = rank_papers([low, high], minimal_config)
        assert result[0]["title"] == "H"
