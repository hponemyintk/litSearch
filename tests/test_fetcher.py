"""
Unit tests for fetcher.py URL resolution.
These tests cover the parsing/pattern logic only — no live network calls.
Run with: python -m pytest test_fetcher.py -v
"""
import pytest
from fetcher import (
    _try_arxiv,
    _try_doi_in_url,
    _try_publisher_url,
    resolve_url_to_ss_paper_id,
    parse_arxiv_id,
    _is_recent,
)
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# _try_arxiv
# ---------------------------------------------------------------------------

class TestTryArxiv:
    def test_abs_url(self):
        assert _try_arxiv("https://arxiv.org/abs/2505.10960") == "ARXIV:2505.10960"

    def test_pdf_url(self):
        assert _try_arxiv("https://arxiv.org/pdf/2505.10960") == "ARXIV:2505.10960"

    def test_pdf_url_with_version(self):
        # Version suffix should be stripped
        assert _try_arxiv("https://arxiv.org/pdf/2505.10960v2") == "ARXIV:2505.10960"

    def test_arxiv_colon_prefix(self):
        assert _try_arxiv("arxiv:2301.07987") == "ARXIV:2301.07987"

    def test_bare_id_not_matched(self):
        # Bare ID without arxiv.org or arxiv: prefix should NOT match
        # (avoids false positives on date-like strings)
        assert _try_arxiv("2505.10960") is None

    def test_non_arxiv_url_returns_none(self):
        assert _try_arxiv("https://www.nature.com/articles/s41586-024-08328-6") is None

    def test_pdf_direct_link_returns_none(self):
        assert _try_arxiv("https://kumo.ai/research/kumo_relational_foundation_model.pdf") is None

    def test_five_digit_id(self):
        assert _try_arxiv("https://arxiv.org/abs/2301.07987") == "ARXIV:2301.07987"

    def test_returns_arxiv_prefix(self):
        result = _try_arxiv("https://arxiv.org/pdf/2510.06377")
        assert result is not None and result.startswith("ARXIV:")


# ---------------------------------------------------------------------------
# _try_doi_in_url
# ---------------------------------------------------------------------------

class TestTryDoiInUrl:
    def test_doi_org_url(self):
        result = _try_doi_in_url("https://doi.org/10.1038/s41586-024-08328-6")
        assert result == "DOI:10.1038/s41586-024-08328-6"

    def test_doi_in_path(self):
        result = _try_doi_in_url("https://example.com/paper/10.1234/some.paper")
        assert result == "DOI:10.1234/some.paper"

    def test_no_doi_returns_none(self):
        assert _try_doi_in_url("https://arxiv.org/abs/2505.10960") is None

    def test_nature_url_without_doi_prefix(self):
        # nature.com URL does NOT have 10.1038 in it — should return None here
        result = _try_doi_in_url("https://www.nature.com/articles/s41586-024-08328-6")
        assert result is None

    def test_strips_trailing_punctuation(self):
        result = _try_doi_in_url("https://doi.org/10.1234/abc.def/")
        assert result is not None
        assert not result.endswith("/")


# ---------------------------------------------------------------------------
# _try_publisher_url
# ---------------------------------------------------------------------------

class TestTryPublisherUrl:
    def test_nature_articles(self):
        url = "https://www.nature.com/articles/s41586-024-08328-6"
        result = _try_publisher_url(url)
        assert result == "DOI:10.1038/s41586-024-08328-6"

    def test_nature_subdomain(self):
        url = "https://www.nature.com/articles/s41467-023-12345-0"
        result = _try_publisher_url(url)
        assert result == "DOI:10.1038/s41467-023-12345-0"

    def test_science_org(self):
        url = "https://www.science.org/doi/10.1126/science.abc1234"
        result = _try_publisher_url(url)
        assert result == "DOI:10.1126/science.abc1234"

    def test_pnas(self):
        url = "https://www.pnas.org/doi/abs/10.1073/pnas.2024001121"
        result = _try_publisher_url(url)
        assert result == "DOI:10.1073/pnas.2024001121"

    def test_pnas_full(self):
        url = "https://www.pnas.org/doi/full/10.1073/pnas.2024001121"
        result = _try_publisher_url(url)
        assert result == "DOI:10.1073/pnas.2024001121"

    def test_arxiv_not_matched(self):
        assert _try_publisher_url("https://arxiv.org/abs/2505.10960") is None

    def test_kumo_pdf_not_matched(self):
        url = "https://kumo.ai/research/kumo_relational_foundation_model.pdf"
        assert _try_publisher_url(url) is None

    def test_unknown_journal_returns_none(self):
        assert _try_publisher_url("https://journals.example.com/article/12345") is None


# ---------------------------------------------------------------------------
# resolve_url_to_ss_paper_id — pure-parsing cases only
# (network-dependent strategies 4 & 5 are tested separately / with mocks)
# ---------------------------------------------------------------------------

class TestResolveUrlToSsPaperId:
    # --- arXiv (strategy 1) ---
    def test_arxiv_abs(self):
        assert resolve_url_to_ss_paper_id(
            "https://arxiv.org/abs/2505.10960"
        ) == "ARXIV:2505.10960"

    def test_arxiv_pdf(self):
        assert resolve_url_to_ss_paper_id(
            "https://arxiv.org/pdf/2505.10960"
        ) == "ARXIV:2505.10960"

    def test_all_four_arxiv_inputs(self):
        for url in [
            "https://arxiv.org/pdf/2505.10960",
            "https://arxiv.org/pdf/2510.06377",
            "https://arxiv.org/pdf/2511.08667",
            "https://arxiv.org/pdf/2502.17361",
        ]:
            result = resolve_url_to_ss_paper_id(url)
            assert result is not None and result.startswith("ARXIV:"), \
                f"Failed for {url}: got {result}"

    # --- Nature (strategy 3 — publisher pattern) ---
    def test_nature_url(self):
        result = resolve_url_to_ss_paper_id(
            "https://www.nature.com/articles/s41586-024-08328-6"
        )
        assert result == "DOI:10.1038/s41586-024-08328-6"

    # --- Whitespace stripping ---
    def test_strips_leading_trailing_whitespace(self):
        result = resolve_url_to_ss_paper_id(
            "  https://arxiv.org/pdf/2505.10960  "
        )
        assert result == "ARXIV:2505.10960"

    # --- Strategy priority: arXiv wins over DOI if both present ---
    def test_arxiv_takes_priority(self):
        # A hypothetical URL that has both arXiv ID and DOI-like string
        url = "https://arxiv.org/abs/2505.10960?doi=10.1234/foo"
        result = resolve_url_to_ss_paper_id(url)
        assert result == "ARXIV:2505.10960"


# ---------------------------------------------------------------------------
# parse_arxiv_id — backward compatibility
# ---------------------------------------------------------------------------

class TestParseArxivId:
    def test_abs_url(self):
        assert parse_arxiv_id("https://arxiv.org/abs/2301.07987") == "2301.07987"

    def test_pdf_url(self):
        assert parse_arxiv_id("https://arxiv.org/pdf/2505.10960") == "2505.10960"

    def test_strips_version(self):
        assert parse_arxiv_id("https://arxiv.org/abs/2301.07987v3") == "2301.07987"

    def test_non_arxiv_returns_none(self):
        assert parse_arxiv_id("https://www.nature.com/articles/s41586-024-08328-6") is None

    def test_returns_bare_id_without_prefix(self):
        result = parse_arxiv_id("https://arxiv.org/pdf/2510.06377")
        assert result == "2510.06377"  # no "ARXIV:" prefix


# ---------------------------------------------------------------------------
# _is_recent
# ---------------------------------------------------------------------------

class TestIsRecent:
    def _cutoff(self, days_ago: int) -> datetime:
        from datetime import timedelta
        return datetime.now(tz=timezone.utc) - timedelta(days=days_ago)

    def test_today_is_recent(self):
        cutoff = self._cutoff(1)
        today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        assert _is_recent(today, cutoff) is True

    def test_old_date_not_recent(self):
        cutoff = self._cutoff(1)
        assert _is_recent("2020-01-01", cutoff) is False

    def test_empty_string_returns_false(self):
        assert _is_recent("", self._cutoff(1)) is False

    def test_invalid_date_returns_false(self):
        assert _is_recent("not-a-date", self._cutoff(1)) is False

    def test_exactly_at_cutoff_is_recent(self):
        cutoff = datetime(2024, 1, 14, 0, 0, 0, tzinfo=timezone.utc)
        assert _is_recent("2024-01-14", cutoff) is True
