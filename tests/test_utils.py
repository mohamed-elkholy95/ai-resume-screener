"""Tests for src/utils.py — text processing, file helpers, and scoring utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    clamp,
    compute_text_hash,
    extract_sections,
    format_score,
    jaccard_similarity,
    keyword_density,
    parse_year_range,
    slugify,
    text_stats,
    truncate_text,
)


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic_slug(self) -> None:
        assert slugify("Senior Python Developer") == "senior-python-developer"

    def test_special_characters_removed(self) -> None:
        assert slugify("ML/AI Engineer (Remote)") == "mlai-engineer-remote"

    def test_unicode_normalisation(self) -> None:
        assert slugify("Résumé Screening") == "resume-screening"

    def test_multiple_spaces_collapsed(self) -> None:
        assert slugify("data   science   lead") == "data-science-lead"

    def test_empty_string(self) -> None:
        assert slugify("") == ""


# ---------------------------------------------------------------------------
# truncate_text
# ---------------------------------------------------------------------------


class TestTruncateText:
    def test_short_text_unchanged(self) -> None:
        assert truncate_text("hello", max_chars=100) == "hello"

    def test_long_text_truncated(self) -> None:
        result = truncate_text("a" * 200, max_chars=50)
        assert len(result) == 50
        assert result.endswith("…")

    def test_custom_suffix(self) -> None:
        result = truncate_text("abcdefgh", max_chars=5, suffix="...")
        assert result.endswith("...")
        assert len(result) == 5

    def test_exact_boundary(self) -> None:
        text = "x" * 100
        assert truncate_text(text, max_chars=100) == text


# ---------------------------------------------------------------------------
# extract_sections
# ---------------------------------------------------------------------------


class TestExtractSections:
    def test_finds_common_headers(self) -> None:
        text = "Experience\nWorked at Corp.\nEducation\nB.S. CS"
        sections = extract_sections(text)
        assert "experience" in sections
        assert "education" in sections

    def test_no_headers_returns_full_text(self) -> None:
        text = "Just a block of text with no section headers."
        sections = extract_sections(text)
        assert "full_text" in sections

    def test_skills_header(self) -> None:
        text = "Skills\nPython, Java\nExperience\n5 years at Acme"
        sections = extract_sections(text)
        assert "skills" in sections or "skill" in sections


# ---------------------------------------------------------------------------
# compute_text_hash
# ---------------------------------------------------------------------------


class TestComputeTextHash:
    def test_deterministic(self) -> None:
        assert compute_text_hash("hello") == compute_text_hash("hello")

    def test_different_inputs_differ(self) -> None:
        assert compute_text_hash("abc") != compute_text_hash("xyz")

    def test_returns_12_char_hex(self) -> None:
        result = compute_text_hash("test input")
        assert len(result) == 12
        assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# parse_year_range
# ---------------------------------------------------------------------------


class TestParseYearRange:
    def test_standard_range(self) -> None:
        result = parse_year_range("2018 – 2022")
        assert result == (2018, 2022)

    def test_present_range(self) -> None:
        result = parse_year_range("2020 - Present")
        assert result is not None
        assert result[0] == 2020
        # end year should be current year
        assert result[1] >= 2024

    def test_no_match_returns_none(self) -> None:
        assert parse_year_range("no dates here") is None

    def test_pre_1970_returns_none(self) -> None:
        # The parser only considers years >= 1970 as valid
        assert parse_year_range("1960 - 1965") is None


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_within_range(self) -> None:
        assert clamp(0.5) == 0.5

    def test_below_low(self) -> None:
        assert clamp(-0.1) == 0.0

    def test_above_high(self) -> None:
        assert clamp(1.5) == 1.0

    def test_custom_bounds(self) -> None:
        assert clamp(15, low=10, high=20) == 15
        assert clamp(5, low=10, high=20) == 10
        assert clamp(25, low=10, high=20) == 20


# ---------------------------------------------------------------------------
# format_score
# ---------------------------------------------------------------------------


class TestFormatScore:
    def test_as_percent(self) -> None:
        assert format_score(0.856) == "85.6%"

    def test_as_decimal(self) -> None:
        result = format_score(0.856, as_percent=False)
        assert "%" not in result
        assert "0.856" in result

    def test_zero_score(self) -> None:
        assert format_score(0.0) == "0.0%"

    def test_perfect_score(self) -> None:
        assert format_score(1.0) == "100.0%"

    def test_custom_decimals(self) -> None:
        assert format_score(0.8567, decimals=2) == "85.67%"


# ---------------------------------------------------------------------------
# text_stats
# ---------------------------------------------------------------------------


class TestTextStats:
    def test_empty_text(self) -> None:
        stats = text_stats("")
        assert stats["char_count"] == 0
        assert stats["word_count"] == 0
        assert stats["unique_word_ratio"] == 0.0

    def test_basic_text(self) -> None:
        stats = text_stats("hello world hello")
        assert stats["word_count"] == 3
        assert stats["unique_word_ratio"] == pytest.approx(2 / 3, abs=0.01)

    def test_multiline(self) -> None:
        stats = text_stats("line one\nline two\nline three")
        assert stats["line_count"] == 3

    def test_avg_word_length(self) -> None:
        stats = text_stats("abc defgh")
        # "abc" = 3, "defgh" = 5, avg = 4.0
        assert stats["avg_word_length"] == 4.0


# ---------------------------------------------------------------------------
# keyword_density
# ---------------------------------------------------------------------------


class TestKeywordDensity:
    def test_basic_density(self) -> None:
        text = "python java python python java"
        density = keyword_density(text, ["python", "java"])
        # python appears 3/5 = 60%, java 2/5 = 40%
        assert density["python"] == pytest.approx(60.0, abs=0.1)
        assert density["java"] == pytest.approx(40.0, abs=0.1)

    def test_missing_keyword(self) -> None:
        density = keyword_density("hello world", ["rust"])
        assert density["rust"] == 0.0

    def test_empty_text(self) -> None:
        density = keyword_density("", ["python"])
        assert density["python"] == 0.0

    def test_case_insensitive(self) -> None:
        density = keyword_density("Python PYTHON python", ["python"])
        assert density["python"] > 0


# ---------------------------------------------------------------------------
# jaccard_similarity
# ---------------------------------------------------------------------------


class TestJaccardSimilarity:
    def test_identical_sets(self) -> None:
        assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_disjoint_sets(self) -> None:
        assert jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_partial_overlap(self) -> None:
        # {a, b} ∩ {b, c} = {b}, union = {a, b, c} → 1/3
        result = jaccard_similarity({"a", "b"}, {"b", "c"})
        assert result == pytest.approx(1 / 3, abs=0.01)

    def test_both_empty(self) -> None:
        assert jaccard_similarity(set(), set()) == 0.0

    def test_one_empty(self) -> None:
        assert jaccard_similarity({"a"}, set()) == 0.0
