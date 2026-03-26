"""Shared utility helpers for the AI Resume Screener.

Provides text normalisation, file I/O, score formatting, and text-analysis
utilities used across the matching, scoring, and evaluation pipelines.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """Convert a text string to a URL/filename-safe slug.

    Args:
        text: Input string.

    Returns:
        Lowercase hyphen-separated slug.

    Example:
        >>> slugify("Senior Python Developer")
        'senior-python-developer'
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def truncate_text(text: str, max_chars: int = 4096, suffix: str = "…") -> str:
    """Truncate text to a maximum character limit.

    Args:
        text: Input string.
        max_chars: Maximum characters to retain.
        suffix: String appended when truncation occurs.

    Returns:
        Truncated string, or original string if within limit.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def extract_sections(text: str) -> Dict[str, str]:
    """Split resume text into named sections.

    Looks for common resume section headers and partitions the text
    accordingly.

    Args:
        text: Cleaned resume text.

    Returns:
        Dict mapping normalised section name to section content.
    """
    section_headers = [
        r"(?:work\s+)?experience",
        r"education",
        r"skills?",
        r"technical\s+skills?",
        r"certifications?",
        r"projects?",
        r"summary",
        r"objective",
        r"publications?",
        r"awards?",
        r"languages?",
        r"volunteer",
        r"references?",
    ]
    pattern = re.compile(
        r"^(?:" + "|".join(section_headers) + r")\s*[:\-]?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    sections: Dict[str, str] = {}
    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        header = match.group(0).strip().lower().rstrip(": -")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[header] = text[start:end].strip()

    if not sections:
        sections["full_text"] = text

    return sections


def compute_text_hash(text: str) -> str:
    """Compute a short SHA-256 hash of a text string for deduplication.

    Args:
        text: Input text.

    Returns:
        12-character hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Number / year utilities
# ---------------------------------------------------------------------------


def parse_year_range(text: str) -> Optional[tuple[int, int]]:
    """Parse a year range string like ``"2018 – 2022"`` or ``"2020 - Present"``.

    Args:
        text: Date range string.

    Returns:
        ``(start_year, end_year)`` tuple where end_year uses the current year
        for "Present". Returns ``None`` if parsing fails.
    """
    import datetime

    current_year = datetime.datetime.now().year
    pattern = re.compile(
        r"(\d{4})\s*[-\u2013\u2014]\s*(Present|Current|(\d{4}))",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        start = int(match.group(1))
        end_str = match.group(2)
        end = current_year if end_str.lower() in ("present", "current") else int(end_str)
        if 1970 <= start <= current_year and start <= end:
            return start, end
    return None


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a float to [low, high].

    Args:
        value: Input value.
        low: Minimum bound.
        high: Maximum bound.

    Returns:
        Clamped float.
    """
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# File utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> Path:
    """Create a directory (and parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        :class:`pathlib.Path` of the created/existing directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_read_text(file_path: str, encoding: str = "utf-8") -> Optional[str]:
    """Read a text file, returning ``None`` on error instead of raising.

    Args:
        file_path: Path to the file.
        encoding: File encoding.

    Returns:
        File contents as a string, or ``None`` if the file cannot be read.
    """
    try:
        return Path(file_path).read_text(encoding=encoding, errors="replace")
    except OSError as exc:
        logger.error("Cannot read file '%s': %s", file_path, exc)
        return None


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> None:
    """Configure root logger for the project.

    Args:
        level: Logging level string (e.g. ``"DEBUG"``, ``"INFO"``).
        log_file: Optional path to write logs to a file in addition to stdout.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if log_file:
        ensure_dir(str(Path(log_file).parent))
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )
    logger.debug("Logging initialised at level %s", level.upper())


# ---------------------------------------------------------------------------
# Score formatting
# ---------------------------------------------------------------------------


def format_score(score: float, as_percent: bool = True, decimals: int = 1) -> str:
    """Format a score float as a percentage or decimal string.

    Args:
        score: Score value in [0, 1].
        as_percent: If ``True``, multiply by 100 and append ``%``.
        decimals: Number of decimal places.

    Returns:
        Formatted string.
    """
    if as_percent:
        return f"{score * 100:.{decimals}f}%"
    return f"{score:.{decimals + 2}f}"


# ---------------------------------------------------------------------------
# Text statistics
# ---------------------------------------------------------------------------


def text_stats(text: str) -> Dict[str, Any]:
    """Compute basic statistics about a text document.

    Useful for resume quality checks — extremely short or long documents
    often indicate parsing errors.

    Args:
        text: Input text.

    Returns:
        Dict with ``char_count``, ``word_count``, ``line_count``,
        ``avg_word_length``, and ``unique_word_ratio``.
    """
    if not text:
        return {
            "char_count": 0,
            "word_count": 0,
            "line_count": 0,
            "avg_word_length": 0.0,
            "unique_word_ratio": 0.0,
        }
    words = text.split()
    word_count = len(words)
    unique_words = set(w.lower() for w in words)
    avg_len = sum(len(w) for w in words) / word_count if word_count else 0.0

    return {
        "char_count": len(text),
        "word_count": word_count,
        "line_count": text.count("\n") + 1,
        "avg_word_length": round(avg_len, 2),
        "unique_word_ratio": round(len(unique_words) / word_count, 4) if word_count else 0.0,
    }


def keyword_density(text: str, keywords: List[str]) -> Dict[str, float]:
    """Calculate the density of each keyword in the text.

    Density = (occurrences of keyword / total words) × 100.

    Args:
        text: Source text.
        keywords: List of keywords to check (case-insensitive).

    Returns:
        Dict mapping each keyword to its density percentage.
    """
    if not text or not keywords:
        return {kw: 0.0 for kw in keywords}

    lower_text = text.lower()
    total_words = max(len(lower_text.split()), 1)
    densities: Dict[str, float] = {}

    for kw in keywords:
        # Count non-overlapping occurrences (case-insensitive)
        pattern = re.compile(r"\b" + re.escape(kw.lower()) + r"\b")
        count = len(pattern.findall(lower_text))
        densities[kw] = round((count / total_words) * 100, 4)

    return densities


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets.

    Jaccard = |A ∩ B| / |A ∪ B|.  Returns 0.0 when both sets are empty.

    Args:
        set_a: First set.
        set_b: Second set.

    Returns:
        Similarity score in [0, 1].
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return round(intersection / union, 4) if union else 0.0
