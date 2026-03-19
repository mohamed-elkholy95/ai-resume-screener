"""
ner_extractor.py — Named Entity Recognition for resume text.
Uses spaCy for base NER and custom pattern matching for domain entities.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from skill_taxonomy import get_all_skills, match_skill

logger = logging.getLogger(__name__)

try:
    import spacy
    from spacy.tokens import DocBin
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logger.warning("spaCy not installed — using pattern-based NER only")


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_EXPERIENCE_PATTERNS = [
    re.compile(r"(\d+)\+?\s*(?:to\s*\d+\s*)?years?\s+(?:of\s+)?(?:relevant\s+)?experience", re.IGNORECASE),
    re.compile(r"experience\s*(?:of\s+)?(\d+)\+?\s*years?", re.IGNORECASE),
    re.compile(r"(\d+)\s*\+\s*years?\s+(?:in|of|with)", re.IGNORECASE),
]

_EDUCATION_PATTERNS = [
    re.compile(r"\b(ph\.?d\.?|doctor(?:ate)?)\b", re.IGNORECASE),
    re.compile(r"\b(m\.?s\.?c?\.?|master(?:\'?s)?|m\.?eng\.?)\b", re.IGNORECASE),
    re.compile(r"\b(b\.?s\.?c?\.?|b\.?a\.?|bachelor(?:\'?s)?|b\.?eng\.?)\b", re.IGNORECASE),
    re.compile(r"\b(associate(?:\'?s)?|a\.?s\.?)\b", re.IGNORECASE),
    re.compile(
        r"\b(computer science|data science|software engineering|electrical engineering|information technology)\b",
        re.IGNORECASE,
    ),
]

_CERTIFICATION_PATTERNS = [
    re.compile(
        r"\b(AWS Certified [A-Za-z ]+|Google Cloud Professional [A-Za-z ]+|"
        r"Microsoft Certified[: ][A-Za-z ]+|CFA|PMP|CISSP|CPA|CISA|CISM|"
        r"Certified [A-Za-z ]+Professional|[A-Z]{2,6}-[A-Z]{1,4})\b",
        re.IGNORECASE,
    )
]

_JOB_TITLE_PATTERNS = [
    re.compile(
        r"\b(senior|junior|lead|staff|principal|head of|director of|vp of|chief)?\s*"
        r"(software engineer|data scientist|ml engineer|machine learning engineer|"
        r"data engineer|backend engineer|frontend engineer|full.stack engineer|"
        r"devops engineer|sre|site reliability engineer|product manager|"
        r"data analyst|research scientist|ai engineer|platform engineer)\b",
        re.IGNORECASE,
    )
]


# ---------------------------------------------------------------------------
# ResumeNER
# ---------------------------------------------------------------------------


class ResumeNER:
    """Extract named entities from resume text using spaCy + pattern matching."""

    def __init__(self, model_path: Optional[str] = None) -> None:
        """Initialize the NER extractor.

        Args:
            model_path: Path to a custom spaCy model directory. If None,
                        loads en_core_web_sm (or falls back to pattern-only mode).
        """
        self._nlp: Any = None
        self._all_skills = get_all_skills()

        if HAS_SPACY:
            self._load_spacy_model(model_path)
        else:
            logger.info("Operating in pattern-only NER mode")

    def _load_spacy_model(self, model_path: Optional[str]) -> None:
        """Load spaCy model, falling back to en_core_web_sm."""
        try:
            if model_path and Path(model_path).exists():
                self._nlp = spacy.load(model_path)
                logger.info("Loaded custom spaCy model from %s", model_path)
            else:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded en_core_web_sm spaCy model")
        except OSError:
            logger.warning(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
            self._nlp = None

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract all entity types from resume text.

        Args:
            text: Cleaned resume text.

        Returns:
            Dict mapping entity type labels to lists of extracted strings.
            Entity types: SKILL, EDUCATION, EXPERIENCE, COMPANY, JOB_TITLE,
                          CERTIFICATION, LANGUAGE, DATE, TECHNOLOGY.
        """
        entities: dict[str, list[str]] = {
            "SKILL": [],
            "EDUCATION": [],
            "EXPERIENCE": [],
            "COMPANY": [],
            "JOB_TITLE": [],
            "CERTIFICATION": [],
            "LANGUAGE": [],
            "DATE": [],
            "TECHNOLOGY": [],
        }

        # Pattern-based extraction
        entities["SKILL"] = self.extract_skills(text)
        entities["EDUCATION"] = self._extract_education(text)
        entities["EXPERIENCE"] = self._extract_experience_strings(text)
        entities["CERTIFICATION"] = self._extract_certifications(text)
        entities["JOB_TITLE"] = self._extract_job_titles(text)

        # spaCy-based extraction
        if self._nlp is not None:
            doc = self._nlp(text[:100_000])  # spaCy has token limits
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    entities["COMPANY"].append(ent.text.strip())
                elif ent.label_ == "DATE":
                    entities["DATE"].append(ent.text.strip())
                elif ent.label_ == "LANGUAGE":
                    entities["LANGUAGE"].append(ent.text.strip())

        # Deduplicate all lists
        for key in entities:
            seen: set[str] = set()
            unique: list[str] = []
            for item in entities[key]:
                low = item.lower()
                if low not in seen:
                    seen.add(low)
                    unique.append(item)
            entities[key] = unique

        return entities

    def extract_skills(self, text: str) -> list[str]:
        """Extract skills using taxonomy matching + pattern matching.

        Args:
            text: Resume text.

        Returns:
            List of canonical skill names found in the text.
        """
        found: list[str] = []
        lower_text = text.lower()

        for skill in self._all_skills:
            # Build a word-boundary pattern for multi-word skills too
            pattern = re.compile(
                r"(?<![a-z])" + re.escape(skill) + r"(?![a-z])",
                re.IGNORECASE,
            )
            if pattern.search(lower_text):
                found.append(skill)

        # Also try to match any capitalized tokens via fuzzy matching
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#.\-/]{1,30}", text)
        for token in tokens:
            canonical = match_skill(token, threshold=0.85)
            if canonical and canonical not in found:
                found.append(canonical)

        return found

    def extract_experience_years(self, text: str) -> float:
        """Extract the maximum years of experience from text.

        Args:
            text: Resume or job description text.

        Returns:
            Float years (maximum found), or 0.0 if none detected.
        """
        years_found: list[float] = []
        for pattern in _EXPERIENCE_PATTERNS:
            for match in pattern.finditer(text):
                try:
                    years_found.append(float(match.group(1)))
                except (IndexError, ValueError):
                    pass
        return max(years_found, default=0.0)

    def train_custom_ner(
        self,
        training_data: list[dict[str, Any]],
        output_dir: str,
    ) -> None:
        """Fine-tune a custom spaCy NER model on annotated training data.

        Args:
            training_data: List of dicts with keys "text" and "entities"
                           where entities is a list of (start, end, label) tuples.
            output_dir: Directory to save the trained model.

        Raises:
            ImportError: If spaCy is not installed.
            RuntimeError: If base model could not be loaded.
        """
        if not HAS_SPACY:
            raise ImportError("spaCy is required for training: pip install spacy")

        if self._nlp is None:
            raise RuntimeError("No base spaCy model loaded for training")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build DocBin from training data
        db = DocBin()
        for example in training_data:
            doc = self._nlp.make_doc(example["text"])
            ents = []
            for start, end, label in example.get("entities", []):
                span = doc.char_span(start, end, label=label)
                if span is not None:
                    ents.append(span)
            doc.ents = ents  # type: ignore[assignment]
            db.add(doc)

        train_file = output_path / "train.spacy"
        db.to_disk(str(train_file))
        logger.info(
            "Training data saved to %s (%d examples). Run: python -m spacy train",
            train_file,
            len(training_data),
        )

    def evaluate(self, test_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Evaluate NER extraction quality against ground-truth annotations.

        Args:
            test_data: List of dicts with "text" and "entities" keys
                       (same format as training_data).

        Returns:
            Dict with per-entity-type precision, recall, and F1 scores.
        """
        from evaluation import compute_ner_metrics  # local import to avoid circular

        predicted_all: list[dict[str, list[str]]] = []
        ground_truth_all: list[dict[str, list[str]]] = []

        for example in test_data:
            predicted = self.extract_entities(example["text"])
            gt_entities: dict[str, list[str]] = {}
            for start, end, label in example.get("entities", []):
                gt_entities.setdefault(label, []).append(
                    example["text"][start:end]
                )
            predicted_all.append(predicted)
            ground_truth_all.append(gt_entities)

        return compute_ner_metrics(predicted_all, ground_truth_all)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_education(self, text: str) -> list[str]:
        found: list[str] = []
        for pattern in _EDUCATION_PATTERNS:
            for m in pattern.finditer(text):
                found.append(m.group(0).strip())
        return found

    def _extract_experience_strings(self, text: str) -> list[str]:
        found: list[str] = []
        for pattern in _EXPERIENCE_PATTERNS:
            for m in pattern.finditer(text):
                found.append(m.group(0).strip())
        return found

    def _extract_certifications(self, text: str) -> list[str]:
        found: list[str] = []
        for pattern in _CERTIFICATION_PATTERNS:
            for m in pattern.finditer(text):
                found.append(m.group(0).strip())
        return found

    def _extract_job_titles(self, text: str) -> list[str]:
        found: list[str] = []
        for pattern in _JOB_TITLE_PATTERNS:
            for m in pattern.finditer(text):
                title = m.group(0).strip()
                if title:
                    found.append(title)
        return found
