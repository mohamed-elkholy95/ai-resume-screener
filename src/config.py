"""Global configuration for the AI Resume Screener project.

Central configuration hub for paths, scoring weights, model settings, and
API parameters.  All scoring weights are validated at import time to ensure
they sum to 1.0 (within floating-point tolerance).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
MODEL_DIR: Path = DATA_DIR / "models"
REPORTS_DIR: Path = DATA_DIR / "reports"

# Ensure directories exist at import time
for _dir in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Weight validation helper
# ---------------------------------------------------------------------------

def _validate_weights(weights: dict[str, float], name: str = "weights") -> None:
    """Verify that a weight dictionary sums to 1.0 (±0.001).

    Raises:
        ValueError: If any weight is negative or if the sum deviates from 1.0.
    """
    for key, val in weights.items():
        if val < 0:
            raise ValueError(f"{name}['{key}'] is negative ({val})")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-3):
        raise ValueError(
            f"{name} must sum to 1.0, got {total:.4f}: {weights}"
        )


# ---------------------------------------------------------------------------
# Scoring weights — must sum to 1.0
# ---------------------------------------------------------------------------
MATCH_WEIGHTS: dict[str, float] = {
    "skill_match": 0.40,
    "experience_match": 0.30,
    "education_match": 0.20,
    "semantic_similarity": 0.10,
}

_validate_weights(MATCH_WEIGHTS, "MATCH_WEIGHTS")

# ---------------------------------------------------------------------------
# Pre-built scoring profiles for common hiring scenarios
# ---------------------------------------------------------------------------

SCORING_PROFILES: dict[str, dict[str, float]] = {
    "default": MATCH_WEIGHTS,
    "skills_heavy": {
        "skill_match": 0.55,
        "experience_match": 0.20,
        "education_match": 0.15,
        "semantic_similarity": 0.10,
    },
    "experience_heavy": {
        "skill_match": 0.30,
        "experience_match": 0.45,
        "education_match": 0.15,
        "semantic_similarity": 0.10,
    },
    "balanced": {
        "skill_match": 0.30,
        "experience_match": 0.30,
        "education_match": 0.20,
        "semantic_similarity": 0.20,
    },
}

# Validate every built-in profile at import time
for _profile_name, _profile_weights in SCORING_PROFILES.items():
    _validate_weights(_profile_weights, f"SCORING_PROFILES['{_profile_name}']")

# ---------------------------------------------------------------------------
# Match thresholds (composite score → label)
# ---------------------------------------------------------------------------
MATCH_THRESHOLDS: dict[str, float] = {
    "strong": 0.75,
    "moderate": 0.50,
    "weak": 0.30,
}

# ---------------------------------------------------------------------------
# NER entity types
# ---------------------------------------------------------------------------
NER_ENTITY_TYPES: list[str] = [
    "SKILL",
    "EDUCATION",
    "EXPERIENCE",
    "COMPANY",
    "JOB_TITLE",
    "CERTIFICATION",
    "LANGUAGE",
    "DATE",
    "TECHNOLOGY",
]

# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
CLASSIFICATION_LABELS: list[str] = [
    "Strong Match",
    "Moderate Match",
    "Weak Match",
    "No Match",
]

# Composite score boundaries for each class
CLASSIFICATION_THRESHOLDS: dict[str, float] = {
    "strong": 0.75,
    "moderate": 0.55,
    "weak": 0.35,
    # below 0.35 → "No Match"
}

# Supported model types for ResumeClassifier
SUPPORTED_CLASSIFIER_MODELS: list[str] = [
    "logistic_regression",
    "random_forest",
    "bert",
]

# ---------------------------------------------------------------------------
# NLP / Embedding models
# ---------------------------------------------------------------------------
SPACY_MODEL: str = "en_core_web_sm"
SENTENCE_TRANSFORMER_MODEL: str = "all-mpnet-base-v2"
BERT_MODEL: str = "bert-base-uncased"

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
API_HOST: str = "0.0.0.0"
API_PORT: int = 8001
API_TITLE: str = "AI Resume Screener API"
API_VERSION: str = "v1"
API_PREFIX: str = "/api/v1"

# Maximum resume file size in bytes (10 MB)
MAX_RESUME_SIZE_BYTES: int = 10 * 1024 * 1024

# Supported upload MIME types
SUPPORTED_MIME_TYPES: list[str] = [
    "text/plain",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
]
