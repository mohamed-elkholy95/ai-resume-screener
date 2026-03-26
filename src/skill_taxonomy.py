"""
skill_taxonomy.py — Canonical skill database and matching utilities.
Provides structured skill categories, aliases, and fuzzy matching.
"""

from __future__ import annotations

import difflib
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Skill database
# ---------------------------------------------------------------------------

SKILL_DATABASE: dict[str, list[str]] = {
    "programming_languages": [
        "python",
        "javascript",
        "typescript",
        "java",
        "c",
        "c++",
        "c#",
        "go",
        "rust",
        "scala",
        "kotlin",
        "swift",
        "ruby",
        "php",
        "r",
        "matlab",
        "perl",
        "haskell",
        "lua",
        "julia",
        "dart",
        "elixir",
        "clojure",
        "groovy",
        "bash",
    ],
    "ml_ai": [
        "machine learning",
        "deep learning",
        "natural language processing",
        "nlp",
        "computer vision",
        "reinforcement learning",
        "neural networks",
        "convolutional neural networks",
        "recurrent neural networks",
        "transformers",
        "bert",
        "gpt",
        "llm",
        "pytorch",
        "tensorflow",
        "keras",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "catboost",
        "hugging face",
        "langchain",
        "openai",
        "stable diffusion",
        "generative ai",
        "rag",
        "vector databases",
        "embeddings",
        "feature engineering",
        "model deployment",
    ],
    "data_engineering": [
        "sql",
        "postgresql",
        "mysql",
        "mongodb",
        "redis",
        "elasticsearch",
        "apache spark",
        "apache kafka",
        "apache airflow",
        "dbt",
        "snowflake",
        "databricks",
        "hadoop",
        "hive",
        "presto",
        "pandas",
        "numpy",
        "etl",
        "data pipelines",
        "data warehousing",
    ],
    "cloud_devops": [
        "aws",
        "azure",
        "google cloud",
        "gcp",
        "docker",
        "kubernetes",
        "terraform",
        "ansible",
        "jenkins",
        "github actions",
        "gitlab ci",
        "ci/cd",
        "linux",
        "nginx",
        "helm",
        "prometheus",
        "grafana",
        "datadog",
        "cloudformation",
    ],
    "web_development": [
        "react",
        "vue",
        "angular",
        "next.js",
        "node.js",
        "express",
        "fastapi",
        "django",
        "flask",
        "graphql",
        "rest api",
        "html",
        "css",
        "tailwind",
        "webpack",
        "vite",
        "redux",
        "svelte",
        "nuxt",
    ],
    "soft_skills": [
        "communication",
        "teamwork",
        "leadership",
        "problem solving",
        "critical thinking",
        "project management",
        "agile",
        "scrum",
        "time management",
        "collaboration",
        "mentoring",
    ],
}

# Canonical aliases: alternate → canonical
SKILL_ALIASES: dict[str, str] = {
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "k8s": "kubernetes",
    "tf": "tensorflow",
    "sk": "scikit-learn",
    "sklearn": "scikit-learn",
    "postgres": "postgresql",
    "mongo": "mongodb",
    "es": "elasticsearch",
    "hf": "hugging face",
    "gai": "generative ai",
    "gen ai": "generative ai",
    "vue.js": "vue",
    "reactjs": "react",
    "nodejs": "node.js",
    "nextjs": "next.js",
    "cpp": "c++",
    "csharp": "c#",
    "golang": "go",
    "keras": "keras",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "gcp": "google cloud",
    "aws lambda": "aws",
    "amazon web services": "aws",
    "microsoft azure": "azure",
    "google cloud platform": "google cloud",
}

# Pre-compute flat list for matching
_ALL_SKILLS: list[str] = [
    skill for skills in SKILL_DATABASE.values() for skill in skills
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_all_skills() -> list[str]:
    """Return a flat list of all canonical skills."""
    return list(_ALL_SKILLS)


def match_skill(input_skill: str, threshold: float = 0.8) -> Optional[str]:
    """Fuzzy-match an input string against the canonical skill list.

    Args:
        input_skill: Raw skill string to match.
        threshold: Minimum similarity ratio (0–1) to accept a match.

    Returns:
        Canonical skill name if a match is found, else None.
    """
    normalized = input_skill.strip().lower()

    # Direct hit
    if normalized in _ALL_SKILLS:
        return normalized

    # Alias lookup
    if normalized in SKILL_ALIASES:
        return SKILL_ALIASES[normalized]

    # Fuzzy match
    matches = difflib.get_close_matches(
        normalized, _ALL_SKILLS, n=1, cutoff=threshold
    )
    if matches:
        logger.debug("Fuzzy matched '%s' -> '%s'", input_skill, matches[0])
        return matches[0]

    # Also fuzzy-match against alias keys
    alias_matches = difflib.get_close_matches(
        normalized, list(SKILL_ALIASES.keys()), n=1, cutoff=threshold
    )
    if alias_matches:
        canonical = SKILL_ALIASES[alias_matches[0]]
        logger.debug(
            "Alias fuzzy matched '%s' -> '%s' -> '%s'",
            input_skill,
            alias_matches[0],
            canonical,
        )
        return canonical

    return None


def get_skills_by_category() -> dict[str, list[str]]:
    """Return a copy of the full skill database grouped by category."""
    return {category: list(skills) for category, skills in SKILL_DATABASE.items()}


def categorize_skill(skill: str) -> str:
    """Return the category name for a given skill.

    Args:
        skill: Canonical skill name.

    Returns:
        Category name, or "unknown" if the skill is not found.
    """
    normalized = skill.strip().lower()
    for category, skills in SKILL_DATABASE.items():
        if normalized in skills:
            return category
    # Try alias resolution first
    canonical = match_skill(normalized)
    if canonical:
        for category, skills in SKILL_DATABASE.items():
            if canonical in skills:
                return category
    return "unknown"


# ---------------------------------------------------------------------------
# Proficiency levels
# ---------------------------------------------------------------------------

PROFICIENCY_LEVELS: dict[str, int] = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}

# Keyword patterns that hint at proficiency in resume text
PROFICIENCY_INDICATORS: dict[str, list[str]] = {
    "expert": [
        "expert in", "mastery of", "deep expertise", "led development of",
        "architected", "extensive experience with", "8+ years",
    ],
    "advanced": [
        "advanced knowledge", "proficient in", "strong experience",
        "5+ years", "designed and implemented", "production experience",
    ],
    "intermediate": [
        "familiar with", "working knowledge", "2+ years",
        "contributed to", "used in projects",
    ],
    "beginner": [
        "basic knowledge", "coursework in", "learning",
        "introductory", "exposure to",
    ],
}


def infer_proficiency(skill: str, context: str) -> str:
    """Infer the proficiency level for a skill based on surrounding context.

    Scans the text around a skill mention for proficiency indicator phrases.
    Returns the highest matching level, defaulting to ``"intermediate"`` when
    no indicators are found (a conservative middle-ground assumption).

    Args:
        skill: Canonical skill name.
        context: Resume text (or a relevant excerpt).

    Returns:
        One of ``"beginner"``, ``"intermediate"``, ``"advanced"``, ``"expert"``.
    """
    lower_context = context.lower()

    # Check from highest to lowest proficiency
    for level in ("expert", "advanced", "intermediate", "beginner"):
        for indicator in PROFICIENCY_INDICATORS[level]:
            # Look for the indicator near the skill mention
            if indicator in lower_context and skill.lower() in lower_context:
                return level

    return "intermediate"


def compute_skill_coverage(
    resume_skills: list[str],
    jd_required: list[str],
    jd_preferred: list[str],
) -> dict[str, float]:
    """Compute category-level skill coverage between resume and JD.

    Groups both the candidate's skills and the JD's requirements by
    taxonomy category, then computes the coverage ratio per category.

    Args:
        resume_skills: Skills extracted from resume.
        jd_required: Required skills from the job description.
        jd_preferred: Preferred skills from the job description.

    Returns:
        Dict mapping category names to coverage ratios (0–1).
    """
    # Gather all JD skills by category
    jd_all = [s.lower() for s in jd_required + jd_preferred]
    resume_set = {s.lower() for s in resume_skills}

    category_needs: dict[str, list[str]] = {}
    for skill in jd_all:
        cat = categorize_skill(skill)
        category_needs.setdefault(cat, []).append(skill)

    coverage: dict[str, float] = {}
    for cat, needed in category_needs.items():
        matched = sum(1 for s in needed if s in resume_set)
        coverage[cat] = round(matched / len(needed), 4) if needed else 1.0

    return coverage
