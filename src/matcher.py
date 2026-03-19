"""
matcher.py — Resume-to-Job-Description matching and scoring.
Computes multi-dimensional similarity scores between resumes and job descriptions.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    import numpy as np  # type: ignore[import]
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers not installed — using TF-IDF fallback")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not installed — semantic similarity limited")

from data_collection import JobDescription, Resume


# ---------------------------------------------------------------------------
# Education level ordering
# ---------------------------------------------------------------------------

_EDU_ORDER: dict[str, int] = {
    "High School": 0,
    "Associate's": 1,
    "Bachelor's": 2,
    "Master's": 3,
    "PhD": 4,
}


class ResumeMatcher:
    """Compute match scores between Resume and JobDescription objects."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the matcher with an optional sentence embedding model.

        Args:
            embedding_model: Name of the sentence-transformers model to use.
                             Falls back to TF-IDF cosine similarity if unavailable.
        """
        self._model: Any = None
        self._tfidf: Any = None

        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self._model = SentenceTransformer(embedding_model)
                logger.info("Loaded sentence transformer: %s", embedding_model)
            except Exception as exc:
                logger.warning("Could not load sentence transformer: %s", exc)

        if self._model is None and HAS_SKLEARN:
            self._tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            logger.info("Using TF-IDF cosine similarity fallback")

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def compute_match_score(self, resume: Resume, jd: JobDescription) -> float:
        """Compute a composite match score between a resume and job description.

        Args:
            resume: Parsed Resume object.
            jd: Parsed JobDescription object.

        Returns:
            Float score in [0, 1].
        """
        breakdown = self._compute_breakdown(resume, jd)
        weights = {
            "skill_match": 0.40,
            "experience": 0.30,
            "education": 0.20,
            "semantic": 0.10,
        }
        score = (
            breakdown["skill_score"] * weights["skill_match"]
            + breakdown["experience_score"] * weights["experience"]
            + breakdown["education_score"] * weights["education"]
            + breakdown["semantic_score"] * weights["semantic"]
        )
        return round(min(max(score, 0.0), 1.0), 4)

    def compute_skill_match(
        self,
        resume_skills: list[str],
        required: list[str],
        preferred: list[str],
    ) -> dict[str, Any]:
        """Compare resume skills against required and preferred skills.

        Args:
            resume_skills: Skills extracted from resume.
            required: Required skills from the job description.
            preferred: Preferred skills from the job description.

        Returns:
            Dict with keys: matched_required, missing_required,
            matched_preferred, missing_preferred, additional, skill_score.
        """
        resume_set = {s.lower() for s in resume_skills}
        required_set = {s.lower() for s in required}
        preferred_set = {s.lower() for s in preferred}

        matched_required = [s for s in required if s.lower() in resume_set]
        missing_required = [s for s in required if s.lower() not in resume_set]
        matched_preferred = [s for s in preferred if s.lower() in resume_set]
        missing_preferred = [s for s in preferred if s.lower() not in resume_set]
        additional = [s for s in resume_skills if s.lower() not in required_set | preferred_set]

        # Skill score: required skills count 70%, preferred 30%
        req_score = (len(matched_required) / len(required_set)) if required_set else 1.0
        pref_score = (len(matched_preferred) / len(preferred_set)) if preferred_set else 1.0
        skill_score = req_score * 0.7 + pref_score * 0.3

        return {
            "matched_required": matched_required,
            "missing_required": missing_required,
            "matched_preferred": matched_preferred,
            "missing_preferred": missing_preferred,
            "additional": additional,
            "skill_score": round(skill_score, 4),
        }

    def compute_experience_match(
        self, resume_years: float, required_years: float
    ) -> float:
        """Score experience match as a float in [0, 1].

        Args:
            resume_years: Years of experience from resume.
            required_years: Minimum years required by the job.

        Returns:
            1.0 if resume meets or exceeds requirement,
            proportional score otherwise.
        """
        if required_years <= 0:
            return 1.0
        if resume_years >= required_years:
            return 1.0
        # Partial credit: graceful degradation
        ratio = resume_years / required_years
        # Use a curve that rewards partial experience
        return round(min(ratio, 1.0), 4)

    def compute_education_match(
        self, resume_edu: list[str], required_level: str
    ) -> float:
        """Score education match as a float in [0, 1].

        Args:
            resume_edu: List of education mentions from resume.
            required_level: Required education level string.

        Returns:
            1.0 if resume education meets or exceeds requirement.
        """
        if not required_level or required_level not in _EDU_ORDER:
            return 1.0

        required_rank = _EDU_ORDER[required_level]

        # Find highest education level in resume
        resume_rank = -1
        edu_lower = " ".join(resume_edu).lower()

        for level, rank in sorted(_EDU_ORDER.items(), key=lambda x: x[1], reverse=True):
            if level.lower() in edu_lower or any(
                kw in edu_lower for kw in _get_edu_keywords(level)
            ):
                resume_rank = rank
                break

        if resume_rank < 0:
            # No recognized level found
            return 0.5

        if resume_rank >= required_rank:
            return 1.0

        # Partial credit for being one level below
        gap = required_rank - resume_rank
        return round(max(0.0, 1.0 - gap * 0.25), 4)

    def compute_semantic_similarity(
        self, resume_text: str, jd_text: str
    ) -> float:
        """Compute semantic similarity between resume and JD texts.

        Uses sentence-transformers if available, otherwise TF-IDF cosine similarity.

        Args:
            resume_text: Clean resume text.
            jd_text: Job description text.

        Returns:
            Cosine similarity score in [0, 1].
        """
        if not resume_text or not jd_text:
            return 0.0

        if self._model is not None:
            import numpy as np  # noqa: F811

            embeddings = self._model.encode(
                [resume_text[:512], jd_text[:512]], show_progress_bar=False
            )
            cos_sim = float(
                np.dot(embeddings[0], embeddings[1])
                / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-9)
            )
            return round(max(0.0, cos_sim), 4)

        if HAS_SKLEARN:
            return self._tfidf_similarity(resume_text, jd_text)

        # Jaccard fallback
        return self._jaccard_similarity(resume_text, jd_text)

    def batch_score_resumes(
        self,
        resumes: list[Resume],
        jd: JobDescription,
        top_k: int = 10,
    ) -> list[tuple[Resume, float, dict[str, Any]]]:
        """Score and rank a list of resumes against a single job description.

        Args:
            resumes: List of Resume objects.
            jd: Target JobDescription.
            top_k: Return only the top-k results.

        Returns:
            List of (Resume, score, breakdown) tuples sorted by descending score.
        """
        results: list[tuple[Resume, float, dict[str, Any]]] = []

        for resume in resumes:
            breakdown = self._compute_breakdown(resume, jd)
            score = self.compute_match_score(resume, jd)
            results.append((resume, score, breakdown))

        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(
            "Batch scored %d resumes; top score=%.3f",
            len(resumes),
            results[0][1] if results else 0.0,
        )
        return results[:top_k]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_breakdown(
        self, resume: Resume, jd: JobDescription
    ) -> dict[str, Any]:
        """Compute individual component scores and skill breakdown."""
        skill_result = self.compute_skill_match(
            resume.skills, jd.required_skills, jd.preferred_skills
        )
        exp_score = self.compute_experience_match(
            resume.experience_years, jd.min_experience
        )
        edu_score = self.compute_education_match(resume.education, jd.education_level)
        sem_score = self.compute_semantic_similarity(resume.clean_text, jd.raw_text)

        return {
            "skill_score": skill_result["skill_score"],
            "skill_detail": skill_result,
            "experience_score": exp_score,
            "education_score": edu_score,
            "semantic_score": sem_score,
        }

    def _tfidf_similarity(self, text1: str, text2: str) -> float:
        """TF-IDF cosine similarity between two texts."""
        try:
            matrix = self._tfidf.fit_transform([text1[:10000], text2[:10000]])
            sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
            return round(float(sim), 4)
        except Exception as exc:
            logger.warning("TF-IDF similarity failed: %s", exc)
            return self._jaccard_similarity(text1, text2)

    @staticmethod
    def _jaccard_similarity(text1: str, text2: str) -> float:
        """Jaccard similarity as a last-resort fallback."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return round(intersection / union, 4) if union else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_edu_keywords(level: str) -> list[str]:
    """Return keyword synonyms for a given education level."""
    mapping = {
        "PhD": ["phd", "ph.d", "doctorate", "doctoral"],
        "Master's": ["master", "m.s.", "msc", "m.eng"],
        "Bachelor's": ["bachelor", "b.s.", "bsc", "b.a.", "b.eng"],
        "Associate's": ["associate"],
        "High School": ["high school", "ged"],
    }
    return mapping.get(level, [])
