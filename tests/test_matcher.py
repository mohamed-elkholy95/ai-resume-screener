"""Tests for the matcher module."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from src.matcher import ResumeMatcher
from src.data_collection import Resume, JobDescription


@pytest.fixture
def matcher():
    """Create a ResumeMatcher with TF-IDF fallback (no sentence-transformers)."""
    with patch("src.matcher.HAS_SENTENCE_TRANSFORMERS", False):
        m = ResumeMatcher.__new__(ResumeMatcher)
        m._model = None
        m._tfidf = MagicMock()
        # Make _tfidf_similarity return a fixed value
        m._tfidf_similarity = MagicMock(return_value=0.72)
        return m


@pytest.fixture
def sample_resume():
    return Resume(
        raw_text="Experienced ML engineer with Python and TensorFlow skills",
        filename="test.pdf",
        skills=["python", "tensorflow", "docker", "sql", "git"],
        experience_years=5.0,
        education=["BS Computer Science"],
    )


@pytest.fixture
def sample_jd():
    return JobDescription(
        raw_text="Senior ML Engineer with Python, TensorFlow, and Docker",
        title="Senior ML Engineer",
        required_skills=["python", "tensorflow", "docker"],
        preferred_skills=["aws", "kubernetes"],
        min_experience=3.0,
        education_level="bachelor",
    )


class TestSkillMatch:
    """Test skill matching."""

    def test_skill_match_exact(self, matcher):
        result = matcher.compute_skill_match(
            resume_skills=["python", "tensorflow", "docker"],
            required=["python", "tensorflow"],
            preferred=["docker"],
        )
        assert isinstance(result, dict)
        assert "skill_score" in result
        assert result["skill_score"] == 1.0
        assert len(result["matched_required"]) == 2
        assert len(result["missing_required"]) == 0

    def test_skill_match_missing(self, matcher):
        result = matcher.compute_skill_match(
            resume_skills=["python"],
            required=["python", "tensorflow", "docker"],
            preferred=["aws"],
        )
        assert len(result["missing_required"]) == 2
        assert "tensorflow" in result["missing_required"]
        assert "docker" in result["missing_required"]
        assert result["skill_score"] < 1.0

    def test_skill_match_additional(self, matcher):
        result = matcher.compute_skill_match(
            resume_skills=["python", "java", "go", "rust"],
            required=["python"],
            preferred=[],
        )
        assert "java" in result["additional"]
        assert "go" in result["additional"]
        assert "rust" in result["additional"]
        assert result["skill_score"] == 1.0

    def test_skill_match_empty_resume(self, matcher):
        result = matcher.compute_skill_match(
            resume_skills=[],
            required=["python", "docker"],
            preferred=[],
        )
        # Required score = 0/2 = 0.0, preferred empty = 1.0, weighted: 0*0.7 + 1.0*0.3 = 0.3
        assert result["skill_score"] == 0.3
        assert len(result["missing_required"]) == 2

    def test_skill_match_empty_requirements(self, matcher):
        result = matcher.compute_skill_match(
            resume_skills=["python", "docker"],
            required=[],
            preferred=[],
        )
        assert result["skill_score"] == 1.0
        assert len(result["additional"]) == 2


class TestExperienceMatch:
    """Test experience matching."""

    def test_experience_sufficient(self, matcher):
        score = matcher.compute_experience_match(5.0, 3.0)
        assert score == 1.0

    def test_experience_insufficient(self, matcher):
        score = matcher.compute_experience_match(1.0, 5.0)
        assert 0.0 <= score < 1.0

    def test_experience_exact_match(self, matcher):
        score = matcher.compute_experience_match(3.0, 3.0)
        assert score == 1.0

    def test_experience_no_requirement(self, matcher):
        score = matcher.compute_experience_match(5.0, 0.0)
        assert score == 1.0


class TestSemanticSimilarity:
    """Test semantic similarity."""

    def test_identical_text(self, matcher):
        score = matcher.compute_semantic_similarity("hello world", "hello world")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_different_text(self, matcher):
        score = matcher.compute_semantic_similarity(
            "machine learning engineer", "construction worker"
        )
        assert isinstance(score, float)

    def test_empty_text(self, matcher):
        score = matcher.compute_semantic_similarity("", "hello")
        assert score == 0.0


class TestMatchScore:
    """Test composite match scoring."""

    def test_match_score_range(self, matcher, sample_resume, sample_jd):
        score = matcher.compute_match_score(sample_resume, sample_jd)
        assert 0.0 <= score <= 1.0
        assert isinstance(score, float)


class TestBatchScoring:
    """Test batch scoring."""

    def test_batch_returns_sorted(self, matcher, sample_jd):
        resumes = [
            Resume(
                raw_text="Expert in all fields",
                filename=f"r{i}.pdf",
                skills=["python", "tensorflow", "docker", "aws", "kubernetes"],
                experience_years=10.0,
            )
            for i in range(3)
        ]
        results = matcher.batch_score_resumes(resumes, sample_jd, top_k=3)
        assert len(results) == 3
        # Scores should be non-increasing (sorted descending)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)
