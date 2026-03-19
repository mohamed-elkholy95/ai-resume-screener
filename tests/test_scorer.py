"""Tests for the scorer module."""
import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from src.scorer import ResumeScorer, get_match_label
from src.data_collection import Resume, JobDescription


@pytest.fixture
def mock_matcher():
    m = MagicMock()
    m.compute_match_score.return_value = 0.85
    m.compute_skill_match.return_value = {"skill_score": 0.9}
    m.compute_experience_match.return_value = 0.8
    m.compute_education_match.return_value = 0.7
    m.compute_semantic_similarity.return_value = 0.6
    return m


@pytest.fixture
def mock_ner():
    m = MagicMock()
    m.extract_entities.return_value = {"skills": ["python"], "companies": ["TechCorp"]}
    return m


@pytest.fixture
def sample_resume():
    return Resume(
        raw_text="ML engineer with Python, TensorFlow, Docker, SQL. 5 years experience. BS CS.",
        filename="test.pdf",
        skills=["python", "tensorflow", "docker", "sql"],
        experience_years=5.0,
        education=["BS Computer Science"],
    )


@pytest.fixture
def sample_jd():
    return JobDescription(
        raw_text="Senior ML Engineer with Python, TensorFlow, Docker, SQL",
        title="Senior ML Engineer",
        required_skills=["python", "tensorflow", "docker"],
        preferred_skills=["sql", "aws"],
        min_experience=3.0,
        education_level="bachelor",
    )


class TestGetMatchLabel:
    """Test match label generation."""

    @pytest.mark.parametrize("score,expected", [
        (0.85, "Strong"),
        (0.60, "Moderate"),
        (0.35, "Weak"),
        (0.15, "No"),
        (0.75, "Strong"),
        (0.50, "Moderate"),
        (0.30, "Weak"),
        (0.0, "No"),
        (1.0, "Strong"),
    ])
    def test_label(self, score, expected):
        label = get_match_label(score)
        assert expected in label


class TestResumeScorer:
    """Test ResumeScorer class."""

    def test_init(self, mock_matcher, mock_ner):
        scorer = ResumeScorer(matcher=mock_matcher, ner=mock_ner)
        assert scorer.matcher is mock_matcher
        assert scorer.ner is mock_ner

    def test_init_with_classifier(self, mock_matcher, mock_ner):
        classifier = MagicMock()
        scorer = ResumeScorer(matcher=mock_matcher, ner=mock_ner, classifier=classifier)
        assert scorer.classifier is classifier

    def test_generate_score_report(self, mock_matcher, mock_ner, sample_resume, sample_jd):
        scorer = ResumeScorer(matcher=mock_matcher, ner=mock_ner)
        report = scorer.generate_score_report(sample_resume, sample_jd)
        assert isinstance(report, dict)
        assert "composite_score" in report
        assert 0.0 <= report["composite_score"] <= 1.0

    def test_generate_score_report_fields(self, mock_matcher, mock_ner, sample_resume, sample_jd):
        scorer = ResumeScorer(matcher=mock_matcher, ner=mock_ner)
        report = scorer.generate_score_report(sample_resume, sample_jd)
        assert "component_scores" in report
        assert "match_label" in report

    def test_export_rankings_csv(self, mock_matcher, mock_ner, tmp_path):
        scorer = ResumeScorer(matcher=mock_matcher, ner=mock_ner)
        rankings = [
            {
                "rank": 1,
                "candidate": "a.pdf",
                "overall_score": 0.9,
            },
            {
                "rank": 2,
                "candidate": "b.pdf",
                "overall_score": 0.6,
            },
        ]
        output = str(tmp_path / "rankings")
        path = scorer.export_rankings(rankings, format="csv", output_path=output)
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "a.pdf" in content
        assert "b.pdf" in content

    def test_rank_resumes(self, mock_matcher, mock_ner, sample_jd):
        scorer = ResumeScorer(matcher=mock_matcher, ner=mock_ner)
        resumes = [
            Resume(raw_text="test", filename=f"r{i}.pdf",
                   skills=["python"], experience_years=5.0)
            for i in range(3)
        ]
        results = scorer.rank_resumes(resumes, sample_jd, top_k=2)
        assert isinstance(results, list)
