"""
conftest.py — pytest fixtures for AI Resume Screener tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src/ is importable
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection import JobDescription, Resume


# ---------------------------------------------------------------------------
# Text fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_resume_text() -> str:
    """A realistic mock resume covering skills, education, and experience."""
    return """
Jane Smith
jane.smith@example.com | [PHONE] | linkedin.com/in/janesmith

SUMMARY
Senior Machine Learning Engineer with 7 years of experience building
production ML systems. Strong background in Python, PyTorch, and MLOps.

SKILLS
- Programming: Python, TypeScript, SQL, Bash
- ML/AI: PyTorch, TensorFlow, Scikit-learn, Hugging Face, LangChain, XGBoost
- Data Engineering: Apache Spark, Apache Airflow, dbt, Snowflake, PostgreSQL
- Cloud/DevOps: AWS, Docker, Kubernetes, GitHub Actions, Terraform
- Web: FastAPI, React, REST API

EXPERIENCE

Senior ML Engineer — TechCorp Inc. (2021 – Present)
- Led development of a real-time recommendation engine serving 10M+ users
- Reduced model inference latency by 40% using ONNX quantization
- Mentored a team of 4 junior engineers

ML Engineer — DataStartup LLC (2019 – 2021)
- Built end-to-end NLP pipeline for document classification (BERT fine-tuning)
- Deployed models to AWS SageMaker with CI/CD via GitHub Actions

Data Scientist — Analytics Co. (2017 – 2019)
- Developed churn prediction models using XGBoost and LightGBM
- Created automated reporting dashboards with Python and Tableau

EDUCATION
M.S. Computer Science — Stanford University (2017)
B.S. Electrical Engineering — UC Berkeley (2015)

CERTIFICATIONS
AWS Certified Machine Learning Specialty
Google Cloud Professional Data Engineer
""".strip()


@pytest.fixture
def sample_jd_text() -> str:
    """A realistic job description for a Senior ML Engineer role."""
    return """
Senior Machine Learning Engineer

We are looking for a Senior ML Engineer to join our AI platform team.

Requirements:
- 5+ years of experience in machine learning or data science
- Strong Python skills and experience with PyTorch or TensorFlow
- Experience with cloud platforms (AWS, GCP, or Azure)
- Knowledge of MLOps practices (model versioning, CI/CD, monitoring)
- Bachelor's degree in Computer Science, Engineering, or related field

Preferred:
- Experience with Kubernetes and container orchestration
- Familiarity with LLMs and RAG architectures
- Experience with distributed data processing (Spark, Kafka)
- Master's degree or PhD preferred

Responsibilities:
- Design and implement scalable ML training and inference pipelines
- Collaborate with data engineers and product teams
- Mentor junior ML engineers
""".strip()


@pytest.fixture
def sample_resumes_list(sample_resume_text: str) -> list[Resume]:
    """A list of 5 resumes with varying quality/relevance."""
    return [
        # Strong match
        Resume(
            raw_text=sample_resume_text,
            filename="jane_smith.txt",
            skills=["python", "pytorch", "tensorflow", "aws", "kubernetes", "docker"],
            education=["M.S. Computer Science", "B.S. Electrical Engineering"],
            experience_years=7.0,
        ),
        # Moderate match — fewer skills, less experience
        Resume(
            raw_text="Junior ML engineer with 2 years experience. Python, scikit-learn, SQL.",
            filename="bob_jones.txt",
            skills=["python", "scikit-learn", "sql"],
            education=["B.S. Computer Science"],
            experience_years=2.0,
        ),
        # Weak match — different domain
        Resume(
            raw_text="Frontend developer with 5 years React and TypeScript experience.",
            filename="alice_wang.txt",
            skills=["react", "typescript", "javascript", "css", "html"],
            education=["B.S. Information Technology"],
            experience_years=5.0,
        ),
        # Moderate match — right skills, no formal education listed
        Resume(
            raw_text="Self-taught engineer. Python, PyTorch, AWS, Docker. 4 years experience.",
            filename="carlos_diaz.txt",
            skills=["python", "pytorch", "aws", "docker"],
            education=[],
            experience_years=4.0,
        ),
        # No match — unrelated field
        Resume(
            raw_text="Accounting professional with 10 years in finance and Excel.",
            filename="diana_patel.txt",
            skills=["excel", "accounting", "financial analysis"],
            education=["B.S. Accounting"],
            experience_years=10.0,
        ),
    ]


@pytest.fixture
def sample_jd(sample_jd_text: str) -> JobDescription:
    """A parsed JobDescription fixture."""
    return JobDescription(
        raw_text=sample_jd_text,
        title="Senior ML Engineer",
        required_skills=["python", "pytorch", "aws"],
        preferred_skills=["kubernetes", "spark", "llm"],
        min_experience=5.0,
        education_level="Bachelor's",
    )


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_matcher() -> MagicMock:
    """A MagicMock standing in for ResumeMatcher with pre-configured return values."""
    mock = MagicMock()
    mock.compute_match_score.return_value = 0.85
    mock.compute_skill_match.return_value = {
        "matched_required": ["python", "pytorch"],
        "missing_required": ["aws"],
        "matched_preferred": ["kubernetes"],
        "missing_preferred": ["spark"],
        "additional": ["tensorflow"],
        "skill_score": 0.78,
    }
    mock.compute_experience_match.return_value = 1.0
    mock.compute_education_match.return_value = 1.0
    mock.compute_semantic_similarity.return_value = 0.72
    mock.batch_score_resumes.return_value = []
    return mock


@pytest.fixture
def mock_classifier() -> MagicMock:
    """A MagicMock standing in for ResumeClassifier."""
    mock = MagicMock()
    mock.predict.return_value = {
        "label": "suitable",
        "probability": 0.88,
        "confidence": 0.88,
    }
    mock.explain_prediction.return_value = {
        "model_type": "logistic_regression",
        "top_features": [
            {"feature": "python", "contribution": 0.45},
            {"feature": "pytorch", "contribution": 0.32},
        ],
    }
    mock._is_trained = True
    return mock
