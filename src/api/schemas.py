"""Pydantic request and response models for the Resume Screener API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared / base models
# ---------------------------------------------------------------------------


class SkillAnalysis(BaseModel):
    """Skill match breakdown."""

    matched_required: List[str] = Field(
        default_factory=list, description="Required skills found in the resume."
    )
    missing_required: List[str] = Field(
        default_factory=list, description="Required skills absent from the resume."
    )
    matched_preferred: List[str] = Field(
        default_factory=list, description="Preferred skills found in the resume."
    )
    required_match_pct: float = Field(0.0, ge=0.0, le=100.0)
    preferred_match_pct: float = Field(0.0, ge=0.0, le=100.0)


class ComponentScores(BaseModel):
    """Individual component scores contributing to the composite."""

    skill_match: float = Field(..., ge=0.0, le=1.0)
    experience_match: float = Field(..., ge=0.0, le=1.0)
    education_match: float = Field(..., ge=0.0, le=1.0)
    semantic_similarity: float = Field(..., ge=0.0, le=1.0)


class MLPrediction(BaseModel):
    """ML classifier prediction."""

    class_label: str
    class_index: int = Field(..., ge=0, le=3)
    probabilities: Dict[str, float]
    confidence: float = Field(..., ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Job Description models
# ---------------------------------------------------------------------------


class JobDescriptionRequest(BaseModel):
    """Request body for submitting a job description."""

    title: str = Field(..., min_length=2, max_length=200, description="Job title.")
    raw_text: str = Field(
        ..., min_length=50, description="Full job description text."
    )
    required_skills: List[str] = Field(
        default_factory=list, description="Hard-requirement skills."
    )
    preferred_skills: List[str] = Field(
        default_factory=list, description="Nice-to-have skills."
    )
    min_experience: float = Field(
        0.0, ge=0.0, le=50.0, description="Minimum years of experience required."
    )
    education_level: str = Field(
        "", description="Minimum education level (e.g. 'Bachelor')."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobDescriptionResponse(BaseModel):
    """Response returned after submitting a job description."""

    jd_id: str = Field(..., description="Unique identifier for this JD.")
    title: str
    created_at: datetime
    required_skills: List[str]
    preferred_skills: List[str]
    min_experience: float
    education_level: str


# ---------------------------------------------------------------------------
# Resume models
# ---------------------------------------------------------------------------


class ResumeTextRequest(BaseModel):
    """Request body for submitting a resume as plain text."""

    text: str = Field(..., min_length=100, description="Full resume text.")
    filename: str = Field("uploaded_resume.txt", description="Logical filename.")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ResumeResponse(BaseModel):
    """Response returned after a resume is submitted."""

    resume_id: str
    filename: str
    skills_found: int = Field(..., description="Number of skills extracted.")
    experience_years: float
    created_at: datetime


# ---------------------------------------------------------------------------
# Screening / scoring models
# ---------------------------------------------------------------------------


class ScreenRequest(BaseModel):
    """Request to screen a batch of resumes against a job description."""

    jd_id: str = Field(..., description="ID of the target job description.")
    resume_ids: List[str] = Field(
        ..., min_length=1, description="List of resume IDs to screen."
    )
    top_k: int = Field(
        10, ge=1, le=200, description="Maximum number of candidates to return."
    )

    @field_validator("resume_ids")
    @classmethod
    def no_duplicate_ids(cls, v: List[str]) -> List[str]:
        if len(v) != len(set(v)):
            raise ValueError("Duplicate resume_ids are not allowed.")
        return v


class CandidateScore(BaseModel):
    """Score summary for a single candidate in a ranking."""

    rank: int
    resume_id: str
    filename: str
    composite_score: float = Field(..., ge=0.0, le=1.0)
    match_label: str
    component_scores: ComponentScores
    skill_analysis: SkillAnalysis
    ml_prediction: Optional[MLPrediction] = None


class ScreenResponse(BaseModel):
    """Response for a screening request."""

    jd_id: str
    jd_title: str
    total_screened: int
    rankings: List[CandidateScore]
    generated_at: datetime


# ---------------------------------------------------------------------------
# Detail / report models
# ---------------------------------------------------------------------------


class ScoreDetailRequest(BaseModel):
    """Request for a detailed score report for a single candidate."""

    resume_id: str
    jd_id: str


class ScoreDetailResponse(BaseModel):
    """Detailed score report for a single candidate."""

    resume_id: str
    jd_id: str
    composite_score: float
    match_label: str
    component_scores: ComponentScores
    skill_analysis: SkillAnalysis
    experience: Dict[str, Any]
    education: Dict[str, Any]
    ml_prediction: Optional[MLPrediction] = None
    markdown_report: str = ""
    generated_at: datetime


class ExplainRequest(BaseModel):
    """Request an explainability report for a resume–JD pair."""

    resume_id: str
    jd_id: str
    method: str = Field("lime", description="Explanation method: 'lime' or 'shap'.")
    top_features: int = Field(10, ge=1, le=50)


class ExplainResponse(BaseModel):
    """Explainability output for a prediction."""

    resume_id: str
    jd_id: str
    prediction: MLPrediction
    top_features: List[List[Any]]  # List of (feature_name, importance) pairs
    explanation_text: str
    generated_at: datetime


# ---------------------------------------------------------------------------
# Export models
# ---------------------------------------------------------------------------


class ExportRequest(BaseModel):
    """Request to export ranking results."""

    jd_id: str
    resume_ids: Optional[List[str]] = None
    format: str = Field("csv", description="Export format: 'csv', 'json', or 'excel'.")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        if v not in ("csv", "json", "excel"):
            raise ValueError("format must be one of: csv, json, excel")
        return v


class ExportResponse(BaseModel):
    """Response containing the path/URL to the exported file."""

    file_path: str
    format: str
    row_count: int
    generated_at: datetime


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """API health-check response."""

    status: str = "ok"
    version: str
    models_loaded: Dict[str, bool]
    timestamp: datetime
