"""
main.py — FastAPI application entry point for the AI Resume Screener API.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

# Ensure src/ is on the path when running from the api/ subdirectory
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fastapi import FastAPI, HTTPException, Request  # type: ignore[import]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import]
from fastapi.responses import JSONResponse  # type: ignore[import]

from api.schemas import (
    BatchScoreResponse,
    ErrorResponse,
    JobDescriptionInput,
    MatchScoreRequest,
    MatchScoreResponse,
    ResumeUpload,
)
from data_collection import JobDescription, JobDescriptionParser, Resume, ResumeParser
from matcher import ResumeMatcher
from scorer import ResumeScorer, get_match_label

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application state (initialized at startup)
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize heavy components once at startup, clean up at shutdown."""
    logger.info("Starting AI Resume Screener API — initializing components…")
    _state["resume_parser"] = ResumeParser()
    _state["jd_parser"] = JobDescriptionParser()
    _state["matcher"] = ResumeMatcher()
    _state["scorer"] = ResumeScorer()
    logger.info("All components initialized successfully")
    yield
    logger.info("Shutting down AI Resume Screener API")
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI Resume Screener API",
    description=(
        "REST API for scoring, ranking, and classifying resumes against "
        "job descriptions using NLP and ML techniques."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins in development; restrict in production via env var
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", response_model=dict, tags=["System"])
async def health_check() -> dict[str, Any]:
    """Return API health status and component availability."""
    return {
        "status": "healthy",
        "components": {
            "resume_parser": "ready" if "resume_parser" in _state else "not_initialized",
            "jd_parser": "ready" if "jd_parser" in _state else "not_initialized",
            "matcher": "ready" if "matcher" in _state else "not_initialized",
            "scorer": "ready" if "scorer" in _state else "not_initialized",
        },
    }


# ---------------------------------------------------------------------------
# Scoring endpoints
# ---------------------------------------------------------------------------


@app.post("/score", response_model=MatchScoreResponse, tags=["Scoring"])
async def score_resume(request: MatchScoreRequest) -> MatchScoreResponse:
    """Score a single resume against a job description.

    Returns an overall match score, label, breakdown, and recommendations.
    """
    matcher: ResumeMatcher = _state["matcher"]
    scorer: ResumeScorer = _state["scorer"]
    resume_parser: ResumeParser = _state["resume_parser"]
    jd_parser: JobDescriptionParser = _state["jd_parser"]

    resume = resume_parser.parse_text(request.resume_text, filename="api_upload.txt")
    jd = jd_parser.parse_text(request.jd_text, title="API Request")

    report = scorer.generate_score_report(resume, jd)

    return MatchScoreResponse(
        score=report["overall_score"],
        label=report["label"],
        breakdown=report["components"],
        recommendations=report["recommendations"],
    )


@app.post("/batch-score", response_model=BatchScoreResponse, tags=["Scoring"])
async def batch_score_resumes(
    resumes: list[ResumeUpload],
    jd_input: JobDescriptionInput,
) -> BatchScoreResponse:
    """Score multiple resumes against a single job description.

    Returns ranked results and summary statistics.
    """
    resume_parser: ResumeParser = _state["resume_parser"]
    jd_parser: JobDescriptionParser = _state["jd_parser"]
    scorer: ResumeScorer = _state["scorer"]

    parsed_resumes: list[Resume] = [
        resume_parser.parse_text(r.text, filename=r.filename) for r in resumes
    ]
    jd = jd_parser.parse_text(jd_input.text, title=jd_input.title)
    # Attach skills/experience from input if provided
    jd.required_skills = jd_input.required_skills
    jd.preferred_skills = jd_input.preferred_skills
    jd.min_experience = jd_input.min_experience
    jd.education_level = jd_input.education_level

    rankings = scorer.rank_resumes(parsed_resumes, jd, top_k=len(parsed_resumes))
    strong_matches = sum(1 for r in rankings if r.score >= 0.75)

    results = [
        {
            "rank": r.rank,
            "filename": r.resume.filename,
            "score": r.score,
            "label": r.label,
        }
        for r in rankings
    ]

    return BatchScoreResponse(
        results=results,
        total=len(rankings),
        strong_matches=strong_matches,
    )
