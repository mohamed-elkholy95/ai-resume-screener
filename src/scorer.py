"""Composite scoring, ranking, and report generation for resume screening."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import MATCH_THRESHOLDS, MATCH_WEIGHTS, PROCESSED_DIR
from src.data_collection import JobDescription, Resume

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------


def get_match_label(score: float) -> str:
    """Map a composite score to a human-readable match label.

    Args:
        score: Composite score in [0, 1].

    Returns:
        One of ``"Strong Match"``, ``"Moderate Match"``, ``"Weak Match"``,
        or ``"No Match"``.
    """
    if score >= MATCH_THRESHOLDS["strong"]:
        return "Strong Match"
    if score >= MATCH_THRESHOLDS["moderate"]:
        return "Moderate Match"
    if score >= MATCH_THRESHOLDS["weak"]:
        return "Weak Match"
    return "No Match"


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------


class ResumeScorer:
    """Composite scoring and ranking pipeline.

    Orchestrates the :class:`~src.matcher.ResumeMatcher`,
    :class:`~src.ner_extractor.ResumeNER`, and optionally the
    :class:`~src.classifier.ResumeClassifier` to produce a full ranking.
    """

    def __init__(
        self,
        matcher: Any,
        ner: Any,
        classifier: Optional[Any] = None,
    ) -> None:
        """Initialise the scorer.

        Args:
            matcher: Configured :class:`~src.matcher.ResumeMatcher` instance.
            ner: Configured :class:`~src.ner_extractor.ResumeNER` instance.
            classifier: Optional trained :class:`~src.classifier.ResumeClassifier`.
                If ``None``, ML-based predictions are skipped.
        """
        self.matcher = matcher
        self.ner = ner
        self.classifier = classifier

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def generate_score_report(
        self, resume: Resume, jd: JobDescription
    ) -> Dict[str, Any]:
        """Generate a complete scoring report for a single candidate.

        Populates ``resume.entities``, ``resume.skills``,
        ``resume.experience_years``, and ``resume.education`` if not already
        set, then computes all component scores.

        Args:
            resume: Parsed resume (entities populated in-place if missing).
            jd: Target job description.

        Returns:
            Detailed report dict with composite score, component breakdown,
            skill analysis, and optional ML prediction.
        """
        # Ensure entities are extracted
        if not resume.entities:
            resume.entities = self.ner.extract_entities(resume.clean_text or resume.raw_text)

        if not resume.skills:
            resume.skills = self.ner.extract_skills(resume.clean_text or resume.raw_text)

        if resume.experience_years == 0.0:
            resume.experience_years = self.ner.extract_experience_years(
                resume.clean_text or resume.raw_text
            )

        if not resume.education:
            resume.education = resume.entities.get("EDUCATION", [])

        # Component scores
        skill_result = self.matcher.compute_skill_match(
            resume.skills,
            jd.required_skills,
            jd.preferred_skills,
        )
        experience_score = self.matcher.compute_experience_match(
            resume.experience_years,
            jd.min_experience,
        )
        education_text = " ".join(resume.education) if resume.education else ""
        education_score = self.matcher.compute_education_match(
            education_text,
            jd.education_level,
        )
        semantic_score = self.matcher.compute_semantic_similarity(
            resume.clean_text or resume.raw_text,
            jd.raw_text,
        )

        composite = (
            skill_result["skill_score"] * MATCH_WEIGHTS["skill_match"]
            + experience_score * MATCH_WEIGHTS["experience_match"]
            + education_score * MATCH_WEIGHTS["education_match"]
            + semantic_score * MATCH_WEIGHTS["semantic_similarity"]
        )
        composite = round(max(0.0, min(1.0, composite)), 4)

        report: Dict[str, Any] = {
            "candidate": resume.filename,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "composite_score": composite,
            "match_label": get_match_label(composite),
            "component_scores": {
                "skill_match": round(skill_result["skill_score"], 4),
                "experience_match": round(experience_score, 4),
                "education_match": round(education_score, 4),
                "semantic_similarity": round(semantic_score, 4),
            },
            "weights": MATCH_WEIGHTS,
            "skill_analysis": {
                "matched_required": skill_result.get("matched_required", []),
                "missing_required": skill_result.get("missing_required", []),
                "matched_preferred": skill_result.get("matched_preferred", []),
                "required_match_pct": round(skill_result.get("required_match", 0) * 100, 1),
                "preferred_match_pct": round(skill_result.get("preferred_match", 0) * 100, 1),
            },
            "experience": {
                "candidate_years": resume.experience_years,
                "required_years": jd.min_experience,
                "score": round(experience_score, 4),
            },
            "education": {
                "candidate_education": resume.education,
                "required_level": jd.education_level,
                "score": round(education_score, 4),
            },
        }

        # Optional ML prediction
        if self.classifier is not None:
            try:
                ml_result = self.classifier.predict(resume, jd)
                report["ml_prediction"] = ml_result
            except Exception as exc:
                logger.warning("ML prediction failed: %s", exc)
                report["ml_prediction"] = None

        return report

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_resumes(
        self,
        resumes: List[Resume],
        jd: JobDescription,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Score and rank a list of resumes against a job description.

        Args:
            resumes: List of :class:`~src.data_collection.Resume` objects.
            jd: Target job description.
            top_k: Maximum number of candidates to return. Pass 0 for all.

        Returns:
            List of score-report dicts sorted by descending composite score,
            each augmented with a ``rank`` field.
        """
        scored: List[Dict[str, Any]] = []

        for resume in resumes:
            try:
                report = self.generate_score_report(resume, jd)
                scored.append(report)
            except Exception as exc:
                logger.error("Error scoring '%s': %s", resume.filename, exc)
                scored.append(
                    {
                        "candidate": resume.filename,
                        "composite_score": 0.0,
                        "match_label": "No Match",
                        "error": str(exc),
                    }
                )

        scored.sort(key=lambda r: r.get("composite_score", 0.0), reverse=True)

        limit = top_k if top_k > 0 else len(scored)
        for rank, report in enumerate(scored[:limit], start=1):
            report["rank"] = rank

        logger.info(
            "Ranked %d resumes — top score: %.4f",
            len(resumes),
            scored[0].get("composite_score", 0.0) if scored else 0.0,
        )
        return scored[:limit]

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_rankings(
        self,
        rankings: List[Dict[str, Any]],
        format: str = "csv",
        output_path: Optional[str] = None,
    ) -> str:
        """Export ranked results to CSV, JSON, or Excel.

        Args:
            rankings: Output from :meth:`rank_resumes`.
            format: One of ``"csv"``, ``"json"``, or ``"excel"``.
            output_path: File path without extension. Defaults to
                ``<PROCESSED_DIR>/rankings_<timestamp>``.

        Returns:
            Absolute path of the saved file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_stem = str(PROCESSED_DIR / f"rankings_{timestamp}")
        stem = output_path or default_stem

        if format == "csv":
            return self._export_csv(rankings, f"{stem}.csv")
        if format == "json":
            return self._export_json(rankings, f"{stem}.json")
        if format == "excel":
            return self._export_excel(rankings, f"{stem}.xlsx")

        raise ValueError(f"Unsupported export format: '{format}'. Use csv/json/excel.")

    def _flatten_row(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten a score report into a single-level dict for tabular export."""
        comp = report.get("component_scores", {})
        skill_a = report.get("skill_analysis", {})
        exp = report.get("experience", {})
        edu = report.get("education", {})
        ml = report.get("ml_prediction") or {}

        return {
            "rank": report.get("rank", ""),
            "candidate": report.get("candidate", ""),
            "composite_score": report.get("composite_score", 0.0),
            "match_label": report.get("match_label", ""),
            "skill_match": comp.get("skill_match", 0.0),
            "experience_match": comp.get("experience_match", 0.0),
            "education_match": comp.get("education_match", 0.0),
            "semantic_similarity": comp.get("semantic_similarity", 0.0),
            "matched_required_skills": ", ".join(skill_a.get("matched_required", [])),
            "missing_required_skills": ", ".join(skill_a.get("missing_required", [])),
            "candidate_years": exp.get("candidate_years", 0.0),
            "required_years": exp.get("required_years", 0.0),
            "ml_prediction": ml.get("class_label", ""),
            "ml_confidence": ml.get("confidence", ""),
            "generated_at": report.get("generated_at", ""),
        }

    def _export_csv(self, rankings: List[Dict[str, Any]], file_path: str) -> str:
        rows = [self._flatten_row(r) for r in rankings]
        if not rows:
            logger.warning("No rankings to export.")
            Path(file_path).touch()
            return file_path

        with open(file_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        logger.info("CSV exported to %s (%d rows)", file_path, len(rows))
        return file_path

    def _export_json(self, rankings: List[Dict[str, Any]], file_path: str) -> str:
        with open(file_path, "w", encoding="utf-8") as fh:
            json.dump(rankings, fh, indent=2, default=str)
        logger.info("JSON exported to %s", file_path)
        return file_path

    def _export_excel(self, rankings: List[Dict[str, Any]], file_path: str) -> str:
        try:
            import pandas as pd  # type: ignore
        except ImportError as exc:
            raise ImportError("pandas is required for Excel export.") from exc

        rows = [self._flatten_row(r) for r in rankings]
        df = pd.DataFrame(rows)
        df.to_excel(file_path, index=False)
        logger.info("Excel exported to %s", file_path)
        return file_path

    # ------------------------------------------------------------------
    # Formatted report
    # ------------------------------------------------------------------

    def format_report_markdown(self, report: Dict[str, Any]) -> str:
        """Render a score report as a human-readable Markdown string.

        Args:
            report: Output of :meth:`generate_score_report`.

        Returns:
            Markdown-formatted string.
        """
        comp = report.get("component_scores", {})
        skill_a = report.get("skill_analysis", {})
        exp = report.get("experience", {})
        ml = report.get("ml_prediction") or {}

        matched = ", ".join(skill_a.get("matched_required", [])) or "—"
        missing = ", ".join(skill_a.get("missing_required", [])) or "—"

        ml_section = ""
        if ml:
            ml_section = (
                f"\n## ML Prediction\n"
                f"- Class: **{ml.get('class_label', '—')}** "
                f"(confidence: {ml.get('confidence', 0):.1%})\n"
            )

        return (
            f"# Candidate Score Report\n"
            f"**Candidate:** {report.get('candidate', '—')}\n"
            f"**Generated:** {report.get('generated_at', '—')}\n\n"
            f"## Overall Score: {report.get('composite_score', 0):.1%} — "
            f"**{report.get('match_label', '—')}**\n\n"
            f"## Component Breakdown\n"
            f"| Component            | Score  | Weight |\n"
            f"|----------------------|--------|--------|\n"
            f"| Skill Match          | {comp.get('skill_match', 0):.1%}  | 40%    |\n"
            f"| Experience Match     | {comp.get('experience_match', 0):.1%}  | 30%    |\n"
            f"| Education Match      | {comp.get('education_match', 0):.1%}  | 20%    |\n"
            f"| Semantic Similarity  | {comp.get('semantic_similarity', 0):.1%}  | 10%    |\n\n"
            f"## Skills Analysis\n"
            f"- ✅ **Matched required:** {matched}\n"
            f"- ❌ **Missing required:** {missing}\n"
            f"- Required match: {skill_a.get('required_match_pct', 0):.1f}%\n\n"
            f"## Experience\n"
            f"- Candidate: {exp.get('candidate_years', 0):.1f} yrs | "
            f"Required: {exp.get('required_years', 0):.1f} yrs\n"
            f"{ml_section}"
        )
