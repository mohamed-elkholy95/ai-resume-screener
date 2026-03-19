"""Utility helpers for AI Resume Screener dashboard."""
from __future__ import annotations

import io
import json
import csv
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_score(score: float) -> str:
    """Return score as a two-decimal percentage string (e.g. '87.50%')."""
    return f"{score:.1f}%"


def format_file_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 ** 2:.1f} MB"


def format_experience(years: float | None) -> str:
    """Format years of experience as a readable string."""
    if years is None:
        return "Unknown"
    if years == 1.0:
        return "1 year"
    if years == int(years):
        return f"{int(years)} years"
    return f"{years:.1f} years"


# ---------------------------------------------------------------------------
# Score-to-label / color helpers
# ---------------------------------------------------------------------------

def get_score_label(score: float) -> str:
    """Return human-readable match label for a score (0–100)."""
    if score >= 75:
        return "Strong Match"
    if score >= 50:
        return "Moderate Match"
    if score >= 25:
        return "Weak Match"
    return "No Match"


def get_score_color(score: float) -> str:
    """Return hex color for a score (0–100)."""
    if score >= 75:
        return "#2ecc71"   # green
    if score >= 50:
        return "#f39c12"   # yellow/orange
    return "#e74c3c"       # red


def get_score_emoji(score: float) -> str:
    """Return emoji indicator for a score."""
    if score >= 75:
        return "🟢"
    if score >= 50:
        return "🟡"
    return "🔴"


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def export_to_csv(candidates: list[dict[str, Any]]) -> bytes:
    """Serialize candidate rankings to UTF-8 CSV bytes."""
    if not candidates:
        return b""
    output = io.StringIO()
    fieldnames = list(candidates[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in candidates:
        writer.writerow({k: v for k, v in row.items() if k in fieldnames})
    return output.getvalue().encode("utf-8")


def export_to_json(candidates: list[dict[str, Any]]) -> bytes:
    """Serialize candidate data to pretty-printed JSON bytes."""
    return json.dumps(candidates, indent=2, default=str).encode("utf-8")


def generate_pdf_report(candidates: list[dict[str, Any]], title: str = "Resume Screener Report") -> bytes:
    """Generate a simple PDF report for top candidates.

    Uses reportlab when available; falls back to plain-text PDF stub.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.enums import TA_CENTER

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                topMargin=0.75 * inch, bottomMargin=0.75 * inch)
        styles = getSampleStyleSheet()
        story = []

        # Title
        title_style = ParagraphStyle(
            "CustomTitle",
            parent=styles["Title"],
            fontSize=20,
            textColor=colors.HexColor("#2ecc71"),
            alignment=TA_CENTER,
            spaceAfter=12,
        )
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 0.25 * inch))

        # Table
        if candidates:
            table_data = [["Rank", "Name", "Score", "Skills %", "Status"]]
            for i, c in enumerate(candidates[:20], 1):
                table_data.append([
                    str(i),
                    str(c.get("name", "Unknown"))[:30],
                    f"{c.get('overall_score', 0):.1f}%",
                    f"{c.get('skill_match_pct', 0):.1f}%",
                    str(c.get("status", "Pending")),
                ])
            table = Table(table_data, colWidths=[0.5 * inch, 2.5 * inch, 1 * inch, 1 * inch, 1.2 * inch])
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#262730")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(table)

        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        # Minimal fallback: plain text encoded as PDF-like bytes
        text = f"{title}\n{'=' * len(title)}\n\n"
        for i, c in enumerate(candidates[:20], 1):
            text += (
                f"{i}. {c.get('name', 'Unknown')} — "
                f"Score: {c.get('overall_score', 0):.1f}% | "
                f"Status: {c.get('status', 'Pending')}\n"
            )
        return text.encode("utf-8")


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------

def get_candidates_df(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert candidates list to a display-ready DataFrame."""
    if not candidates:
        return pd.DataFrame(columns=["name", "overall_score", "skill_match_pct",
                                     "experience_years", "education", "status"])
    df = pd.DataFrame(candidates)
    # Ensure columns exist with defaults
    defaults = {
        "overall_score": 0.0,
        "skill_match_pct": 0.0,
        "experience_years": None,
        "education": "Unknown",
        "status": "Pending",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    df = df.sort_values("overall_score", ascending=False).reset_index(drop=True)
    df.index += 1  # 1-based rank
    return df
