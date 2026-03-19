"""Global sidebar components for AI Resume Screener dashboard."""
from __future__ import annotations

import streamlit as st

COLOR_STRONG = "#2ecc71"
COLOR_MODERATE = "#f39c12"
COLOR_WEAK = "#e74c3c"
COLOR_CARD = "#262730"


def render_settings_sidebar() -> dict:
    """Render global settings panel in the sidebar.

    Returns a dict with keys:
        - similarity_threshold: float (0–100)
        - classification_model: str
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Model Settings")

    classification_model = st.sidebar.selectbox(
        "Classification Model",
        options=[
            "TF-IDF + Cosine Similarity",
            "Sentence-BERT (all-MiniLM-L6-v2)",
            "OpenAI text-embedding-3-small",
            "Keyword Matching (Fast)",
        ],
        index=0,
        help="Model used to compute resume–JD similarity.",
    )

    similarity_threshold = st.sidebar.slider(
        "Similarity Threshold (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Resumes scoring at or above this threshold are flagged as strong matches.",
    )

    return {
        "classification_model": classification_model,
        "similarity_threshold": similarity_threshold,
    }


def render_jd_summary(jd: dict | None) -> None:
    """Render job description summary card in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Job Description")

    if not jd or not jd.get("title"):
        st.sidebar.markdown(
            """
            <div style="
                background-color: #1e1e2e;
                border: 1px dashed #555;
                border-radius: 8px;
                padding: 14px;
                color: #888;
                font-size: 13px;
                text-align: center;
            ">
                📄 Upload a JD to start
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    title = jd.get("title", "Untitled Position")
    skills = jd.get("required_skills", [])
    experience = jd.get("experience_years")
    education = jd.get("education_level", "")

    skills_preview = ", ".join(skills[:5])
    if len(skills) > 5:
        skills_preview += f" +{len(skills) - 5} more"

    exp_text = f"{experience}+ years" if experience else "Not specified"

    st.sidebar.markdown(
        f"""
        <div style="
            background-color: {COLOR_CARD};
            border-radius: 10px;
            padding: 14px;
            font-size: 13px;
        ">
            <div style="font-weight: 700; font-size: 15px; color: #fafafa; margin-bottom: 8px;">
                {title}
            </div>
            <div style="color: #aaa; margin-bottom: 4px;">
                🎯 <b>Skills:</b> <span style="color: #fafafa;">{skills_preview or 'None set'}</span>
            </div>
            <div style="color: #aaa; margin-bottom: 4px;">
                📅 <b>Experience:</b> <span style="color: #fafafa;">{exp_text}</span>
            </div>
            <div style="color: #aaa;">
                🎓 <b>Education:</b> <span style="color: #fafafa;">{education or 'Not specified'}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_stats(stats: dict) -> None:
    """Render quick statistics cards in the sidebar.

    Expected keys in stats:
        - uploaded: int
        - processed: int
        - ranked: int
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Quick Stats")

    uploaded = stats.get("uploaded", 0)
    processed = stats.get("processed", 0)
    ranked = stats.get("ranked", 0)

    def _stat_card(icon: str, label: str, value: int, color: str = "#fafafa") -> str:
        return f"""
        <div style="
            background-color: {COLOR_CARD};
            border-radius: 8px;
            padding: 10px 14px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        ">
            <span style="font-size: 13px; color: #aaa;">{icon} {label}</span>
            <span style="font-weight: 700; font-size: 16px; color: {color};">{value}</span>
        </div>
        """

    html = (
        _stat_card("📁", "Uploaded", uploaded)
        + _stat_card("⚙️", "Processed", processed, COLOR_MODERATE)
        + _stat_card("🏆", "Ranked", ranked, COLOR_STRONG)
    )
    st.sidebar.markdown(html, unsafe_allow_html=True)
