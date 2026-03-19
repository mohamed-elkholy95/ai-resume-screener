"""Score card rendering components for AI Resume Screener dashboard."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from components.utils import get_score_color, get_score_label, format_score

COLOR_BG = "#0e1117"
COLOR_CARD = "#262730"
COLOR_STRONG = "#2ecc71"
COLOR_MODERATE = "#f39c12"
COLOR_WEAK = "#e74c3c"


def render_score_gauge(score: float, label: str = "") -> None:
    """Render a large Plotly gauge showing 0–100% with color coding."""
    color = get_score_color(score)
    match_label = label or get_score_label(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix="%", font=dict(size=36, color=color)),
        title=dict(text=match_label, font=dict(size=16, color=color)),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickwidth=1,
                tickcolor="#fafafa",
                tickfont=dict(color="#fafafa"),
            ),
            bar=dict(color=color, thickness=0.35),
            bgcolor=COLOR_CARD,
            borderwidth=0,
            steps=[
                dict(range=[0, 25],   color="#3d1a1a"),
                dict(range=[25, 50],  color="#3d2e1a"),
                dict(range=[50, 75],  color="#2d3a1a"),
                dict(range=[75, 100], color="#1a3d1a"),
            ],
            threshold=dict(
                line=dict(color=color, width=3),
                thickness=0.8,
                value=score,
            ),
        ),
    ))

    fig.update_layout(
        paper_bgcolor=COLOR_BG,
        font=dict(color="#fafafa"),
        height=220,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_match_badge(score: float) -> None:
    """Render a color-coded match label badge using st.markdown."""
    color = get_score_color(score)
    label = get_score_label(score)

    if score >= 75:
        emoji = "✅"
    elif score >= 50:
        emoji = "⚠️"
    elif score >= 25:
        emoji = "🔶"
    else:
        emoji = "❌"

    st.markdown(
        f"""
        <div style="
            display: inline-block;
            background-color: {color}22;
            border: 1.5px solid {color};
            color: {color};
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 15px;
            letter-spacing: 0.5px;
        ">
            {emoji}&nbsp;{label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_breakdown(breakdown: dict[str, float]) -> None:
    """Render a card with labeled progress bars for each score component."""
    component_labels = {
        "skill_match": "🎯 Skill Match",
        "experience_match": "📅 Experience Match",
        "education_match": "🎓 Education Match",
        "semantic_similarity": "🔗 Semantic Similarity",
        "overall": "⭐ Overall Score",
    }

    st.markdown(
        """
        <style>
        .score-row { margin-bottom: 12px; }
        .score-label { font-size: 13px; font-weight: 600; margin-bottom: 4px; color: #fafafa; }
        .progress-bg {
            background-color: #1e1e2e;
            border-radius: 6px;
            height: 10px;
            width: 100%;
            overflow: hidden;
        }
        .progress-fill {
            height: 10px;
            border-radius: 6px;
            transition: width 0.4s ease;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for key, label in component_labels.items():
        if key not in breakdown:
            continue
        val = breakdown[key]
        color = get_score_color(val)
        st.markdown(
            f"""
            <div class="score-row">
                <div class="score-label">{label} — <span style="color:{color}">{val:.1f}%</span></div>
                <div class="progress-bg">
                    <div class="progress-fill" style="width:{val}%; background-color:{color};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_skills_comparison(
    matched: list[str],
    missing: list[str],
    additional: list[str],
) -> None:
    """Render three-column skills comparison: Matched / Missing / Additional."""

    def _badge(skill: str, color: str, bg_alpha: str = "33") -> str:
        return (
            f'<span style="'
            f"display:inline-block; margin:3px 4px; padding:4px 10px; "
            f"border-radius:14px; font-size:12px; font-weight:600; "
            f"background-color:{color}{bg_alpha}; border:1px solid {color}; "
            f'color:{color};">{skill}</span>'
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### ✅ Matched Skills")
        if matched:
            badges = "".join(_badge(s, COLOR_STRONG) for s in matched)
            st.markdown(f'<div style="line-height:2;">{badges}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#888;">None matched</span>', unsafe_allow_html=True)

    with col2:
        st.markdown("#### ❌ Missing Skills")
        if missing:
            badges = "".join(_badge(s, COLOR_WEAK) for s in missing)
            st.markdown(f'<div style="line-height:2;">{badges}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#888;">None missing 🎉</span>', unsafe_allow_html=True)

    with col3:
        st.markdown("#### ➕ Additional Skills")
        if additional:
            badges = "".join(_badge(s, "#aaaaaa") for s in additional)
            st.markdown(f'<div style="line-height:2;">{badges}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#888;">None detected</span>', unsafe_allow_html=True)
