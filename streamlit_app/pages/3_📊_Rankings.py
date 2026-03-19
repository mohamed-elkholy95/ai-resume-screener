"""Page 3 — Rankings dashboard with charts and export."""
from __future__ import annotations

import os
import sys
from typing import Any

import pandas as pd
import streamlit as st

_PAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_PAGE_DIR)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from components.sidebar import render_settings_sidebar, render_jd_summary, render_quick_stats
from components.charts import (
    plot_score_distribution,
    plot_skills_coverage,
    plot_experience_distribution,
    plot_skills_gap,
)
from components.utils import (
    get_score_color,
    get_score_label,
    get_score_emoji,
    export_to_csv,
    export_to_json,
    generate_pdf_report,
    format_experience,
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:12px 0 4px 0;">
            <div style="font-size:28px;">🤖</div>
            <div style="font-size:17px;font-weight:800;color:#2ecc71;">AI Resume Screener</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_jd_summary(st.session_state.get("jd"))
    settings = render_settings_sidebar()
    st.session_state["settings"] = settings
    candidates_all = st.session_state.get("candidates", [])
    resumes = st.session_state.get("resumes", [])
    render_quick_stats({
        "uploaded": len(resumes),
        "processed": len(candidates_all),
        "ranked": len(candidates_all),
    })

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔎 Filters")

    score_range = st.sidebar.slider("Score Range (%)", 0, 100, (0, 100), step=5)
    exp_range = st.sidebar.slider("Experience (years)", 0, 30, (0, 30), step=1)

    jd = st.session_state.get("jd")
    required_skills = jd.get("required_skills", []) if jd else []

    min_skill_count = st.sidebar.slider(
        f"Min skills matched (of {len(required_skills)})",
        0, max(len(required_skills), 1), 0,
    ) if required_skills else 0

    sort_by = st.sidebar.selectbox(
        "Sort By",
        ["Score (High→Low)", "Skills Match", "Experience", "Name (A→Z)"],
    )

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("📊 Rankings Dashboard")

candidates: list[dict[str, Any]] = st.session_state.get("candidates", [])

if not candidates:
    st.info("No candidates ranked yet. Complete steps 1 and 2 first.", icon="📋")
    st.stop()

# ---------------------------------------------------------------------------
# Apply filters
# ---------------------------------------------------------------------------
def _filter(c: dict) -> bool:
    score = c.get("overall_score", 0)
    exp = c.get("experience_years") or 0
    matched_count = len(c.get("matched_skills", []))

    if not (score_range[0] <= score <= score_range[1]):
        return False
    if not (exp_range[0] <= exp <= exp_range[1]):
        return False
    if required_skills and matched_count < min_skill_count:
        return False
    return True

filtered = [c for c in candidates if _filter(c)]

# Sort
sort_key_map = {
    "Score (High→Low)": lambda c: -c.get("overall_score", 0),
    "Skills Match": lambda c: -c.get("skill_match_pct", 0),
    "Experience": lambda c: -(c.get("experience_years") or 0),
    "Name (A→Z)": lambda c: c.get("name", "").lower(),
}
filtered.sort(key=sort_key_map.get(sort_by, sort_key_map["Score (High→Low)"]))

# ---------------------------------------------------------------------------
# Top metrics bar
# ---------------------------------------------------------------------------
threshold = settings.get("similarity_threshold", 50)
strong_count = sum(1 for c in candidates if c.get("overall_score", 0) >= 75)
shortlisted = sum(1 for c in candidates if c.get("status") == "Shortlisted")
avg_score = sum(c.get("overall_score", 0) for c in candidates) / len(candidates) if candidates else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("📋 Total Resumes", len(candidates))
m2.metric("📈 Avg Score", f"{avg_score:.1f}%")
m3.metric("💚 Strong Matches", strong_count)
m4.metric("✅ Shortlisted", shortlisted)

st.markdown("---")

# ---------------------------------------------------------------------------
# Export buttons
# ---------------------------------------------------------------------------
exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 1])
with exp_col1:
    csv_bytes = export_to_csv(filtered)
    st.download_button("📥 Export CSV", data=csv_bytes, file_name="rankings.csv",
                       mime="text/csv", use_container_width=True)
with exp_col2:
    json_bytes = export_to_json(filtered)
    st.download_button("📥 Export JSON", data=json_bytes, file_name="rankings.json",
                       mime="application/json", use_container_width=True)
with exp_col3:
    pdf_bytes = generate_pdf_report(filtered, title=f"Top Candidates — {jd['title'] if jd else 'Report'}")
    st.download_button("📥 PDF Report", data=pdf_bytes, file_name="top_candidates.pdf",
                       mime="application/pdf", use_container_width=True)

# ---------------------------------------------------------------------------
# Rankings table
# ---------------------------------------------------------------------------
st.markdown(f"### 🏆 Rankings ({len(filtered)} shown)")

if not filtered:
    st.warning("No candidates match the current filters.")
else:
    # Header row
    hcols = st.columns([0.5, 3, 2, 2, 1.5, 2, 3])
    for col, label in zip(hcols, ["#", "Name", "Score", "Skills %", "Exp.", "Status", "Actions"]):
        col.markdown(f"**{label}**")
    st.markdown("<hr style='margin:2px 0;border-color:#30363d;'>", unsafe_allow_html=True)

    # Track status updates in session state
    if "candidate_statuses" not in st.session_state:
        st.session_state["candidate_statuses"] = {}

    for rank, candidate in enumerate(filtered, 1):
        cid = candidate["id"]
        score = candidate.get("overall_score", 0)
        color = get_score_color(score)
        current_status = st.session_state["candidate_statuses"].get(cid, candidate.get("status", "Pending"))

        row = st.columns([0.5, 3, 2, 2, 1.5, 2, 3])

        row[0].markdown(f"**{rank}**")
        row[1].markdown(f"👤 **{candidate.get('name', 'Unknown')}**")

        # Score with mini progress bar
        row[2].markdown(
            f"""
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="color:{color};font-weight:700;">{score:.1f}%</span>
                <div style="flex:1;background:#1e1e2e;border-radius:4px;height:6px;">
                    <div style="width:{score}%;background:{color};height:6px;border-radius:4px;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sm = candidate.get("skill_match_pct", 0)
        row[3].markdown(
            f'<span style="color:{get_score_color(sm)};font-weight:600;">{sm:.1f}%</span>',
            unsafe_allow_html=True,
        )

        exp = candidate.get("experience_years")
        row[4].markdown(format_experience(exp))

        # Status badge
        status_colors = {
            "Shortlisted": "#2ecc71",
            "Strong Match": "#2ecc71",
            "Review": "#f39c12",
            "Weak Match": "#e74c3c",
            "Rejected": "#e74c3c",
            "Pending": "#888",
        }
        sc = status_colors.get(current_status, "#888")
        row[5].markdown(
            f'<span style="color:{sc};font-weight:600;">{current_status}</span>',
            unsafe_allow_html=True,
        )

        # Action buttons
        action_cols = row[6].columns(3)
        if action_cols[0].button("✅", key=f"sl_{cid}", help="Shortlist"):
            st.session_state["candidate_statuses"][cid] = "Shortlisted"
            st.rerun()
        if action_cols[1].button("⏸", key=f"rv_{cid}", help="Review"):
            st.session_state["candidate_statuses"][cid] = "Review"
            st.rerun()
        if action_cols[2].button("❌", key=f"rj_{cid}", help="Reject"):
            st.session_state["candidate_statuses"][cid] = "Rejected"
            st.rerun()

        st.markdown("<hr style='margin:2px 0;border-color:#1e1e2e;'>", unsafe_allow_html=True)

    # Apply status updates to candidates for detail page navigation
    for c in candidates:
        cid = c["id"]
        if cid in st.session_state["candidate_statuses"]:
            c["status"] = st.session_state["candidate_statuses"][cid]

    # View detail link
    st.markdown("---")
    candidate_names = [f"{i+1}. {c.get('name', 'Unknown')}" for i, c in enumerate(filtered)]
    selected_name = st.selectbox("🔍 View Candidate Detail:", ["— select —"] + candidate_names)
    if selected_name != "— select —":
        idx = int(selected_name.split(".")[0]) - 1
        selected = filtered[idx]
        st.session_state["selected_candidate"] = selected
        st.success(f"✅ Selected: **{selected['name']}** — Navigate to **4 🔍 Candidate Detail**")

# ---------------------------------------------------------------------------
# Charts section
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📈 Analytics")

tab_dist, tab_skills, tab_exp, tab_gap = st.tabs([
    "Score Distribution", "Skills Coverage", "Experience Distribution", "Skills Gap"
])

scores = [c.get("overall_score", 0) for c in candidates]

with tab_dist:
    fig = plot_score_distribution(scores)
    st.plotly_chart(fig, use_container_width=True)

with tab_skills:
    # Compute per-skill coverage across all resumes
    if required_skills:
        skill_coverage = {}
        for skill in required_skills:
            matched_count = sum(
                1 for c in candidates if skill in c.get("matched_skills", [])
            )
            skill_coverage[skill] = (matched_count / len(candidates) * 100) if candidates else 0.0
        fig = plot_skills_coverage(skill_coverage)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No required skills defined in the JD.")

with tab_exp:
    experiences = [c.get("experience_years") for c in candidates if c.get("experience_years") is not None]
    fig = plot_experience_distribution(experiences)
    st.plotly_chart(fig, use_container_width=True)

with tab_gap:
    # Aggregate missing skills
    missing_agg: dict[str, int] = {}
    for c in candidates:
        for skill in c.get("missing_skills", []):
            missing_agg[skill] = missing_agg.get(skill, 0) + 1
    fig = plot_skills_gap(missing_agg)
    st.plotly_chart(fig, use_container_width=True)
