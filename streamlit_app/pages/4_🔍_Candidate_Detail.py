"""Page 4 — Individual candidate deep-dive analysis."""
from __future__ import annotations

import os
import sys
import re
from typing import Any

import streamlit as st

_PAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_PAGE_DIR)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from components.sidebar import render_settings_sidebar, render_jd_summary, render_quick_stats
from components.score_card import (
    render_score_gauge,
    render_match_badge,
    render_score_breakdown,
    render_skills_comparison,
)
from components.charts import plot_score_breakdown, plot_comparison
from components.utils import (
    get_score_color,
    get_score_label,
    format_experience,
    generate_pdf_report,
    export_to_json,
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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def highlight_resume_text(text: str, jd: dict | None) -> str:
    """Return HTML with color-highlighted entities in resume text."""
    if not text:
        return "<em>No text available.</em>"

    # Skills — green
    skills_to_highlight = jd.get("required_skills", []) if jd else []
    for skill in sorted(skills_to_highlight, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
        text = pattern.sub(
            f'<mark style="background:#2ecc7133;color:#2ecc71;border-radius:3px;padding:1px 4px;">{skill}</mark>',
            text,
        )

    # Dates (YYYY or Month YYYY) — orange
    date_pattern = re.compile(
        r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+\d{4}\b|\b(19|20)\d{2}\b',
        re.IGNORECASE,
    )
    text = date_pattern.sub(
        lambda m: f'<mark style="background:#f39c1233;color:#f39c12;border-radius:3px;padding:1px 4px;">{m.group()}</mark>',
        text,
    )

    # Education keywords — purple
    edu_keywords = [
        "PhD", "Ph.D", "Doctorate", "Master", "Bachelor", "MBA",
        "B.S.", "M.S.", "B.A.", "M.A.", "Associate",
    ]
    for kw in sorted(edu_keywords, key=len, reverse=True):
        pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
        text = pattern.sub(
            f'<mark style="background:#9b59b633;color:#9b59b6;border-radius:3px;padding:1px 4px;">{kw}</mark>',
            text,
        )

    return text.replace("\n", "<br>")


def generate_recommendation(candidate: dict, jd: dict | None) -> dict:
    """Generate recommendation text, strengths, concerns, and interview questions."""
    score = candidate.get("overall_score", 0)
    matched = candidate.get("matched_skills", [])
    missing = candidate.get("missing_skills", [])
    exp = candidate.get("experience_years")
    req_exp = jd.get("experience_years") if jd else None
    edu = candidate.get("education", "Unknown")

    # Recommendation text
    label = get_score_label(score)
    if score >= 75:
        rec_text = (
            f"**{candidate.get('name', 'This candidate')}** is a **Strong Match** for the {jd['title'] if jd else 'position'} "
            f"with an overall score of {score:.1f}%. They demonstrate strong alignment with the required skill set "
            f"and are recommended for an initial screening interview."
        )
    elif score >= 50:
        rec_text = (
            f"**{candidate.get('name', 'This candidate')}** is a **Moderate Match** for the {jd['title'] if jd else 'position'} "
            f"with an overall score of {score:.1f}%. They meet several key requirements but have gaps in some areas. "
            f"Consider for a phone screen to assess fit more closely."
        )
    else:
        rec_text = (
            f"**{candidate.get('name', 'This candidate')}** is a **Weak Match** for the {jd['title'] if jd else 'position'} "
            f"with an overall score of {score:.1f}%. Significant gaps exist in required skills or experience. "
            f"Consider only if the candidate pool is limited."
        )

    # Strengths
    strengths = []
    if matched:
        strengths.append(f"Demonstrated {len(matched)} of {len(matched) + len(missing)} required skills: {', '.join(matched[:5])}")
    if exp and req_exp and exp >= req_exp:
        strengths.append(f"{exp:.1f} years of experience meets/exceeds the {req_exp:.1f}+ years requirement")
    elif exp and not req_exp:
        strengths.append(f"{exp:.1f} years of professional experience")
    if candidate.get("education_match_pct", 0) >= 80:
        strengths.append(f"Education level ({edu}) meets position requirements")
    if candidate.get("semantic_similarity", 0) >= 70:
        strengths.append("Resume language closely aligns with the job description")
    if not strengths:
        strengths.append("Candidate profile reviewed")

    # Concerns
    concerns = []
    if missing:
        concerns.append(f"Missing {len(missing)} required skill(s): {', '.join(missing[:5])}")
    if exp and req_exp and exp < req_exp:
        concerns.append(f"Experience ({exp:.1f} yrs) below the required {req_exp:.1f}+ years")
    if exp is None:
        concerns.append("Experience level could not be determined from resume")
    if candidate.get("education_match_pct", 0) < 60:
        concerns.append(f"Education level ({edu}) may not meet position requirements")
    if not concerns:
        concerns.append("No significant concerns identified")

    # Interview questions
    questions = []
    for skill in matched[:3]:
        questions.append(f"Describe a project where you applied **{skill}** at scale.")
    for skill in missing[:2]:
        questions.append(f"Do you have any exposure to **{skill}**? How would you approach learning it?")
    if exp and req_exp and exp < req_exp:
        questions.append("Can you walk us through your most complex technical project to date?")
    if not questions:
        questions = ["Tell me about your most impactful technical project.", "How do you stay current with industry trends?"]

    return {
        "recommendation": rec_text,
        "strengths": strengths,
        "concerns": concerns,
        "interview_questions": questions,
    }


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("🔍 Candidate Detail")

candidates = st.session_state.get("candidates", [])
if not candidates:
    st.info("No candidates found. Complete steps 1–2 first.", icon="📋")
    st.stop()

# Candidate selector
candidate_options = {f"{c.get('name', 'Unknown')} — {c.get('overall_score', 0):.1f}%": c for c in candidates}
default_idx = 0

selected_candidate = st.session_state.get("selected_candidate")
if selected_candidate:
    for i, c in enumerate(candidates):
        if c["id"] == selected_candidate["id"]:
            default_idx = i
            break

selected_label = st.selectbox(
    "Select Candidate",
    options=list(candidate_options.keys()),
    index=default_idx,
)
candidate = candidate_options[selected_label]
jd = st.session_state.get("jd")

st.markdown("---")

# ---------------------------------------------------------------------------
# Header row: name + gauge + badge
# ---------------------------------------------------------------------------
header_col, gauge_col = st.columns([2, 1])

with header_col:
    score = candidate.get("overall_score", 0)
    color = get_score_color(score)

    st.markdown(
        f"""
        <div style="margin-bottom: 16px;">
            <div style="font-size: 28px; font-weight: 800; color: #fafafa;">
                👤 {candidate.get("name", "Unknown Candidate")}
            </div>
            <div style="font-size: 13px; color: #888; margin-top: 4px;">
                {candidate.get("filename", "")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_match_badge(score)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick action buttons
    act_cols = st.columns(3)
    if act_cols[0].button("✅ Shortlist", use_container_width=True, type="primary"):
        candidate["status"] = "Shortlisted"
        st.session_state.setdefault("candidate_statuses", {})[candidate["id"]] = "Shortlisted"
        st.success("✅ Shortlisted!")
    if act_cols[1].button("⏸ Review", use_container_width=True):
        candidate["status"] = "Review"
        st.session_state.setdefault("candidate_statuses", {})[candidate["id"]] = "Review"
        st.info("⏸ Marked for Review")
    if act_cols[2].button("❌ Reject", use_container_width=True):
        candidate["status"] = "Rejected"
        st.session_state.setdefault("candidate_statuses", {})[candidate["id"]] = "Rejected"
        st.warning("❌ Rejected")

with gauge_col:
    render_score_gauge(score, get_score_label(score))

# ---------------------------------------------------------------------------
# Score Breakdown
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📊 Score Breakdown")

breakdown_col, radar_col = st.columns([1, 1])

with breakdown_col:
    breakdown = {
        "skill_match": candidate.get("skill_match_pct", 0),
        "experience_match": candidate.get("experience_match_pct", 0),
        "education_match": candidate.get("education_match_pct", 0),
        "semantic_similarity": candidate.get("semantic_similarity", 0),
        "overall": candidate.get("overall_score", 0),
    }
    render_score_breakdown(breakdown)

    # Experience / education details
    exp = candidate.get("experience_years")
    req_exp = jd.get("experience_years") if jd else None
    st.markdown(
        f"""
        <div style="margin-top:12px;font-size:13px;color:#aaa;">
            <b>Experience:</b> <span style="color:#fafafa;">{format_experience(exp)}</span>
            {"vs <span style='color:#fafafa;'>" + format_experience(req_exp) + " required</span>" if req_exp else ""}
            &nbsp;&nbsp;
            <b>Education:</b> <span style="color:#fafafa;">{candidate.get("education", "Unknown")}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with radar_col:
    radar_data = {
        "Skill Match": candidate.get("skill_match_pct", 0),
        "Experience": candidate.get("experience_match_pct", 0),
        "Education": candidate.get("education_match_pct", 0),
        "Semantic": candidate.get("semantic_similarity", 0),
    }
    fig = plot_score_breakdown(radar_data)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Skills analysis
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🎯 Skills Analysis")

render_skills_comparison(
    matched=candidate.get("matched_skills", []),
    missing=candidate.get("missing_skills", []),
    additional=candidate.get("additional_skills", []),
)

# Per-required-skill confidence table
if jd and jd.get("required_skills"):
    st.markdown("#### Required Skills Detail")
    req_skills = jd.get("required_skills", [])
    matched_set = set(candidate.get("matched_skills", []))

    rows = []
    for skill in req_skills:
        found = skill in matched_set
        confidence = 95.0 if found else 0.0  # keyword match is binary; placeholder for fuzzy
        rows.append({
            "Skill": skill,
            "Found": "✅ Yes" if found else "❌ No",
            "Confidence": f"{confidence:.0f}%",
        })

    import pandas as pd
    df_skills = pd.DataFrame(rows)
    st.dataframe(df_skills, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Resume text viewer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 📄 Resume Text Viewer")

resume_text = candidate.get("resume_text", "")
jd_text = jd.get("raw_text", "") if jd else ""

view_tab, compare_tab = st.tabs(["📄 Resume", "↔️ Side-by-Side Comparison"])

with view_tab:
    if resume_text:
        highlighted_html = highlight_resume_text(resume_text, jd)
        # Show legend
        st.markdown(
            """
            <div style="font-size:12px;margin-bottom:8px;color:#aaa;">
                Legend:
                <mark style="background:#2ecc7133;color:#2ecc71;border-radius:3px;padding:1px 4px;">Skills</mark>&nbsp;
                <mark style="background:#9b59b633;color:#9b59b6;border-radius:3px;padding:1px 4px;">Education</mark>&nbsp;
                <mark style="background:#f39c1233;color:#f39c12;border-radius:3px;padding:1px 4px;">Dates</mark>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="
                background:#1e1e2e;
                border:1px solid #30363d;
                border-radius:8px;
                padding:16px;
                font-size:13px;
                line-height:1.8;
                max-height:500px;
                overflow-y:auto;
                color:#fafafa;
                font-family:monospace;
            ">
                {highlighted_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No resume text available.")

with compare_tab:
    r_col, j_col = st.columns(2)
    with r_col:
        st.markdown("**📄 Resume**")
        st.text_area("Resume Text", value=resume_text or "No text", height=400,
                     label_visibility="collapsed", disabled=True)
    with j_col:
        st.markdown("**📋 Job Description**")
        st.text_area("JD Text", value=jd_text or "No JD loaded", height=400,
                     label_visibility="collapsed", disabled=True)

# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 💡 Recommendation")

reco = generate_recommendation(candidate, jd)

st.markdown(reco["recommendation"])

r1, r2 = st.columns(2)

with r1:
    st.markdown("#### 💚 Strengths")
    for s in reco["strengths"]:
        st.markdown(f"- {s}")

with r2:
    st.markdown("#### ⚠️ Concerns")
    for c_item in reco["concerns"]:
        st.markdown(f"- {c_item}")

st.markdown("#### ❓ Suggested Interview Questions")
for i, q in enumerate(reco["interview_questions"], 1):
    st.markdown(f"{i}. {q}")

# ---------------------------------------------------------------------------
# Download & Compare
# ---------------------------------------------------------------------------
st.markdown("---")
dl_col, cmp_col = st.columns([1, 1])

with dl_col:
    pdf_bytes = generate_pdf_report([candidate], title=f"Candidate Report — {candidate.get('name', 'Unknown')}")
    st.download_button(
        "📥 Download PDF Report",
        data=pdf_bytes,
        file_name=f"report_{candidate.get('name', 'candidate').replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

with cmp_col:
    # Compare with another candidate
    other_candidates = [c for c in candidates if c["id"] != candidate["id"]]
    if other_candidates:
        other_options = {f"{c.get('name', 'Unknown')} ({c.get('overall_score', 0):.1f}%)": c
                        for c in other_candidates}
        compare_with_label = st.selectbox("Compare with:", ["— select —"] + list(other_options.keys()))

        if compare_with_label != "— select —":
            other = other_options[compare_with_label]
            st.markdown("---")
            st.markdown(f"#### 📊 Comparison: {candidate.get('name')} vs {other.get('name')}")

            comp_scores = {
                candidate.get("name", "Candidate A"): {
                    "Skill Match": candidate.get("skill_match_pct", 0),
                    "Experience": candidate.get("experience_match_pct", 0),
                    "Education": candidate.get("education_match_pct", 0),
                    "Semantic": candidate.get("semantic_similarity", 0),
                    "Overall": candidate.get("overall_score", 0),
                },
                other.get("name", "Candidate B"): {
                    "Skill Match": other.get("skill_match_pct", 0),
                    "Experience": other.get("experience_match_pct", 0),
                    "Education": other.get("education_match_pct", 0),
                    "Semantic": other.get("semantic_similarity", 0),
                    "Overall": other.get("overall_score", 0),
                },
            }
            fig = plot_comparison(comp_scores)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Upload more resumes to enable comparison.", icon="📋")
