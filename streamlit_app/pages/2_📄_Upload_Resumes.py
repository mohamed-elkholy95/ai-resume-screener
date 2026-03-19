"""Page 2 — Upload and process resumes."""
from __future__ import annotations

import io
import re
import os
import sys
import time
import random
import hashlib
from typing import Any

import streamlit as st

_PAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_PAGE_DIR)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from components.sidebar import render_settings_sidebar, render_jd_summary, render_quick_stats
from components.score_card import render_match_badge
from components.utils import (
    format_file_size, format_score, get_score_color, export_to_csv, export_to_json
)

# Skill taxonomy mirror (subset for entity extraction)
ALL_SKILL_KEYWORDS = [
    "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
    "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
    "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Hugging Face",
    "Transformers", "LLM", "RAG", "Reinforcement Learning", "MLOps",
    "SQL", "NoSQL", "Pandas", "NumPy", "Spark", "Hadoop", "Kafka",
    "Airflow", "DBT", "Tableau", "Power BI", "Excel", "Statistics",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
    "CI/CD", "GitHub Actions", "Jenkins", "Serverless", "Lambda",
    "React", "Vue", "Angular", "Node.js", "FastAPI", "Django", "Flask",
    "REST API", "GraphQL", "HTML", "CSS", "PostgreSQL", "MongoDB", "Redis",
    "Communication", "Leadership", "Teamwork", "Problem Solving",
    "Project Management", "Agile", "Scrum",
]

EDUCATION_KEYWORDS = {
    "PhD / Doctorate": ["phd", "ph.d", "doctorate", "doctoral"],
    "Master's Degree": ["master", "m.s.", "m.sc", "mba", "m.eng"],
    "Bachelor's Degree": ["bachelor", "b.s.", "b.sc", "b.a.", "undergraduate"],
    "Associate's Degree": ["associate"],
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def read_pdf(file_bytes: bytes) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        return ""
    except Exception:
        return ""


def read_docx(file_bytes: bytes) -> str:
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        return ""
    except Exception:
        return ""


def extract_text(file_bytes: bytes, filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return read_pdf(file_bytes)
    if ext == "docx":
        return read_docx(file_bytes)
    return file_bytes.decode("utf-8", errors="replace")


def extract_skills_from_text(text: str) -> list[str]:
    lower = text.lower()
    found = []
    for skill in ALL_SKILL_KEYWORDS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, lower):
            found.append(skill)
    return found


def extract_experience_years(text: str) -> float | None:
    patterns = [
        r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s+of\s+(\d+)\s*\+?\s*years?",
        r"(\d{4})\s*[-–]\s*(?:present|current|now)",
    ]
    found = []
    for p in patterns[:2]:
        for m in re.finditer(p, text, re.IGNORECASE):
            try:
                found.append(float(m.group(1)))
            except ValueError:
                pass
    # Year-range pattern: count year spans
    for m in re.finditer(r"(\d{4})\s*[-–]\s*(present|\d{4})", text, re.IGNORECASE):
        start = int(m.group(1))
        end_raw = m.group(2).lower()
        end = 2024 if end_raw == "present" else int(end_raw)
        if 1980 <= start <= 2024 and end >= start:
            found.append(float(end - start))
    if not found:
        return None
    return max(found)


def extract_education(text: str) -> str:
    lower = text.lower()
    for level, keywords in EDUCATION_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return level
    return "Not Specified"


def compute_match_score(resume_text: str, jd: dict, settings: dict) -> dict[str, Any]:
    """Compute match score between resume text and job description.

    Uses TF-IDF cosine similarity when sklearn is available; falls back
    to keyword overlap scoring.
    """
    required_skills: list[str] = jd.get("required_skills", [])
    resume_skills = extract_skills_from_text(resume_text)

    # Skill match
    matched_skills = [s for s in required_skills if s in resume_skills]
    missing_skills = [s for s in required_skills if s not in resume_skills]
    additional_skills = [s for s in resume_skills if s not in required_skills]
    skill_match_pct = (len(matched_skills) / len(required_skills) * 100) if required_skills else 0.0

    # Experience match
    resume_exp = extract_experience_years(resume_text)
    req_exp = jd.get("experience_years")
    if req_exp and resume_exp is not None:
        exp_ratio = min(resume_exp / req_exp, 1.5) / 1.5
        experience_match_pct = exp_ratio * 100
    elif resume_exp is not None:
        experience_match_pct = min(resume_exp / 5.0, 1.0) * 100
    else:
        experience_match_pct = 50.0  # unknown

    # Education match
    resume_edu = extract_education(resume_text)
    edu_levels = ["Not Specified", "High School", "Associate's Degree",
                  "Bachelor's Degree", "Master's Degree", "PhD / Doctorate"]
    req_edu = jd.get("education_level", "Not Specified")
    try:
        res_idx = edu_levels.index(resume_edu)
        req_idx = edu_levels.index(req_edu)
        education_match_pct = 100.0 if res_idx >= req_idx else max(0, 100 - (req_idx - res_idx) * 20)
    except ValueError:
        education_match_pct = 50.0

    # Semantic similarity (TF-IDF when available, else keyword overlap)
    jd_text = jd.get("raw_text", " ".join(required_skills))
    semantic_pct = _compute_semantic(resume_text, jd_text)

    # Weighted composite
    weights = {"skill": 0.45, "experience": 0.25, "education": 0.10, "semantic": 0.20}
    overall = (
        weights["skill"] * skill_match_pct
        + weights["experience"] * experience_match_pct
        + weights["education"] * education_match_pct
        + weights["semantic"] * semantic_pct
    )

    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "additional_skills": additional_skills,
        "skill_match_pct": round(skill_match_pct, 1),
        "experience_years": resume_exp,
        "experience_match_pct": round(experience_match_pct, 1),
        "education": resume_edu,
        "education_match_pct": round(education_match_pct, 1),
        "semantic_similarity": round(semantic_pct, 1),
        "overall_score": round(overall, 1),
        "score_breakdown": {
            "Skill Match": round(skill_match_pct, 1),
            "Experience": round(experience_match_pct, 1),
            "Education": round(education_match_pct, 1),
            "Semantic Similarity": round(semantic_pct, 1),
        },
    }


def _compute_semantic(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity, fallback to word-overlap Jaccard."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vect = TfidfVectorizer(stop_words="english", max_features=5000)
        mat = vect.fit_transform([text_a, text_b])
        score = cosine_similarity(mat[0:1], mat[1:2])[0][0]
        return float(score * 100)
    except Exception:
        # Jaccard over word tokens
        set_a = set(re.findall(r'\b\w{3,}\b', text_a.lower()))
        set_b = set(re.findall(r'\b\w{3,}\b', text_b.lower()))
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b) * 100


def _file_id(filename: str, size: int) -> str:
    return hashlib.md5(f"{filename}:{size}".encode()).hexdigest()[:8]


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
    candidates = st.session_state.get("candidates", [])
    resumes = st.session_state.get("resumes", [])
    render_quick_stats({
        "uploaded": len(resumes),
        "processed": sum(1 for c in candidates if c.get("status") not in ("Pending", None)),
        "ranked": len(candidates),
    })

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("📄 Upload Resumes")

jd = st.session_state.get("jd")
if not jd:
    st.warning("⚠️ Please complete **Step 1: Job Description** before uploading resumes.", icon="📋")
    st.stop()

st.markdown(f"Screening against: **{jd['title']}** | {len(jd.get('required_skills', []))} required skills")

# ── File uploader ──
uploaded_files = st.file_uploader(
    "Drag and drop resumes here",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
    help="Upload one or more resumes in PDF, DOCX, or TXT format.",
    label_visibility="collapsed",
)

if uploaded_files:
    # Deduplicate against already-tracked resumes
    existing_ids = {r["id"] for r in st.session_state.get("resumes", [])}
    new_resumes = []
    for f in uploaded_files:
        fid = _file_id(f.name, f.size)
        if fid not in existing_ids:
            new_resumes.append({
                "id": fid,
                "name": f.name,
                "size": f.size,
                "bytes": f.read(),
                "status": "Pending",
                "overall_score": None,
            })
    if new_resumes:
        st.session_state.setdefault("resumes", []).extend(new_resumes)

resumes: list[dict] = st.session_state.get("resumes", [])

# ── Action bar ──
if resumes:
    col_analyze, col_clear, col_export_csv, col_export_json = st.columns([2, 1, 1, 1])

    with col_analyze:
        analyze_btn = st.button("🚀 Analyze All", type="primary", use_container_width=True)
    with col_clear:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state["resumes"] = []
            st.session_state["candidates"] = []
            st.rerun()
    with col_export_csv:
        candidates_for_export = st.session_state.get("candidates", [])
        if candidates_for_export:
            csv_bytes = export_to_csv(candidates_for_export)
            st.download_button(
                "📥 CSV",
                data=csv_bytes,
                file_name="resume_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("📥 CSV", disabled=True, use_container_width=True)
    with col_export_json:
        if candidates_for_export:
            json_bytes = export_to_json(candidates_for_export)
            st.download_button(
                "📥 JSON",
                data=json_bytes,
                file_name="resume_results.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.button("📥 JSON", disabled=True, use_container_width=True)

    # ── File list table ──
    st.markdown("---")
    st.markdown("### 📋 Uploaded Resumes")

    header_cols = st.columns([3, 1, 1, 2, 2])
    for col, label in zip(header_cols, ["Filename", "Size", "Status", "Match Score", "Actions"]):
        col.markdown(f"**{label}**")

    st.markdown("<hr style='margin:4px 0;border-color:#30363d;'>", unsafe_allow_html=True)

    candidates_map: dict[str, dict] = {
        c["id"]: c for c in st.session_state.get("candidates", [])
    }

    for resume in resumes:
        rid = resume["id"]
        candidate = candidates_map.get(rid)
        score = candidate["overall_score"] if candidate else None
        status = candidate["status"] if candidate else resume.get("status", "Pending")

        row_cols = st.columns([3, 1, 1, 2, 2])
        row_cols[0].markdown(f"📄 `{resume['name']}`")
        row_cols[1].markdown(format_file_size(resume["size"]))

        # Status badge
        status_colors = {"Pending": "#888", "Processing": "#f39c12", "Scored": "#2ecc71"}
        sc = status_colors.get(status, "#888")
        row_cols[2].markdown(
            f'<span style="color:{sc};font-weight:700;">{status}</span>',
            unsafe_allow_html=True,
        )

        # Score
        if score is not None:
            color = get_score_color(score)
            row_cols[3].markdown(
                f'<span style="color:{color};font-weight:700;font-size:16px;">{score:.1f}%</span>',
                unsafe_allow_html=True,
            )
        else:
            row_cols[3].markdown("—")

        # Actions
        if row_cols[4].button("❌ Remove", key=f"remove_{rid}", use_container_width=True):
            st.session_state["resumes"] = [r for r in resumes if r["id"] != rid]
            st.session_state["candidates"] = [c for c in st.session_state.get("candidates", []) if c["id"] != rid]
            st.rerun()

    # ── Analysis processing ──
    if analyze_btn:
        st.markdown("---")
        st.markdown("### ⚙️ Processing Queue")

        pending = [r for r in resumes if r["id"] not in candidates_map]
        if not pending:
            st.info("All resumes have already been analyzed.")
        else:
            progress_bar = st.progress(0)
            status_placeholder = st.empty()
            results_container = st.container()
            new_candidates = []

            for i, resume in enumerate(pending):
                pct = i / len(pending)
                progress_bar.progress(pct)

                # Step-by-step status
                for step in ["Parsing resume...", "Extracting entities...", "Computing match score..."]:
                    status_placeholder.info(f"**{resume['name']}** — {step}")
                    time.sleep(0.15)

                # Extract text
                text = extract_text(resume["bytes"], resume["name"])

                # Compute scores
                scores = compute_match_score(text, jd, settings)

                candidate = {
                    "id": resume["id"],
                    "name": resume["name"].rsplit(".", 1)[0],
                    "filename": resume["name"],
                    "resume_text": text,
                    "status": "Pending",
                    **scores,
                }

                # Apply threshold
                threshold = settings.get("similarity_threshold", 50)
                if candidate["overall_score"] >= 75:
                    candidate["status"] = "Strong Match"
                elif candidate["overall_score"] >= threshold:
                    candidate["status"] = "Review"
                else:
                    candidate["status"] = "Weak Match"

                new_candidates.append(candidate)

                # Per-resume result card
                with results_container:
                    with st.expander(
                        f"✅ {resume['name']} — {candidate['overall_score']:.1f}%",
                        expanded=candidate["overall_score"] >= threshold,
                    ):
                        c1, c2, c3 = st.columns(3)
                        color = get_score_color(candidate["overall_score"])
                        c1.metric("Overall Score", f"{candidate['overall_score']:.1f}%")
                        c2.metric("Skill Match", f"{candidate['skill_match_pct']:.1f}%")
                        exp = candidate.get("experience_years")
                        c3.metric("Experience", f"{exp:.1f} yrs" if exp else "Unknown")

                        matched = candidate.get("matched_skills", [])
                        missing = candidate.get("missing_skills", [])
                        if matched:
                            st.markdown(
                                "**Matched:** "
                                + " ".join(
                                    f'<span style="background:#2ecc7133;border:1px solid #2ecc71;'
                                    f'color:#2ecc71;border-radius:10px;padding:2px 8px;margin:2px;'
                                    f'font-size:11px;">{s}</span>'
                                    for s in matched
                                ),
                                unsafe_allow_html=True,
                            )
                        if missing:
                            st.markdown(
                                "**Missing:** "
                                + " ".join(
                                    f'<span style="background:#e74c3c33;border:1px solid #e74c3c;'
                                    f'color:#e74c3c;border-radius:10px;padding:2px 8px;margin:2px;'
                                    f'font-size:11px;">{s}</span>'
                                    for s in missing
                                ),
                                unsafe_allow_html=True,
                            )

            progress_bar.progress(1.0)
            status_placeholder.success(f"✅ Processed {len(pending)} resume(s)!")

            # Merge into session state
            existing_candidates = [
                c for c in st.session_state.get("candidates", [])
                if c["id"] not in {nc["id"] for nc in new_candidates}
            ]
            st.session_state["candidates"] = existing_candidates + new_candidates

    # ── Per-resume expanded results ──
    if st.session_state.get("candidates") and not analyze_btn:
        st.markdown("---")
        st.markdown("### 📊 Results Summary")
        for candidate in st.session_state["candidates"]:
            score = candidate.get("overall_score", 0)
            color = get_score_color(score)
            with st.expander(f"**{candidate['name']}** — {score:.1f}%", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Overall Score", f"{score:.1f}%")
                c2.metric("Skill Match", f"{candidate.get('skill_match_pct', 0):.1f}%")
                exp = candidate.get("experience_years")
                c3.metric("Experience", f"{exp:.1f} yrs" if exp else "Unknown")

                matched = candidate.get("matched_skills", [])
                missing = candidate.get("missing_skills", [])
                if matched:
                    st.markdown(
                        "**Matched:** "
                        + " ".join(
                            f'<span style="background:#2ecc7133;border:1px solid #2ecc71;'
                            f'color:#2ecc71;border-radius:10px;padding:2px 8px;margin:2px;'
                            f'font-size:11px;">{s}</span>'
                            for s in matched
                        ),
                        unsafe_allow_html=True,
                    )
                if missing:
                    st.markdown(
                        "**Missing:** "
                        + " ".join(
                            f'<span style="background:#e74c3c33;border:1px solid #e74c3c;'
                            f'color:#e74c3c;border-radius:10px;padding:2px 8px;margin:2px;'
                            f'font-size:11px;">{s}</span>'
                            for s in missing
                        ),
                        unsafe_allow_html=True,
                    )

else:
    st.info("👆 Upload resume files above to get started.", icon="📁")
