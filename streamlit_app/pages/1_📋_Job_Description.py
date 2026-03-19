"""Page 1 — Job Description input, parsing, and skill editor."""
from __future__ import annotations

import re
import sys
import os
import io

import streamlit as st

# Ensure components importable
_PAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_PAGE_DIR)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from components.sidebar import render_settings_sidebar, render_jd_summary, render_quick_stats

# ---------------------------------------------------------------------------
# Skill taxonomy
# ---------------------------------------------------------------------------
SKILL_TAXONOMY: dict[str, list[str]] = {
    "Programming": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
        "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "MATLAB",
    ],
    "ML/AI": [
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "TensorFlow", "PyTorch", "Keras", "scikit-learn", "Hugging Face",
        "Transformers", "LLM", "RAG", "Reinforcement Learning", "MLOps",
    ],
    "Data": [
        "SQL", "NoSQL", "Pandas", "NumPy", "Spark", "Hadoop", "Kafka",
        "Airflow", "DBT", "Tableau", "Power BI", "Excel", "Statistics",
        "Data Wrangling", "ETL", "Data Pipeline",
    ],
    "Cloud": [
        "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
        "CI/CD", "GitHub Actions", "Jenkins", "Serverless", "Lambda",
    ],
    "Web": [
        "React", "Vue", "Angular", "Node.js", "FastAPI", "Django", "Flask",
        "REST API", "GraphQL", "HTML", "CSS", "PostgreSQL", "MongoDB", "Redis",
    ],
    "Soft Skills": [
        "Communication", "Leadership", "Teamwork", "Problem Solving",
        "Project Management", "Agile", "Scrum", "Stakeholder Management",
        "Presentation", "Mentoring",
    ],
}

ALL_SKILLS: list[str] = [s for skills in SKILL_TAXONOMY.values() for s in skills]

EDUCATION_LEVELS = [
    "High School", "Associate's Degree", "Bachelor's Degree",
    "Master's Degree", "PhD / Doctorate", "Not Specified",
]

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_job_title(text: str) -> str:
    """Heuristic: look for 'Job Title:' section or use first non-empty line."""
    for line in text.splitlines():
        m = re.search(r"(?i)job\s*title\s*[:\-–]\s*(.+)", line)
        if m:
            return m.group(1).strip()
    # Fallback: first non-empty line, capped at 80 chars
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and len(stripped) < 120:
            return stripped[:80]
    return ""


def extract_experience(text: str) -> float | None:
    """Return minimum years of experience mentioned (e.g. '3+ years' → 3.0)."""
    patterns = [
        r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience",
        r"(\d+)\s*\-\s*\d+\s*years?\s+(?:of\s+)?experience",
        r"experience\s+of\s+(\d+)\s*\+?\s*years?",
        r"minimum\s+(?:of\s+)?(\d+)\s+years?",
        r"at\s+least\s+(\d+)\s+years?",
    ]
    candidates_found = []
    for pattern in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            try:
                candidates_found.append(float(m.group(1)))
            except ValueError:
                pass
    return min(candidates_found) if candidates_found else None


def extract_education(text: str) -> str:
    """Detect education level from text."""
    lower = text.lower()
    if any(k in lower for k in ["phd", "ph.d", "doctorate", "doctoral"]):
        return "PhD / Doctorate"
    if any(k in lower for k in ["master", "m.s.", "m.sc", "mba", "m.eng"]):
        return "Master's Degree"
    if any(k in lower for k in ["bachelor", "b.s.", "b.sc", "b.a.", "undergraduate"]):
        return "Bachelor's Degree"
    if any(k in lower for k in ["associate"]):
        return "Associate's Degree"
    return "Not Specified"


def extract_skills_from_text(text: str) -> list[str]:
    """Return skills from taxonomy that appear in the text (case-insensitive)."""
    found = []
    lower_text = text.lower()
    for skill in ALL_SKILLS:
        # Use word boundary matching
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, lower_text):
            found.append(skill)
    return found


def read_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    except ImportError:
        return "[PyPDF2 not installed — install it to parse PDF files]"
    except Exception as e:
        return f"[PDF parsing error: {e}]"


def read_docx(file_bytes: bytes) -> str:
    """Extract text from DOCX bytes."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        return "[python-docx not installed — install it to parse DOCX files]"
    except Exception as e:
        return f"[DOCX parsing error: {e}]"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 12px 0 4px 0;">
            <div style="font-size:28px;">🤖</div>
            <div style="font-size:17px;font-weight:800;color:#2ecc71;">AI Resume Screener</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_jd_summary(st.session_state.get("jd"))
    render_settings_sidebar()
    candidates = st.session_state.get("candidates", [])
    resumes = st.session_state.get("resumes", [])
    render_quick_stats({
        "uploaded": len(resumes),
        "processed": sum(1 for c in candidates if c.get("status") not in ("Pending", None)),
        "ranked": len(candidates),
    })

# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.title("📋 Job Description")
st.markdown("Paste or upload a job description to begin screening.")

# ── Input section ──
input_tab, upload_tab = st.tabs(["✏️ Paste Text", "📁 Upload File"])

jd_text: str = ""

with input_tab:
    pasted = st.text_area(
        "Paste Job Description",
        value=st.session_state.get("_jd_raw_text", ""),
        height=350,
        placeholder=(
            "Paste the full job description here...\n\n"
            "e.g.:\nJob Title: Senior Machine Learning Engineer\n\n"
            "We are looking for a Senior ML Engineer with 5+ years of experience...\n\n"
            "Requirements:\n- Python, TensorFlow, PyTorch\n- Bachelor's or Master's degree\n..."
        ),
        label_visibility="collapsed",
    )
    if pasted.strip():
        jd_text = pasted

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload JD file",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
        label_visibility="collapsed",
    )
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
        if ext == "pdf":
            jd_text = read_pdf(file_bytes)
        elif ext == "docx":
            jd_text = read_docx(file_bytes)
        else:
            jd_text = file_bytes.decode("utf-8", errors="replace")

        st.success(f"✅ Loaded: **{uploaded_file.name}** ({len(jd_text):,} chars)")

# ── Auto-extract ──
if jd_text.strip():
    st.markdown("---")
    col_preview, col_info = st.columns([3, 2])

    with col_preview:
        st.markdown("### 📄 JD Preview")
        with st.expander("Show full text", expanded=True):
            st.text_area("JD Text", value=jd_text, height=300, label_visibility="collapsed", disabled=True)

    with col_info:
        st.markdown("### 🔍 Auto-Extracted Info")

        detected_title = extract_job_title(jd_text)
        detected_exp = extract_experience(jd_text)
        detected_edu = extract_education(jd_text)
        detected_skills = extract_skills_from_text(jd_text)

        title_input = st.text_input("Job Title", value=detected_title)
        exp_input = st.number_input(
            "Experience Required (years)",
            min_value=0.0, max_value=30.0,
            value=float(detected_exp) if detected_exp else 0.0,
            step=0.5,
        )
        edu_input = st.selectbox(
            "Education Level",
            options=EDUCATION_LEVELS,
            index=EDUCATION_LEVELS.index(detected_edu) if detected_edu in EDUCATION_LEVELS else 0,
        )

    # ── Skills editor ──
    st.markdown("---")
    st.markdown("### 🎯 Required Skills Editor")
    st.markdown("Auto-detected skills are shown below. Add or remove as needed.")

    # Persist skill state
    if "jd_skills" not in st.session_state or st.session_state.get("_jd_raw_text") != jd_text:
        st.session_state["jd_skills"] = detected_skills.copy()
        st.session_state["_jd_raw_text"] = jd_text

    # Grouped display by category
    for category, skills_in_cat in SKILL_TAXONOMY.items():
        detected_in_cat = [s for s in skills_in_cat if s in st.session_state["jd_skills"]]
        if not detected_in_cat:
            continue
        with st.expander(f"**{category}** ({len(detected_in_cat)} detected)", expanded=True):
            to_remove = []
            cols = st.columns(4)
            for i, skill in enumerate(detected_in_cat):
                col = cols[i % 4]
                keep = col.checkbox(skill, value=True, key=f"skill_check_{category}_{skill}")
                if not keep:
                    to_remove.append(skill)
            for s in to_remove:
                if s in st.session_state["jd_skills"]:
                    st.session_state["jd_skills"].remove(s)

    # Manual skill add
    st.markdown("**Add a skill manually:**")
    add_col, btn_col = st.columns([4, 1])
    with add_col:
        new_skill = st.text_input("Skill name", key="new_skill_input", label_visibility="collapsed",
                                   placeholder="e.g. Airflow, FastAPI, Spark...")
    with btn_col:
        if st.button("➕ Add", use_container_width=True):
            if new_skill.strip() and new_skill.strip() not in st.session_state["jd_skills"]:
                st.session_state["jd_skills"].append(new_skill.strip())
                st.rerun()

    # Show current skills as chips
    if st.session_state["jd_skills"]:
        chips_html = "".join(
            f'<span style="display:inline-block;margin:3px 4px;padding:4px 12px;'
            f'border-radius:14px;font-size:12px;font-weight:600;'
            f'background-color:#2ecc7133;border:1px solid #2ecc71;color:#2ecc71;">'
            f'{s}</span>'
            for s in st.session_state["jd_skills"]
        )
        st.markdown(
            f'<div style="margin:8px 0;line-height:2.2;">{chips_html}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"{len(st.session_state['jd_skills'])} skill(s) selected")
    else:
        st.warning("⚠️ No required skills detected or selected. Add at least one skill.")

    # ── Save & Analyze ──
    st.markdown("---")
    if st.button("💾 Save & Analyze →", type="primary", use_container_width=True):
        if not title_input.strip():
            st.error("❌ Job Title is required.")
        elif not st.session_state["jd_skills"]:
            st.error("❌ At least one required skill must be selected.")
        else:
            st.session_state["jd"] = {
                "title": title_input.strip(),
                "raw_text": jd_text,
                "required_skills": st.session_state["jd_skills"],
                "experience_years": exp_input if exp_input > 0 else None,
                "education_level": edu_input,
            }
            st.success("✅ Job description saved! Navigate to **2 📄 Upload Resumes** to continue.")
            st.balloons()

else:
    st.info("👆 Paste a job description or upload a file to get started.", icon="💡")

# Show saved JD notice
if st.session_state.get("jd"):
    saved_jd = st.session_state["jd"]
    st.markdown("---")
    st.success(
        f"✅ **Active JD:** {saved_jd['title']} | "
        f"{len(saved_jd.get('required_skills', []))} skills | "
        f"Experience: {saved_jd.get('experience_years', 'N/A')} yrs | "
        f"Education: {saved_jd.get('education_level', 'N/A')}"
    )
