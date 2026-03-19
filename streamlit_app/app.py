"""AI Resume Screener — Main Streamlit Application Entry Point."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# AI Resume Screener\nPowered by NLP & machine learning.",
    },
)

# ---------------------------------------------------------------------------
# Global dark-theme CSS injection
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Base dark background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Cards / containers */
    [data-testid="stExpander"],
    [data-testid="stForm"],
    div.stMetric {
        background-color: #262730;
        border-radius: 10px;
        border: 1px solid #30363d;
    }

    /* Metric value */
    [data-testid="stMetricValue"] { color: #2ecc71 !important; }

    /* DataFrames */
    [data-testid="stDataFrame"] table {
        background-color: #1e1e2e;
        color: #fafafa;
    }
    [data-testid="stDataFrame"] tbody tr:hover {
        background-color: #2e2e3e !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2ecc71;
        color: #0e1117;
        border: none;
        font-weight: 700;
        border-radius: 8px;
        padding: 8px 20px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Progress bars */
    .stProgress > div > div { background-color: #2ecc71; }

    /* Input fields */
    .stTextInput input,
    .stTextArea textarea,
    .stSelectbox select {
        background-color: #1e1e2e !important;
        color: #fafafa !important;
        border-color: #30363d !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        background-color: #262730 !important;
        color: #fafafa !important;
        border-radius: 6px 6px 0 0;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2ecc71 !important;
        border-bottom: 2px solid #2ecc71 !important;
    }

    /* Divider */
    hr { border-color: #30363d; }

    /* Alert boxes */
    .stAlert { border-radius: 8px; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #1e1e2e;
        border: 2px dashed #30363d;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "jd": None,              # Current job description dict
        "resumes": [],           # List of resume dicts
        "candidates": [],        # Processed/scored candidate results
        "selected_candidate": None,
        "settings": {
            "classification_model": "TF-IDF + Cosine Similarity",
            "similarity_threshold": 50,
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()

# ---------------------------------------------------------------------------
# Sidebar: branding + settings + JD summary + stats
# ---------------------------------------------------------------------------
import sys
import os

# Ensure components are importable regardless of CWD
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from components.sidebar import render_settings_sidebar, render_jd_summary, render_quick_stats

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 16px 0 8px 0;">
            <div style="font-size: 36px;">🤖</div>
            <div style="font-size: 20px; font-weight: 800; color: #2ecc71; letter-spacing: 0.5px;">
                AI Resume Screener
            </div>
            <div style="font-size: 11px; color: #888; margin-top: 4px;">
                Powered by NLP & ML
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # JD summary
    render_jd_summary(st.session_state.get("jd"))

    # Settings
    settings = render_settings_sidebar()
    st.session_state["settings"] = settings

    # Quick stats
    candidates = st.session_state.get("candidates", [])
    resumes = st.session_state.get("resumes", [])
    render_quick_stats({
        "uploaded": len(resumes),
        "processed": sum(1 for c in candidates if c.get("status") not in ("Pending", None)),
        "ranked": len(candidates),
    })

    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0.0 · AI Resume Screener")

# ---------------------------------------------------------------------------
# Navigation hint on home page
# ---------------------------------------------------------------------------
st.markdown("## 🤖 AI Resume Screener")
st.markdown(
    """
    Welcome! Use the **sidebar navigation** to move through the workflow:

    | Step | Page | Description |
    |------|------|-------------|
    | 1 | 📋 Job Description | Paste or upload a job description and extract required skills |
    | 2 | 📄 Upload Resumes | Upload candidate resumes for processing |
    | 3 | 📊 Rankings | View ranked candidates with charts and export options |
    | 4 | 🔍 Candidate Detail | Deep-dive into individual candidate analysis |
    """
)

st.info("👈 Start by clicking **1 📋 Job Description** in the sidebar.", icon="🚀")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Resumes Uploaded", len(st.session_state.get("resumes", [])))
with col2:
    st.metric("Processed", len(st.session_state.get("candidates", [])))
with col3:
    strong = sum(
        1 for c in st.session_state.get("candidates", [])
        if c.get("overall_score", 0) >= 75
    )
    st.metric("Strong Matches", strong)
