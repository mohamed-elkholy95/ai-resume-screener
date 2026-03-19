<div align="center">

# 📋 AI Resume Screener

**Intelligent resume screening** with skill matching, scoring, NER extraction, and evaluation metrics

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-84%20passed-success?style=flat-square)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## Overview

An **AI-powered resume screening system** that automatically evaluates candidates by matching skills, extracting key entities via NER, computing relevance scores, and generating structured evaluation reports. Designed for HR workflows with configurable scoring weights.

## Architecture

```
Resume Upload → Text Extraction → NER (Skills, Education, Experience)
    → Skill Matching → Scoring Engine → Ranking → Decision (Proceed/Hold/Reject)
```

## Features

- 🎯 **Skill Matching Engine** — Weighted scoring across required, preferred, and bonus skills
- 🏷️ **NER Extraction** — Named Entity Recognition for skills, education, experience, and certifications
- 📊 **Multi-dimensional Scoring** — Configurable weights for skill match, experience level, and education
- 📋 **Decision Pipeline** — Automatic Proceed/Hold/Reject with configurable thresholds
- 📈 **5-Page Dashboard** — Interactive Streamlit UI for resume analysis
- 🔗 **REST API** — Full CRUD for resume screening workflows
- ✅ **84 Tests** — Most comprehensive test suite in the portfolio

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/ai-resume-screener.git
cd ai-resume-screener
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Scoring Formula

```
overall_score = (skill_score × 0.5) + (experience_score × 0.3) + (education_score × 0.2)
skill_score    = (required_score × 0.7) + (preferred_score × 0.3)
```

## Project Structure

```
├── src/
│   ├── api/main.py              # FastAPI endpoints
│   ├── config.py                # Scoring weights & thresholds
│   ├── matcher.py               # Skill matching engine
│   ├── scorer.py                # Multi-dimensional scoring
│   ├── extractor.py             # NER entity extraction
│   └── evaluation.py            # Decision pipeline
├── streamlit_app/pages/         # 5 dashboard pages
├── tests/                       # 84 tests
└── requirements.txt
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
