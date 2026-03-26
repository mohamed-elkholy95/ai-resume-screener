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

The composite score combines four dimensions with configurable weights:

```
composite = (skill × 0.40) + (experience × 0.30) + (education × 0.20) + (semantic × 0.10)
skill_score = (required_match × 0.70) + (preferred_match × 0.30)
```

### Scoring Profiles

| Profile            | Skills | Experience | Education | Semantic |
|--------------------|--------|------------|-----------|----------|
| **default**        | 40%    | 30%        | 20%       | 10%      |
| **skills_heavy**   | 55%    | 20%        | 15%       | 10%      |
| **experience_heavy** | 30%  | 45%        | 15%       | 10%      |
| **balanced**       | 30%    | 30%        | 20%       | 20%      |

Select a profile at scorer initialisation or pass custom weights.

## Fairness & Bias Auditing

Automated screening systems carry inherent bias risks. This project includes:

- **Adverse Impact Ratio (AIR)** — implements the EEOC four-fifths rule to detect disparate impact across demographic groups
- **Score Parity Analysis** — flags mean-score disparities exceeding 10 percentage points between groups
- **PII Masking** — emails and phone numbers are masked during text cleaning to reduce identity-based bias

> ⚠️ These metrics surface statistical patterns — they do not prove discrimination. Always combine with human review.

## Project Structure

```
├── src/
│   ├── api/
│   │   ├── main.py              # FastAPI application
│   │   └── schemas.py           # Pydantic request/response models
│   ├── config.py                # Scoring weights, profiles & thresholds
│   ├── matcher.py               # Skill matching + gap analysis
│   ├── scorer.py                # Multi-dimensional scoring & ranking
│   ├── ner_extractor.py         # NER entity extraction (spaCy + patterns)
│   ├── classifier.py            # ML classifiers (LR / RF / BERT)
│   ├── evaluation.py            # NER, classification, ranking & fairness metrics
│   ├── skill_taxonomy.py        # Canonical skill database & fuzzy matching
│   ├── data_collection.py       # Resume/JD parsing (PDF, DOCX, TXT)
│   └── utils.py                 # Text processing & scoring helpers
├── streamlit_app/pages/         # Interactive dashboard
├── docs/ARCHITECTURE.md         # System design documentation
├── tests/                       # Comprehensive test suite
└── requirements.txt
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
