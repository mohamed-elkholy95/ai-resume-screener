# Architecture — AI Resume Screener

## System Overview

The AI Resume Screener is a multi-stage NLP pipeline that evaluates
candidates against job descriptions using a blend of rule-based matching,
statistical similarity, and optional ML classification.

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Layer                          │
│  Resume (PDF/DOCX/TXT)  ←→  Job Description (text/file)    │
└───────────────┬─────────────────────────┬───────────────────┘
                │                         │
                ▼                         ▼
┌───────────────────────┐   ┌───────────────────────────────┐
│   ResumeParser        │   │   JobDescriptionParser        │
│   - Text extraction   │   │   - Experience extraction     │
│   - Unicode cleanup   │   │   - Education level parsing   │
│   - PII masking       │   │   - Section identification    │
└───────────┬───────────┘   └──────────────┬────────────────┘
            │                              │
            ▼                              │
┌───────────────────────┐                  │
│   ResumeNER           │                  │
│   - spaCy base NER    │                  │
│   - Pattern matching  │                  │
│   - Skill taxonomy    │                  │
│   - Entity dedup      │                  │
└───────────┬───────────┘                  │
            │                              │
            ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ResumeMatcher                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐│
│  │ Skill Match  │ │ Experience   │ │ Semantic Similarity  ││
│  │ (set-based)  │ │ Match        │ │ (SBERT / TF-IDF /   ││
│  │              │ │ (ratio)      │ │  Jaccard fallback)   ││
│  └──────┬───────┘ └──────┬───────┘ └──────────┬───────────┘│
│         │                │                     │            │
│         └────────────────┼─────────────────────┘            │
│                          ▼                                  │
│               Weighted Composite Score                      │
│         skill×0.4 + exp×0.3 + edu×0.2 + sem×0.1           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     ResumeScorer                            │
│  - Orchestrates Matcher + NER + Classifier                  │
│  - Configurable weight profiles (skills_heavy, etc.)        │
│  - Generates detailed reports with skill analysis           │
│  - Ranks candidates with tie-breaking                       │
│  - Exports to CSV / JSON / Excel                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │ FastAPI  │ │Streamlit │ │ CLI / Export  │
        │ REST API │ │Dashboard │ │ (CSV/JSON)   │
        └──────────┘ └──────────┘ └──────────────┘
```

## Component Deep Dive

### 1. Text Extraction (`data_collection.py`)

Handles multi-format input with graceful degradation:

| Format | Library   | Fallback         |
|--------|-----------|------------------|
| PDF    | pypdf     | ImportError msg  |
| DOCX   | python-docx | ImportError msg |
| TXT/MD | stdlib    | Always available |

**Text cleaning pipeline:**
1. Unicode NFKC normalisation
2. Control character removal
3. Bullet character normalisation (•→-)
4. PII masking (email → `[EMAIL]`, phone → `[PHONE]`)
5. Whitespace collapse

### 2. Named Entity Recognition (`ner_extractor.py`)

Dual-strategy extraction:

- **spaCy NER** for ORG, DATE, LANGUAGE entities
- **Regex patterns** for domain-specific entities:
  - Experience years (e.g. "5+ years of experience")
  - Education levels (PhD, Master's, Bachelor's)
  - Certifications (AWS Certified, PMP, CISSP)
  - Job titles (Senior ML Engineer, Staff SRE)
- **Skill taxonomy** matching with fuzzy string matching (difflib)

### 3. Skill Taxonomy (`skill_taxonomy.py`)

Structured skill database organised by category:
- Programming languages (25 skills)
- ML/AI (30 skills)
- Data engineering (20 skills)
- Cloud/DevOps (19 skills)
- Web development (19 skills)
- Soft skills (11 skills)

Features:
- Canonical alias resolution (e.g. "k8s" → "kubernetes")
- Fuzzy matching with configurable threshold (default 0.8)
- Category classification for any skill

### 4. Matching Engine (`matcher.py`)

Four scoring dimensions with configurable weights:

| Dimension          | Default Weight | Method                      |
|--------------------|----------------|-----------------------------|
| Skill Match        | 40%            | Set intersection (req×0.7 + pref×0.3) |
| Experience Match   | 30%            | Ratio with graceful degradation |
| Education Match    | 20%            | Ordinal comparison with partial credit |
| Semantic Similarity| 10%            | SBERT → TF-IDF → Jaccard fallback |

**Gap Analysis** identifies:
- Critical gaps (missing required skills)
- Transferable skills (adjacent technology clusters)
- Prioritised learning recommendations

### 5. Classification (`classifier.py`)

Three model backends:

| Model              | Speed  | Accuracy | Memory  |
|--------------------|--------|----------|---------|
| Logistic Regression| ⚡ Fast | Good     | Low     |
| Random Forest      | Fast   | Good     | Medium  |
| BERT Fine-tune     | Slow   | Best     | High    |

All support:
- Train/predict/explain cycle
- Model persistence (pickle for sklearn, HF for BERT)
- LIME explainability (optional)

### 6. Evaluation (`evaluation.py`)

Metrics for each sub-task:

- **NER**: per-entity precision, recall, F1 (exact string match)
- **Classification**: accuracy, F1 (macro/weighted), ROC-AUC
- **Ranking**: NDCG, MRR, Precision@K

## Data Flow

```
1. User uploads resume + JD
2. Text extracted and cleaned (data_collection)
3. NER extracts skills, education, experience (ner_extractor)
4. Skills canonicalised via taxonomy (skill_taxonomy)
5. Four match dimensions computed (matcher)
6. Weighted composite score calculated (scorer)
7. Optional: ML classifier predicts fit class (classifier)
8. Report generated with breakdown and recommendations
9. Results surfaced via API or dashboard
```

## Configuration

All scoring parameters are centralised in `src/config.py`:

- **Paths**: PROJECT_ROOT, DATA_DIR, MODEL_DIR, REPORTS_DIR
- **Weights**: MATCH_WEIGHTS (validated at import)
- **Profiles**: SCORING_PROFILES (default, skills_heavy, experience_heavy, balanced)
- **Thresholds**: MATCH_THRESHOLDS (strong ≥0.75, moderate ≥0.50, weak ≥0.30)
- **NER**: Entity types, spaCy model name
- **API**: Host, port, CORS, max upload size

## Testing Strategy

| Module          | Test File          | Coverage Focus                    |
|-----------------|--------------------|-----------------------------------|
| data_collection | test_data_collection | Parsing, cleaning, edge cases   |
| matcher         | test_matcher       | Skill matching, scoring, gap analysis |
| scorer          | test_scorer        | Reports, ranking, export         |
| evaluation      | test_evaluation    | NER/classification/ranking metrics |
| skill_taxonomy  | test_skill_taxonomy| Alias resolution, fuzzy matching |
| utils           | test_utils         | Text processing, formatting      |
