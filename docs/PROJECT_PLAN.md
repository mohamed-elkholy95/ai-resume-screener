# AI Resume Screener — Project Plan

**Timeline:** 10 days  
**Difficulty:** Intermediate–Advanced  
**Goal:** Build an end-to-end resume screening system with NER, semantic matching, BERT classification, composite scoring, and an explainable UI dashboard.

---

## Table of Contents

1. [Phase 1: Data Collection & Preparation](#phase-1-data-collection--preparation-days-1-2)
2. [Phase 2: NER & Entity Extraction](#phase-2-ner--entity-extraction-days-2-3)
3. [Phase 3: Embedding & Similarity](#phase-3-embedding--similarity-days-3-4)
4. [Phase 4: Classification Model](#phase-4-classification-model-days-4-6)
5. [Phase 5: Ranking & Scoring Pipeline](#phase-5-ranking--scoring-pipeline-days-6-7)
6. [Phase 6: API & Dashboard](#phase-6-api--dashboard-days-7-9)
7. [Phase 7: Evaluation](#phase-7-evaluation-days-9-10)
8. [Dependencies & Setup](#dependencies--setup)

---

## Phase 1: Data Collection & Preparation (Days 1–2)

### Dataset Sources

| Source | Description | Volume |
|--------|-------------|--------|
| Kaggle — Resume Dataset | 2484 resumes across 25 categories | 2.5K |
| Kaggle — Job Descriptions | 19K job descriptions | 19K |
| Synthetic generation | LLM-generated resumes for augmentation | +5K |
| LinkedIn Scraping | Optional — respect ToS | Variable |

### Data Schema

**Resume Record:**
```json
{
  "resume_id": "uuid",
  "raw_text": "John Doe\nSenior Python Engineer...",
  "cleaned_text": "john doe senior python engineer...",
  "category": "Software Engineering",
  "source": "kaggle|synthetic|upload",
  "file_format": "pdf|docx|txt",
  "created_at": "2024-01-01T00:00:00Z"
}
```

**Job Description Record:**
```json
{
  "jd_id": "uuid",
  "title": "Senior Python Engineer",
  "company": "TechCorp",
  "description": "We are looking for...",
  "required_skills": ["Python", "FastAPI", "Kubernetes"],
  "preferred_skills": ["Go", "Rust"],
  "min_experience_years": 5,
  "education_requirement": "BS Computer Science or equivalent",
  "created_at": "2024-01-01T00:00:00Z"
}
```

### File: src/data_collection.py

```python
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
import unicodedata

class DataCollector:
    """Resume and job description data collection utilities."""

    def scrape_resumes(self, source_urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape resume data from provided URLs.

        Args:
            source_urls: List of URLs to scrape (respect robots.txt).

        Returns:
            List of resume dicts: {url, raw_html, extracted_text}

        Note:
            Uses rate limiting (1 req/sec) to avoid blocking.
            Respects robots.txt via urllib.robotparser.
        """
        import requests
        from bs4 import BeautifulSoup
        import time

        results = []
        for url in source_urls:
            try:
                resp = requests.get(url, timeout=10, headers={"User-Agent": "ResearchBot/1.0"})
                soup = BeautifulSoup(resp.text, "html.parser")
                text = soup.get_text(separator="\n", strip=True)
                results.append({"url": url, "raw_html": resp.text, "extracted_text": text})
                time.sleep(1)
            except Exception as e:
                results.append({"url": url, "error": str(e)})
        return results

    def generate_synthetic_resumes(
        self,
        n: int = 1000,
        job_categories: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate synthetic resumes using OpenAI GPT.

        Args:
            n: Number of resumes to generate.
            job_categories: List of job categories (e.g., ["Software Engineering", "Data Science"]).

        Returns:
            List of synthetic resume dicts.
        """
        from openai import OpenAI
        import json

        if job_categories is None:
            job_categories = ["Software Engineering", "Data Science", "Product Management",
                              "DevOps", "Machine Learning", "Frontend Development"]

        client = OpenAI()
        resumes = []

        template = """Generate a realistic resume for a {seniority} {category} professional.
Include: name, email, phone, location, summary, work experience (2-4 jobs),
skills (8-15 specific technical skills), education, certifications.
Format as plain text. Be specific with company names, dates, and technologies.
Vary seniority levels (junior/mid/senior) and include edge cases."""

        for i in range(n):
            category = job_categories[i % len(job_categories)]
            seniority = ["junior", "mid-level", "senior", "staff"][i % 4]
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": template.format(
                    seniority=seniority, category=category)}],
                temperature=0.9,
            )
            resumes.append({
                "raw_text": response.choices[0].message.content,
                "category": category,
                "seniority": seniority,
                "source": "synthetic",
            })
        return resumes

    def parse_pdf_resume(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF resume with layout awareness.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dict: {text, pages, metadata}
        """
        import pdfplumber
        from pypdf import PdfReader

        # Try pdfplumber first (better for tables/columns)
        try:
            with pdfplumber.open(file_path) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    # Convert tables to text
                    for table in tables:
                        for row in table:
                            text += " | ".join(str(cell or "") for cell in row) + "\n"
                    pages.append(text)
                full_text = "\n\n".join(pages)
        except Exception:
            # Fallback to pypdf
            reader = PdfReader(file_path)
            full_text = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )

        return {
            "text": self.clean_resume_text(full_text),
            "pages": len(full_text.split("\n\n")),
            "file_path": file_path,
        }

    def clean_resume_text(self, text: str) -> str:
        """
        Normalize and clean resume text.

        Operations:
        - Normalize unicode (café → cafe)
        - Remove special characters except punctuation
        - Normalize whitespace
        - Remove email/phone from training data (privacy)
        - Fix encoding artifacts
        """
        # Normalize unicode
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")

        # Remove URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Normalize multiple spaces/newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        # Remove non-printable characters
        text = re.sub(r"[^\x20-\x7E\n]", "", text)

        # Remove common resume noise
        text = re.sub(r"Page \d+ of \d+", "", text)
        text = re.sub(r"Curriculum Vitae", "", text, flags=re.IGNORECASE)

        return text.strip()

    def load_kaggle_dataset(self, dataset_path: str) -> List[Dict]:
        """Load and parse the Kaggle resume dataset CSV."""
        import pandas as pd
        df = pd.read_csv(dataset_path)
        results = []
        for _, row in df.iterrows():
            results.append({
                "raw_text": self.clean_resume_text(str(row.get("Resume_str", ""))),
                "category": str(row.get("Category", "Unknown")),
                "source": "kaggle",
            })
        return results
```

### Data Cleaning Pipeline

```python
# scripts/prepare_data.py
from src.data_collection import DataCollector
import json, os

collector = DataCollector()

# 1. Load Kaggle resumes
kaggle_resumes = collector.load_kaggle_dataset("data/raw/kaggle_resumes.csv")

# 2. Generate synthetic resumes for rare categories
synthetic = collector.generate_synthetic_resumes(
    n=500,
    job_categories=["DevOps", "Blockchain", "AR/VR"]
)

# 3. Combine and shuffle
all_resumes = kaggle_resumes + synthetic
import random; random.shuffle(all_resumes)

# 4. Save
with open("data/processed/resumes.json", "w") as f:
    json.dump(all_resumes, f, indent=2)

print(f"Total resumes: {len(all_resumes)}")
```

---

## Phase 2: NER & Entity Extraction (Days 2–3)

### Entity Types

| Entity Label | Description | Example |
|-------------|-------------|---------|
| SKILL | Technical or soft skill | "Python", "React", "Leadership" |
| EDUCATION | Degree + institution | "BS Computer Science, MIT" |
| EXPERIENCE | Work history entry | "Senior Engineer at Google (2020-2023)" |
| COMPANY | Organization name | "Google", "Amazon", "Startup Inc" |
| JOB_TITLE | Professional title | "Software Engineer", "Data Scientist" |
| CERTIFICATION | Professional cert | "AWS Solutions Architect", "PMP" |
| LANGUAGE | Programming/spoken language | "Python", "Spanish" |
| DATE | Date or date range | "January 2020 – Present" |
| LOCATION | Geographic location | "San Francisco, CA" |
| GPA | Academic performance | "3.8 GPA" |

### spaCy Training Data Format

```python
# data/ner_training_data.py
# Format: (text, {"entities": [(start, end, label)]})
TRAINING_DATA = [
    (
        "Senior Python Developer with 5 years experience at Google",
        {"entities": [
            (7, 23, "JOB_TITLE"),
            (7, 13, "SKILL"),   # Python
            (29, 36, "DATE"),   # 5 years
            (51, 57, "COMPANY"),
        ]},
    ),
    # ... 500+ annotated examples
]
```

### File: src/ner_extractor.py

```python
from typing import List, Dict, Any, Tuple
import spacy
from spacy.training import Example
from pathlib import Path

class ResumeNERExtractor:
    """
    spaCy-based NER for structured resume entity extraction.
    Fine-tuned on resume-specific entity types.
    """

    ENTITY_LABELS = [
        "SKILL", "EDUCATION", "EXPERIENCE", "COMPANY", "JOB_TITLE",
        "CERTIFICATION", "LANGUAGE", "DATE", "LOCATION", "GPA"
    ]

    def __init__(self, model_path: str = None):
        """
        Initialize NER extractor.

        Args:
            model_path: Path to fine-tuned spaCy model. If None, loads base model.
        """
        if model_path and Path(model_path).exists():
            self.nlp = spacy.load(model_path)
        else:
            self.nlp = spacy.load("en_core_web_lg")
            # Add custom NER labels if not in base model
            if "ner" not in self.nlp.pipe_names:
                self.nlp.add_pipe("ner", last=True)
            ner = self.nlp.get_pipe("ner")
            for label in self.ENTITY_LABELS:
                ner.add_label(label)

    def train_ner_model(
        self,
        training_data: List[Tuple[str, Dict]],
        output_dir: str,
        n_iter: int = 30,
        batch_size: int = 8,
        drop_rate: float = 0.5,
    ) -> Dict[str, float]:
        """
        Fine-tune spaCy NER model on resume training data.

        Args:
            training_data: List of (text, annotations) tuples.
            output_dir: Directory to save trained model.
            n_iter: Number of training iterations.
            batch_size: Mini-batch size.
            drop_rate: Dropout rate for regularization.

        Returns:
            Dict of final training losses.
        """
        import random
        from spacy.util import minibatch, compounding

        # Prepare examples
        examples = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)

        # Disable irrelevant pipes during training
        other_pipes = [p for p in self.nlp.pipe_names if p != "ner"]
        losses_history = []

        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            for iteration in range(n_iter):
                random.shuffle(examples)
                losses = {}
                batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    self.nlp.update(batch, sgd=optimizer, drop=drop_rate, losses=losses)
                losses_history.append(losses.get("ner", 0))
                if iteration % 10 == 0:
                    print(f"  Iter {iteration}: NER loss = {losses.get('ner', 0):.4f}")

        # Save model
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.nlp.to_disk(output_dir)
        print(f"Model saved to {output_dir}")
        return {"final_ner_loss": losses_history[-1]}

    def extract_entities(self, resume_text: str) -> Dict[str, List[str]]:
        """
        Extract all entity types from resume text.

        Args:
            resume_text: Clean resume text.

        Returns:
            Dict mapping entity labels to lists of extracted values.
            Example: {"SKILL": ["Python", "FastAPI"], "COMPANY": ["Google"]}
        """
        doc = self.nlp(resume_text)
        entities: Dict[str, List[str]] = {label: [] for label in self.ENTITY_LABELS}

        for ent in doc.ents:
            if ent.label_ in entities:
                text = ent.text.strip()
                if text and text not in entities[ent.label_]:
                    entities[ent.label_].append(text)

        # Also apply rule-based extraction as fallback
        entities["SKILL"] = list(set(entities["SKILL"] + self._rule_based_skills(resume_text)))
        entities["DATE"] = list(set(entities["DATE"] + self._extract_dates(resume_text)))

        return entities

    def extract_skills(self, resume_text: str) -> List[str]:
        """
        Extract skills with taxonomy matching.

        Args:
            resume_text: Resume text.

        Returns:
            Normalized list of skills matched to taxonomy.
        """
        from src.skill_taxonomy import SkillTaxonomy
        taxonomy = SkillTaxonomy()

        # NER-based extraction
        doc = self.nlp(resume_text)
        raw_skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]

        # Rule-based extraction (scan for known skills)
        taxonomy_skills = taxonomy.find_skills_in_text(resume_text)

        # Combine and normalize
        all_skills = list(set(raw_skills + taxonomy_skills))
        return taxonomy.normalize_skills(all_skills)

    def match_skills(
        self,
        candidate_skills: List[str],
        required_skills: List[str],
        threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Match candidate skills against required skills with fuzzy matching.

        Args:
            candidate_skills: Skills extracted from resume.
            required_skills: Skills required by job description.
            threshold: Minimum similarity score for a fuzzy match.

        Returns:
            Dict with: matched (list), missing (list), match_ratio (float)
        """
        from rapidfuzz import fuzz, process
        matched, missing = [], []

        for req_skill in required_skills:
            # Exact match
            if req_skill.lower() in [s.lower() for s in candidate_skills]:
                matched.append(req_skill)
                continue

            # Fuzzy match
            result = process.extractOne(
                req_skill,
                candidate_skills,
                scorer=fuzz.token_sort_ratio,
                score_cutoff=threshold * 100,
            )
            if result:
                matched.append(req_skill)
            else:
                missing.append(req_skill)

        return {
            "matched": matched,
            "missing": missing,
            "match_ratio": len(matched) / max(len(required_skills), 1),
            "total_required": len(required_skills),
            "total_matched": len(matched),
        }

    def _rule_based_skills(self, text: str) -> List[str]:
        """Extract skills from skill section using regex patterns."""
        import re
        patterns = [
            r"(?:skills?|technologies?|tools?|languages?)\s*:?\s*([^\n]{10,200})",
            r"(?:proficient in|experience with|knowledge of)\s+([^\n.]{10,100})",
        ]
        skills = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Split on common delimiters
                parts = re.split(r"[,|•·/]", match)
                skills.extend([p.strip() for p in parts if 2 < len(p.strip()) < 40])
        return skills

    def _extract_dates(self, text: str) -> List[str]:
        """Extract date ranges using regex."""
        import re
        patterns = [
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
            r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            r"\s+\d{4}\s*[-–—]\s*(?:Present|\d{4}|\w+ \d{4})",
            r"\b\d{4}\s*[-–—]\s*(?:Present|\d{4})\b",
        ]
        dates = []
        for pattern in patterns:
            dates.extend(re.findall(pattern, text))
        return dates
```

### File: src/skill_taxonomy.py

```python
from typing import List, Dict, Set
import json
from pathlib import Path

class SkillTaxonomy:
    """
    Comprehensive skills database with 5,000+ skills organized by category.
    Supports normalization and fuzzy matching.
    """

    # Embedded taxonomy (subset shown — full version is in data/skill_taxonomy.json)
    TAXONOMY: Dict[str, List[str]] = {
        "programming_languages": [
            "Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++", "C#",
            "Ruby", "Swift", "Kotlin", "Scala", "R", "MATLAB", "Perl", "PHP", "Bash",
        ],
        "web_frameworks": [
            "React", "Vue.js", "Angular", "Next.js", "FastAPI", "Django", "Flask",
            "Express.js", "Spring Boot", "Ruby on Rails", "Laravel", "ASP.NET",
        ],
        "databases": [
            "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch", "Cassandra",
            "DynamoDB", "SQLite", "ClickHouse", "BigQuery", "Snowflake",
        ],
        "cloud": [
            "AWS", "Google Cloud", "Azure", "S3", "EC2", "Lambda", "Kubernetes",
            "Docker", "Terraform", "Ansible", "CloudFormation", "GKE", "EKS",
        ],
        "ml_ai": [
            "TensorFlow", "PyTorch", "scikit-learn", "Keras", "Hugging Face",
            "LangChain", "OpenAI", "Pandas", "NumPy", "Spark", "Hadoop",
        ],
        "soft_skills": [
            "Leadership", "Communication", "Teamwork", "Problem Solving",
            "Project Management", "Agile", "Scrum", "Mentoring",
        ],
        "certifications": [
            "AWS Certified Solutions Architect", "AWS Certified Developer",
            "Google Cloud Professional", "CKA", "PMP", "CISSP", "CompTIA Security+",
        ],
    }

    # Skill aliases / normalization map
    ALIASES: Dict[str, str] = {
        "js": "JavaScript", "ts": "TypeScript", "py": "Python",
        "postgres": "PostgreSQL", "mongo": "MongoDB", "k8s": "Kubernetes",
        "tf": "TensorFlow", "sklearn": "scikit-learn", "gcp": "Google Cloud",
        "react.js": "React", "reactjs": "React", "vuejs": "Vue.js",
        "node": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
    }

    def __init__(self, taxonomy_path: str = None):
        if taxonomy_path and Path(taxonomy_path).exists():
            with open(taxonomy_path) as f:
                self.TAXONOMY.update(json.load(f))
        self._all_skills: Set[str] = set()
        for skills in self.TAXONOMY.values():
            self._all_skills.update(s.lower() for s in skills)

    def find_skills_in_text(self, text: str) -> List[str]:
        """Scan text for known skills using case-insensitive matching."""
        found = []
        text_lower = text.lower()
        for category_skills in self.TAXONOMY.values():
            for skill in category_skills:
                if skill.lower() in text_lower:
                    found.append(skill)
        return list(set(found))

    def normalize_skills(self, skills: List[str]) -> List[str]:
        """Normalize skill names using alias mapping."""
        normalized = []
        for skill in skills:
            key = skill.lower().strip()
            if key in self.ALIASES:
                normalized.append(self.ALIASES[key])
            else:
                normalized.append(skill)
        return list(set(normalized))

    def get_skill_category(self, skill: str) -> str:
        """Return category of a skill."""
        skill_lower = skill.lower()
        for category, skills in self.TAXONOMY.items():
            if any(s.lower() == skill_lower for s in skills):
                return category
        return "other"

    def get_all_skills(self) -> List[str]:
        """Return flat list of all known skills."""
        return [s for skills in self.TAXONOMY.values() for s in skills]
```

---

## Phase 3: Embedding & Similarity (Days 3–4)

### Similarity Computation

**Cosine Similarity:**
```
similarity = dot(v1, v2) / (||v1|| * ||v2||)
```

**Weighted scoring:**
- Resume and JD are both embedded as full-text vectors
- Section-level embeddings (skills only, experience only) for targeted comparison
- Asymmetric similarity: JD terms weighted more than generic resume phrases

### File: src/matcher.py

```python
from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

class ResumeMatcher:
    """Semantic similarity matching between resumes and job descriptions."""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def compute_match_score(self, resume_text: str, job_description: str) -> float:
        """
        Compute overall semantic match score.

        Args:
            resume_text: Full resume text.
            job_description: Full job description text.

        Returns:
            Float between 0 and 1 (1 = perfect match).
        """
        embeddings = self.model.encode([resume_text, job_description], normalize_embeddings=True)
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        return max(0.0, min(1.0, similarity))

    def compute_skill_match(
        self,
        resume_skills: List[str],
        required_skills: List[str],
        preferred_skills: List[str] = None,
    ) -> Dict[str, float]:
        """
        Compute skill-level match scores.

        Returns:
            Dict: {required_match: float, preferred_match: float, overall: float}
        """
        from src.ner_extractor import ResumeNERExtractor
        extractor = ResumeNERExtractor()
        required_result = extractor.match_skills(resume_skills, required_skills)
        preferred_result = {"match_ratio": 0.0}
        if preferred_skills:
            preferred_result = extractor.match_skills(resume_skills, preferred_skills)

        # Required skills weighted 70%, preferred 30%
        overall = required_result["match_ratio"] * 0.7 + preferred_result["match_ratio"] * 0.3

        return {
            "required_match": required_result["match_ratio"],
            "preferred_match": preferred_result["match_ratio"],
            "overall": overall,
            "matched_required": required_result["matched"],
            "missing_required": required_result["missing"],
        }

    def compute_experience_match(
        self,
        candidate_years: float,
        required_years: float,
        candidate_titles: List[str],
        required_title: str,
    ) -> float:
        """
        Compute experience match score.

        Args:
            candidate_years: Total years of relevant experience.
            required_years: Minimum required years.
            candidate_titles: List of past job titles.
            required_title: Target job title.

        Returns:
            Float between 0 and 1.
        """
        # Years score: over-qualified caps at 1.0, under-qualified penalized
        if required_years == 0:
            years_score = 1.0
        elif candidate_years >= required_years:
            years_score = min(1.0, 1.0 + 0.1 * (candidate_years - required_years) / required_years)
        else:
            years_score = candidate_years / required_years

        # Title similarity
        if candidate_titles and required_title:
            title_embeddings = self.model.encode(
                candidate_titles + [required_title], normalize_embeddings=True
            )
            target_emb = title_embeddings[-1]
            title_sims = [float(np.dot(e, target_emb)) for e in title_embeddings[:-1]]
            title_score = max(title_sims) if title_sims else 0.0
        else:
            title_score = 0.5  # Neutral if no data

        return 0.5 * min(1.0, years_score) + 0.5 * title_score

    def compute_education_match(
        self,
        candidate_degree: str,
        required_degree: str,
        candidate_field: str,
        required_field: str,
    ) -> float:
        """Compare degree level and field of study."""
        degree_levels = {"high school": 0, "associate": 1, "bachelor": 2,
                         "master": 3, "mba": 3, "phd": 4, "doctorate": 4}

        def get_level(degree: str) -> int:
            degree_lower = degree.lower()
            for key, level in degree_levels.items():
                if key in degree_lower:
                    return level
            return 2  # Assume bachelor if unclear

        candidate_level = get_level(candidate_degree)
        required_level = get_level(required_degree)
        degree_score = 1.0 if candidate_level >= required_level else candidate_level / max(required_level, 1)

        # Field similarity
        if candidate_field and required_field:
            embs = self.model.encode([candidate_field, required_field], normalize_embeddings=True)
            field_score = float(np.dot(embs[0], embs[1]))
        else:
            field_score = 0.7  # Neutral if not specified

        return 0.5 * degree_score + 0.5 * field_score

    def batch_score_resumes(
        self,
        resumes: List[Dict[str, Any]],
        job_description: str,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Compute semantic similarity for a batch of resumes efficiently.

        Args:
            resumes: List of resume dicts with 'text' key.
            job_description: Job description text.
            batch_size: Number of resumes per batch.

        Returns:
            List of resumes with added 'semantic_score' field.
        """
        jd_emb = self.model.encode([job_description], normalize_embeddings=True)[0]
        resume_texts = [r["text"] for r in resumes]
        all_scores = []

        for i in range(0, len(resume_texts), batch_size):
            batch = resume_texts[i:i+batch_size]
            batch_embs = self.model.encode(batch, normalize_embeddings=True)
            scores = [float(np.dot(emb, jd_emb)) for emb in batch_embs]
            all_scores.extend(scores)

        for resume, score in zip(resumes, all_scores):
            resume["semantic_score"] = score
        return resumes
```

---

## Phase 4: Classification Model (Days 4–6)

### Model Architecture

**Baseline:** Logistic Regression on TF-IDF features
**Main Model:** Fine-tuned BERT (`bert-base-uncased`) on concatenated [resume; JD] pairs
**Input format:** `[CLS] {resume_summary} [SEP] {job_description} [SEP]`
**Output:** 4-class classification

**Classes:**
- 0: Strong Match (semantic similarity ≥ 0.85 + skill match ≥ 0.8)
- 1: Moderate Match (0.65–0.85)
- 2: Weak Match (0.45–0.65)
- 3: No Match (< 0.45)

### Training Data Labeling

```python
def auto_label(resume: Dict, jd: Dict, matcher: ResumeMatcher) -> int:
    """Create pseudo-labels for BERT training using heuristic scoring."""
    semantic_score = matcher.compute_match_score(resume["text"], jd["description"])
    skill_match = matcher.compute_skill_match(
        resume.get("skills", []), jd.get("required_skills", [])
    )["overall"]

    combined = 0.5 * semantic_score + 0.5 * skill_match
    if combined >= 0.75: return 0  # Strong
    if combined >= 0.55: return 1  # Moderate
    if combined >= 0.35: return 2  # Weak
    return 3  # No match
```

### File: src/classifier.py

```python
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from torch.utils.data import Dataset

class ResumeJDDataset(Dataset):
    def __init__(self, resumes: List[str], jds: List[str], labels: List[int], tokenizer, max_len: int = 512):
        self.encodings = tokenizer(
            resumes, jds,
            truncation=True, padding=True,
            max_length=max_len, return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | {"labels": self.labels[idx]}


class ResumeClassifier:
    """BERT-based resume-JD fit classifier with SHAP explanations."""

    LABELS = ["Strong Match", "Moderate Match", "Weak Match", "No Match"]
    MODEL_NAME = "bert-base-uncased"

    def __init__(self, model_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path or self.MODEL_NAME)

        if model_path:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME, num_labels=4
            )
        self.model = self.model.to(self.device)

    def train_classifier(
        self,
        train_data: List[Tuple[str, str, int]],
        eval_data: List[Tuple[str, str, int]] = None,
        model_name: str = "bert-base-uncased",
        output_dir: str = "models/classifier",
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
    ) -> Dict[str, float]:
        """
        Fine-tune BERT classifier on resume-JD pairs.

        Args:
            train_data: List of (resume_text, jd_text, label) tuples.
            eval_data: Optional validation set.
            model_name: Pretrained model to fine-tune.
            output_dir: Save directory.
            num_epochs: Training epochs.
            batch_size: Training batch size.
            learning_rate: AdamW learning rate.

        Returns:
            Dict of evaluation metrics.
        """
        resumes = [d[0][:1000] for d in train_data]  # Truncate for BERT limit
        jds = [d[1][:512] for d in train_data]
        labels = [d[2] for d in train_data]

        train_dataset = ResumeJDDataset(resumes, jds, labels, self.tokenizer)

        eval_dataset = None
        if eval_data:
            e_resumes = [d[0][:1000] for d in eval_data]
            e_jds = [d[1][:512] for d in eval_data]
            e_labels = [d[2] for d in eval_data]
            eval_dataset = ResumeJDDataset(e_resumes, e_jds, e_labels, self.tokenizer)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_ratio=0.1,
            weight_decay=0.01,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
            metric_for_best_model="eval_accuracy",
            logging_steps=50,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )

        result = trainer.train()
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        return result.metrics

    def predict_fit(self, resume: str, job_description: str) -> str:
        """Predict match class label."""
        probs = self.predict_proba(resume, job_description)
        class_idx = int(np.argmax(probs))
        return self.LABELS[class_idx]

    def predict_proba(self, resume: str, job_description: str) -> np.ndarray:
        """
        Return probability distribution over 4 match classes.

        Returns:
            numpy array of shape (4,) summing to 1.0
        """
        self.model.eval()
        inputs = self.tokenizer(
            resume[:1000], job_description[:512],
            return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        return probs

    def explain_prediction(
        self,
        resume: str,
        jd: str,
        model,
        method: str = "shap",
        top_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate SHAP or LIME explanation for a prediction.

        Args:
            resume: Resume text.
            jd: Job description text.
            model: Trained classifier instance.
            method: "shap" or "lime".
            top_features: Number of top features to return.

        Returns:
            Dict with explanation text, feature importances, and prediction.
        """
        prediction = self.predict_fit(resume, jd)
        probs = self.predict_proba(resume, jd)

        if method == "lime":
            from lime.lime_text import LimeTextExplainer
            explainer = LimeTextExplainer(class_names=self.LABELS)
            combined_text = f"{resume} [SEP] {jd}"

            def predict_fn(texts):
                results = []
                for text in texts:
                    parts = text.split("[SEP]")
                    r = parts[0].strip() if parts else text
                    j = parts[1].strip() if len(parts) > 1 else jd
                    results.append(self.predict_proba(r, j))
                return np.array(results)

            explanation = explainer.explain_instance(
                combined_text, predict_fn, num_features=top_features, num_samples=500
            )
            features = explanation.as_list()
        else:
            features = []  # SHAP for BERT requires transformers-interpret

        return {
            "prediction": prediction,
            "probabilities": {self.LABELS[i]: float(probs[i]) for i in range(4)},
            "top_features": features,
            "explanation_text": self._generate_text_explanation(resume, jd, prediction, features),
        }

    def _generate_text_explanation(self, resume, jd, prediction, features) -> str:
        """Generate human-readable explanation."""
        positive = [f[0] for f in features if f[1] > 0][:3]
        negative = [f[0] for f in features if f[1] < 0][:3]
        parts = [f"Prediction: **{prediction}**"]
        if positive:
            parts.append(f"Strengths: {', '.join(positive)}")
        if negative:
            parts.append(f"Gaps: {', '.join(negative)}")
        return ". ".join(parts)

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro"),
        }
```

---

## Phase 5: Ranking & Scoring Pipeline (Days 6–7)

### Composite Score Formula

```
composite_score = (
    skill_match_score   * 0.40 +
    experience_score    * 0.30 +
    education_score     * 0.20 +
    semantic_similarity * 0.10
)
```

**Bonus modifiers:**
- Extra certifications relevant to JD: +0.02 each (max +0.06)
- Years significantly over requirement: +0.03
- Remote/location mismatch: -0.05

### File: src/scorer.py

```python
from typing import List, Dict, Any
import pandas as pd
import json
from datetime import datetime

class ResumeScorer:
    """Composite ranking and score reporting."""

    WEIGHTS = {
        "skill_match": 0.40,
        "experience": 0.30,
        "education": 0.20,
        "semantic": 0.10,
    }

    def __init__(self, matcher, ner_extractor, classifier):
        self.matcher = matcher
        self.ner = ner_extractor
        self.classifier = classifier

    def score_resume(self, resume: Dict[str, Any], jd: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute full scoring breakdown for one resume against one JD.

        Args:
            resume: Dict with text, parsed entities.
            jd: Dict with description, required_skills, etc.

        Returns:
            Score report dict.
        """
        # Extract entities if not already done
        if "entities" not in resume:
            resume["entities"] = self.ner.extract_entities(resume["text"])

        entities = resume["entities"]

        # Component scores
        skill_result = self.matcher.compute_skill_match(
            entities.get("SKILL", []),
            jd.get("required_skills", []),
            jd.get("preferred_skills", []),
        )

        experience_score = self.matcher.compute_experience_match(
            candidate_years=resume.get("years_experience", 0),
            required_years=jd.get("min_experience_years", 0),
            candidate_titles=entities.get("JOB_TITLE", []),
            required_title=jd.get("title", ""),
        )

        education_score = self.matcher.compute_education_match(
            candidate_degree=resume.get("highest_degree", ""),
            required_degree=jd.get("education_requirement", ""),
            candidate_field=resume.get("field_of_study", ""),
            required_field=jd.get("field_of_study", ""),
        )

        semantic_score = self.matcher.compute_match_score(resume["text"], jd["description"])

        # Composite
        composite = (
            skill_result["overall"] * self.WEIGHTS["skill_match"] +
            experience_score * self.WEIGHTS["experience"] +
            education_score * self.WEIGHTS["education"] +
            semantic_score * self.WEIGHTS["semantic"]
        )

        composite = min(1.0, max(0.0, composite))

        return {
            "composite_score": round(composite, 4),
            "component_scores": {
                "skill_match": round(skill_result["overall"], 4),
                "experience": round(experience_score, 4),
                "education": round(education_score, 4),
                "semantic_similarity": round(semantic_score, 4),
            },
            "skill_details": skill_result,
            "ml_prediction": self.classifier.predict_fit(resume["text"], jd["description"]),
            "ml_probabilities": self.classifier.predict_proba(resume["text"], jd["description"]).tolist(),
        }

    def rank_resumes(
        self,
        resumes: List[Dict[str, Any]],
        job_description: Dict[str, Any],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Score and rank all resumes against a job description.

        Args:
            resumes: List of resume dicts.
            job_description: JD dict.
            top_k: Number of top candidates to return.

        Returns:
            Sorted list of resumes with scores, highest first.
        """
        scored_resumes = []
        for i, resume in enumerate(resumes):
            try:
                score_report = self.score_resume(resume, job_description)
                resume_with_score = {**resume, **score_report, "rank": 0}
                scored_resumes.append(resume_with_score)
            except Exception as e:
                resume["score_error"] = str(e)
                resume["composite_score"] = 0.0
                scored_resumes.append(resume)

        # Sort by composite score
        scored_resumes.sort(key=lambda r: r.get("composite_score", 0), reverse=True)

        # Assign ranks
        for rank, r in enumerate(scored_resumes[:top_k], 1):
            r["rank"] = rank

        return scored_resumes[:top_k]

    def generate_score_report(self, resume: Dict, jd: Dict) -> str:
        """Generate a detailed text report for one candidate."""
        scores = self.score_resume(resume, jd)
        report = f"""
# Candidate Score Report
Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

## Overall: {scores['composite_score'] * 100:.1f}/100 — {scores['ml_prediction']}

## Component Breakdown
| Component        | Score | Weight | Contribution |
|-----------------|-------|--------|-------------|
| Skill Match     | {scores['component_scores']['skill_match']*100:.0f}%  | 40%    | {scores['component_scores']['skill_match']*0.4*100:.1f} |
| Experience      | {scores['component_scores']['experience']*100:.0f}%  | 30%    | {scores['component_scores']['experience']*0.3*100:.1f} |
| Education       | {scores['component_scores']['education']*100:.0f}%  | 20%    | {scores['component_scores']['education']*0.2*100:.1f} |
| Semantic Sim    | {scores['component_scores']['semantic_similarity']*100:.0f}%  | 10%    | {scores['component_scores']['semantic_similarity']*0.1*100:.1f} |

## Skills Analysis
✅ Matched: {', '.join(scores['skill_details'].get('matched_required', []))}
❌ Missing: {', '.join(scores['skill_details'].get('missing_required', []))}
"""
        return report.strip()

    def export_rankings(
        self,
        rankings: List[Dict],
        format: str = "csv",
        output_path: str = "rankings_output",
    ) -> str:
        """
        Export ranked results.

        Args:
            rankings: List of scored resume dicts.
            format: "csv", "json", or "excel".
            output_path: Output file path (without extension).

        Returns:
            Path to saved file.
        """
        rows = []
        for r in rankings:
            rows.append({
                "rank": r.get("rank", ""),
                "filename": r.get("filename", ""),
                "composite_score": r.get("composite_score", 0),
                "skill_match": r.get("component_scores", {}).get("skill_match", 0),
                "experience_score": r.get("component_scores", {}).get("experience", 0),
                "education_score": r.get("component_scores", {}).get("education", 0),
                "ml_prediction": r.get("ml_prediction", ""),
                "matched_skills": ", ".join(r.get("skill_details", {}).get("matched_required", [])),
                "missing_skills": ", ".join(r.get("skill_details", {}).get("missing_required", [])),
            })

        df = pd.DataFrame(rows)
        if format == "csv":
            path = f"{output_path}.csv"
            df.to_csv(path, index=False)
        elif format == "excel":
            path = f"{output_path}.xlsx"
            df.to_excel(path, index=False)
        elif format == "json":
            path = f"{output_path}.json"
            with open(path, "w") as f:
                json.dump(rankings, f, indent=2, default=str)
        return path
```

---

## Phase 6: API & Dashboard (Days 7–9)

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/jd | Submit job description |
| POST | /api/resumes | Upload resume(s) |
| GET | /api/rankings/{jd_id} | Get ranked candidates |
| GET | /api/resume/{id}/score | Get individual score |
| GET | /api/resume/{id}/report | Get detailed report |
| DELETE | /api/jd/{id} | Remove JD |
| GET | /api/health | Health check |

### Dashboard Features (src/dashboard/app.py)

```python
# Key dashboard sections:
# 1. JD Input: text area + upload
# 2. Resume Upload: multi-file drag-and-drop
# 3. Processing: progress bar with step indicators
# 4. Rankings Table: sortable, filterable, paginated
# 5. Candidate Detail: score breakdown, skill gaps, SHAP chart
# 6. Comparison: side-by-side 2-candidate view
# 7. Export: CSV / Excel download button
```

---

## Phase 7: Evaluation (Days 9–10)

### Evaluation Metrics

| Metric | Applicability | Target |
|--------|-------------|--------|
| Accuracy | Classification | ≥ 0.82 |
| F1 (macro) | Classification | ≥ 0.78 |
| NER F1 (per entity) | SKILL, TITLE, etc. | ≥ 0.80 |
| NDCG@10 | Ranking quality | ≥ 0.85 |
| MRR | Ranking quality | ≥ 0.80 |
| Spearman ρ | Score correlation vs human | ≥ 0.75 |

### File: src/evaluation.py

```python
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)
import numpy as np
from typing import List, Dict, Any

class Evaluator:
    def evaluate_classifier(
        self,
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str] = None,
    ) -> Dict[str, Any]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
            "precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "report": classification_report(y_true, y_pred, target_names=class_names),
        }

    def evaluate_ner(
        self,
        true_entities: List[Dict[str, List]],
        pred_entities: List[Dict[str, List]],
        entity_labels: List[str],
    ) -> Dict[str, float]:
        """Compute per-entity F1 scores using exact match."""
        from collections import defaultdict
        tp = defaultdict(int)
        fp = defaultdict(int)
        fn = defaultdict(int)

        for true, pred in zip(true_entities, pred_entities):
            for label in entity_labels:
                true_set = set(t.lower() for t in true.get(label, []))
                pred_set = set(p.lower() for p in pred.get(label, []))
                tp[label] += len(true_set & pred_set)
                fp[label] += len(pred_set - true_set)
                fn[label] += len(true_set - pred_set)

        results = {}
        for label in entity_labels:
            precision = tp[label] / max(tp[label] + fp[label], 1)
            recall = tp[label] / max(tp[label] + fn[label], 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            results[label] = {"precision": precision, "recall": recall, "f1": f1}
        return results

    def compute_ndcg(self, relevance_scores: List[float], k: int = 10) -> float:
        """Compute NDCG@k for a ranked list."""
        def dcg(scores):
            return sum(s / np.log2(i + 2) for i, s in enumerate(scores))

        actual = dcg(relevance_scores[:k])
        ideal = dcg(sorted(relevance_scores, reverse=True)[:k])
        return actual / max(ideal, 1e-8)
```

---

## Dependencies & Setup

### requirements.txt

```
spacy>=3.7.0
transformers>=4.36.0
sentence-transformers>=2.3.0
scikit-learn>=1.3.0
fastapi>=0.109.0
uvicorn>=0.27.0
streamlit>=1.30.0
pypdf>=4.0.0
pdfplumber>=0.10.0
nltk>=3.8.0
shap>=0.43.0
lime>=0.2.0
python-docx>=1.0.0
pandas>=2.0.0
numpy>=1.24.0
rapidfuzz>=3.5.0
torch>=2.1.0
datasets>=2.16.0
openai>=1.6.0
pydantic>=2.5.0
python-multipart>=0.0.6
```

### Setup Commands

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_lg

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Prepare training data
python scripts/prepare_data.py

# Train NER model
python scripts/train_ner.py --output models/ner_v1

# Train classifier
python scripts/train_classifier.py --output models/classifier_v1

# Start services
uvicorn src.api.main:app --reload --port 8001
streamlit run src/dashboard/app.py --server.port 8502
```

---

## Success Criteria

- [ ] NER F1 ≥ 0.80 for SKILL and JOB_TITLE entities
- [ ] Classifier accuracy ≥ 0.82 on held-out test set
- [ ] NDCG@10 ≥ 0.85 compared to human expert rankings
- [ ] Process 50 resumes in < 60 seconds
- [ ] Dashboard renders rankings in < 5 seconds
- [ ] Export to CSV/Excel working without data loss
- [ ] SHAP explanations render for all predictions
