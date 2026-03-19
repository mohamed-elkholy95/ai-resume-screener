"""Tests for the data collection module."""
import pytest
import tempfile
from pathlib import Path

from src.data_collection import ResumeParser, Resume, JobDescriptionParser, JobDescription


SAMPLE_RESUME_TEXT = """
John Doe
123 Main Street, New York, NY 10001
john.doe@email.com | (555) 123-4567

PROFESSIONAL SUMMARY
Experienced software engineer with 8 years of experience in Python and machine learning.

SKILLS
Python, Java, JavaScript, Machine Learning, TensorFlow, PyTorch, SQL, Docker, Kubernetes, AWS,
React, Node.js, Git, CI/CD, Agile, Leadership

EXPERIENCE
Senior Software Engineer - TechCorp Inc. (2020 - Present)
- Led a team of 5 developers building ML pipeline
- Developed REST APIs using FastAPI
- Deployed applications on AWS using Docker and Kubernetes

Software Engineer - DataCo (2016 - 2020)
- Built data processing pipelines using Python and Spark
- Developed machine learning models for customer churn prediction

EDUCATION
Bachelor of Science in Computer Science - MIT (2016)

CERTIFICATIONS
AWS Solutions Architect, Certified Kubernetes Administrator
"""


SAMPLE_JD_TEXT = """
Senior ML Engineer

We are looking for a Senior Machine Learning Engineer to join our team.

Requirements:
- 5+ years of experience in software engineering
- Strong proficiency in Python
- Experience with machine learning frameworks (TensorFlow, PyTorch)
- Experience with cloud platforms (AWS, GCP)
- Docker and Kubernetes
- SQL and data processing

Preferred:
- Experience with MLOps tools
- Spark or big data experience
- PhD in Computer Science or related field

Education: Bachelor's degree minimum
"""


class TestResumeParser:
    """Test resume parsing."""

    def test_parse_text(self):
        parser = ResumeParser()
        resume = parser.parse_text(SAMPLE_RESUME_TEXT, "john_doe.txt")
        assert isinstance(resume, Resume)
        assert resume.filename == "john_doe.txt"
        assert len(resume.clean_text) > 0

    def test_parse_text_empty(self):
        parser = ResumeParser()
        resume = parser.parse_text("", "empty.txt")
        assert isinstance(resume, Resume)
        # __post_init__ calls clean_text on raw_text
        assert resume.clean_text == ""

    def test_parse_text_file(self, tmp_path):
        parser = ResumeParser()
        text_file = tmp_path / "test_resume.txt"
        text_file.write_text(SAMPLE_RESUME_TEXT)

        resume = parser.parse_file(str(text_file))
        assert isinstance(resume, Resume)
        assert resume.filename == "test_resume.txt"
        assert "python" in resume.clean_text.lower()

    def test_parse_md_file(self, tmp_path):
        parser = ResumeParser()
        md_file = tmp_path / "resume.md"
        md_file.write_text(SAMPLE_RESUME_TEXT)

        resume = parser.parse_file(str(md_file))
        assert isinstance(resume, Resume)
        assert "python" in resume.clean_text.lower()

    def test_parse_unsupported_format(self, tmp_path):
        parser = ResumeParser()
        json_file = tmp_path / "data.json"
        json_file.write_text('{"name": "Test"}')

        with pytest.raises(ValueError, match="Unsupported file extension"):
            parser.parse_file(str(json_file))


class TestCleanText:
    """Test text cleaning."""

    def test_removes_extra_whitespace(self):
        result = ResumeParser.clean_text("hello    world\n\n\n")
        assert "hello   world" not in result
        assert result == "hello world"

    def test_masks_email(self):
        result = ResumeParser.clean_text("Contact: user@example.com")
        assert "user@example.com" not in result
        assert "[EMAIL]" in result

    def test_masks_phone(self):
        result = ResumeParser.clean_text("Phone: 555-123-4567")
        assert "555-123-4567" not in result
        assert "[PHONE]" in result

    def test_normalizes_bullets(self):
        text = "Skills:\n\u2022 Python\n\u2023 Java\n• React"
        result = ResumeParser.clean_text(text)
        assert "\u2022" not in result
        assert "\u2023" not in result
        assert "•" not in result


class TestJobDescriptionParser:
    """Test job description parsing."""

    def test_parse_text(self):
        parser = JobDescriptionParser()
        jd = parser.parse_text(SAMPLE_JD_TEXT, "Senior ML Engineer")
        assert isinstance(jd, JobDescription)
        assert jd.title == "Senior ML Engineer"
        assert len(jd.raw_text) > 0

    def test_parse_file(self, tmp_path):
        parser = JobDescriptionParser()
        jd_file = tmp_path / "ml_engineer.txt"
        jd_file.write_text(SAMPLE_JD_TEXT)

        jd = parser.parse_file(str(jd_file))
        assert isinstance(jd, JobDescription)
        assert "ml engineer" in jd.title.lower()

    def test_parse_file_infers_title(self, tmp_path):
        parser = JobDescriptionParser()
        jd_file = tmp_path / "senior-data-scientist.txt"
        jd_file.write_text("Job description here")

        jd = parser.parse_file(str(jd_file))
        assert "senior" in jd.title.lower()
        assert "data" in jd.title.lower()
        assert "scientist" in jd.title.lower()


class TestResumeDataclass:
    """Test Resume dataclass."""

    def test_post_init_cleans_text(self):
        resume = Resume(raw_text="Hello    World", filename="test.txt")
        # __post_init__ should auto-clean the raw_text
        assert resume.clean_text != ""
        assert "Hello    World" not in resume.clean_text

    def test_custom_values(self):
        resume = Resume(
            raw_text="test",
            filename="test.txt",
            clean_text="custom cleaned",
            skills=["python", "ml"],
            experience_years=5.0,
        )
        assert resume.skills == ["python", "ml"]
        assert resume.experience_years == 5.0
        # If clean_text is provided and non-empty, __post_init__ should keep it
        assert resume.clean_text == "custom cleaned"

    def test_default_values(self):
        resume = Resume(raw_text="", filename="empty.txt")
        assert resume.entities == {}
        assert resume.skills == []
        assert resume.experience_years == 0.0
        assert resume.education == []
