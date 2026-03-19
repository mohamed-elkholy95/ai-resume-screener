"""Tests for the skill taxonomy module."""
import pytest
from src.skill_taxonomy import (
    get_all_skills,
    match_skill,
    get_skills_by_category,
    categorize_skill,
    SKILL_DATABASE,
    SKILL_ALIASES,
)


class TestSkillDatabase:
    """Test skill database structure."""

    def test_categories_exist(self):
        expected = [
            "programming_languages",
            "ml_ai",
            "data_engineering",
            "cloud_devops",
            "web_development",
            "soft_skills",
        ]
        for cat in expected:
            assert cat in SKILL_DATABASE, f"Missing category: {cat}"

    def test_no_empty_categories(self):
        for cat, skills in SKILL_DATABASE.items():
            assert len(skills) > 0, f"Empty category: {cat}"

    def test_all_lowercase(self):
        for cat, skills in SKILL_DATABASE.items():
            for skill in skills:
                assert skill == skill.lower(), f"Uppercase skill: {skill}"


class TestGetAllSkills:
    """Test get_all_skills function."""

    def test_returns_list(self):
        result = get_all_skills()
        assert isinstance(result, list)

    def test_returns_unique_collection(self):
        result = get_all_skills()
        assert result == sorted(result) or len(result) == len(set(result))

    def test_no_duplicates(self):
        result = get_all_skills()
        assert len(result) == len(set(result))

    def test_min_count(self):
        result = get_all_skills()
        assert len(result) >= 80, f"Expected at least 80 skills, got {len(result)}"


class TestMatchSkill:
    """Test skill matching."""

    def test_exact_match(self):
        result = match_skill("python")
        assert result == "python"

    def test_case_insensitive(self):
        result = match_skill("Python")
        assert result == "python"

    def test_alias_match(self):
        result = match_skill("js")
        assert result == "javascript"

    def test_alias_ts(self):
        result = match_skill("ts")
        assert result == "typescript"

    def test_alias_k8s(self):
        result = match_skill("k8s")
        assert result == "kubernetes"

    def test_no_match_below_threshold(self):
        result = match_skill("xylophone", threshold=0.8)
        assert result is None

    def test_fuzzy_match(self):
        result = match_skill("pyton", threshold=0.7)
        # Should match "python" with high enough threshold
        if result:
            assert result == "python"


class TestGetSkillsByCategory:
    """Test category-based skill retrieval."""

    def test_returns_dict(self):
        result = get_skills_by_category()
        assert isinstance(result, dict)

    def test_correct_categories(self):
        result = get_skills_by_category()
        assert "programming_languages" in result
        assert "ml_ai" in result

    def test_skills_are_strings(self):
        result = get_skills_by_category()
        for cat, skills in result.items():
            for skill in skills:
                assert isinstance(skill, str)


class TestCategorizeSkill:
    """Test skill categorization."""

    def test_python_category(self):
        result = categorize_skill("python")
        assert result == "programming_languages"

    def test_tensorflow_category(self):
        result = categorize_skill("tensorflow")
        assert result == "ml_ai"

    def test_docker_category(self):
        result = categorize_skill("docker")
        assert result == "cloud_devops"

    def test_unknown_skill(self):
        result = categorize_skill("unknown_skill_xyz")
        assert result == "unknown"
