"""Tests for evaluation metrics."""
import pytest
import numpy as np
from src.evaluation import (
    compute_adverse_impact_ratio,
    compute_classification_metrics,
    compute_ranking_metrics,
    compute_score_parity,
    generate_evaluation_report,
)


class TestClassificationMetrics:
    """Test classification metric computation."""

    def test_perfect_classification(self):
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 1, 0, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["recall_macro"] == 1.0
        assert metrics["precision_macro"] == 1.0

    def test_all_negative(self):
        y_true = [0, 0, 0]
        y_pred = [0, 0, 0]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0

    def test_all_positive(self):
        y_true = [1, 1, 1]
        y_pred = [1, 1, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0

    def test_balanced(self):
        y_true = [1, 1, 0, 0, 1, 0]
        y_pred = [1, 0, 0, 0, 1, 1]
        metrics = compute_classification_metrics(y_true, y_pred)
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics

    def test_with_probabilities(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 1]
        # Must be a 2D probability matrix
        y_proba = [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.6, 0.4]]
        metrics = compute_classification_metrics(y_true, y_pred, y_proba=y_proba)
        assert "roc_auc_macro" in metrics
        assert 0.0 <= metrics["roc_auc_macro"] <= 1.0

    def test_metrics_range(self):
        y_true = [1, 1, 0, 0, 1, 0, 1, 0]
        y_pred = [1, 0, 0, 0, 1, 1, 0, 0]
        metrics = compute_classification_metrics(y_true, y_pred)
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"

    def test_returns_report(self):
        metrics = compute_classification_metrics([1, 0], [1, 0])
        assert "report" in metrics
        assert isinstance(metrics["report"], str)


class TestRankingMetrics:
    """Test ranking metric computation."""

    def test_perfect_ranking(self):
        rankings = ["doc1", "doc2", "doc3"]
        relevant = ["doc1", "doc2", "doc3"]
        metrics = compute_ranking_metrics(rankings, relevant)
        assert metrics["ndcg"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_no_relevant(self):
        rankings = ["doc1", "doc2", "doc3"]
        relevant = ["doc4", "doc5"]
        metrics = compute_ranking_metrics(rankings, relevant)
        assert metrics["ndcg"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_partial_relevant(self):
        rankings = ["doc2", "doc3", "doc1"]
        relevant = ["doc1", "doc2", "doc3"]
        metrics = compute_ranking_metrics(rankings, relevant)
        # All docs are relevant, so ndcg should be 1.0
        assert metrics["ndcg"] == 1.0
        # doc2 is at rank 1 (relevant), so mrr should be 1.0
        assert metrics["mrr"] == 1.0

    def test_single_relevant_at_top(self):
        rankings = ["doc1", "doc2", "doc3"]
        relevant = ["doc1"]
        metrics = compute_ranking_metrics(rankings, relevant)
        assert metrics["mrr"] == 1.0
        assert metrics["p@1"] == 1.0

    def test_single_relevant_at_bottom(self):
        rankings = ["doc2", "doc3", "doc1"]
        relevant = ["doc1"]
        metrics = compute_ranking_metrics(rankings, relevant)
        assert metrics["mrr"] < 1.0
        assert metrics["mrr"] > 0.0

    def test_empty_rankings(self):
        metrics = compute_ranking_metrics([], ["doc1"])
        assert metrics["mrr"] == 0.0

    def test_precision_at_k(self):
        rankings = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant = ["doc1", "doc3"]
        metrics = compute_ranking_metrics(rankings, relevant, k_values=[1, 3, 5])
        assert "p@1" in metrics
        assert "p@3" in metrics
        assert "p@5" in metrics
        assert metrics["p@1"] == 1.0  # doc1 is relevant
        assert metrics["p@3"] == pytest.approx(2/3, abs=0.01)
        assert metrics["p@5"] == pytest.approx(2/5, abs=0.01)


class TestEvaluationReport:
    """Test report generation."""

    def test_generate_report_with_classification(self):
        clf_metrics = compute_classification_metrics([1, 0, 1, 0], [1, 0, 0, 1])
        results = {
            "classification_metrics": clf_metrics,
            "metadata": {"model": "logistic_regression"},
        }
        report = generate_evaluation_report(results)
        assert isinstance(report, str)
        assert "logistic_regression" in report
        assert "Classification Metrics" in report
        assert "##" in report  # Markdown headers

    def test_report_with_ranking(self):
        rank_metrics = compute_ranking_metrics(
            ["doc1", "doc2", "doc3"], ["doc1", "doc2"])
        results = {"ranking_metrics": rank_metrics}
        report = generate_evaluation_report(results)
        assert isinstance(report, str)
        assert "ndcg" in report.lower()

    def test_report_with_ner(self):
        ner_metrics = {
            "NAME": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "tp": 9, "fp": 1, "fn": 2},
            "MACRO_AVG": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
        }
        results = {"ner_metrics": ner_metrics}
        report = generate_evaluation_report(results)
        assert "NER Entity Metrics" in report
        assert "NAME" in report

    def test_empty_results(self):
        report = generate_evaluation_report({})
        assert isinstance(report, str)
        assert "Evaluation Report" in report

    def test_combined_report(self):
        clf = compute_classification_metrics([1, 0], [1, 0])
        rank = compute_ranking_metrics(["a", "b"], ["a"])
        results = {
            "classification_metrics": clf,
            "ranking_metrics": rank,
            "metadata": {"run": "test"},
        }
        report = generate_evaluation_report(results)
        assert "Classification" in report
        assert "Ranking" in report
        assert "test" in report


class TestAdverseImpactRatio:
    """Test the EEOC four-fifths rule implementation."""

    def test_equal_pass_rates(self):
        result = compute_adverse_impact_ratio({"A": 0.60, "B": 0.60})
        assert result["flagged"] is False
        assert result["air_values"]["A"] == 1.0

    def test_flagged_when_below_threshold(self):
        # Group B has 50% pass rate vs Group A's 80% → AIR = 0.625 < 0.80
        result = compute_adverse_impact_ratio({"A": 0.80, "B": 0.50})
        assert result["flagged"] is True
        assert result["air_values"]["B"] < 0.80

    def test_not_flagged_at_boundary(self):
        # Exactly at 80% of reference → AIR = 0.80, not flagged
        result = compute_adverse_impact_ratio({"A": 1.0, "B": 0.80})
        assert result["flagged"] is False

    def test_custom_reference_group(self):
        result = compute_adverse_impact_ratio(
            {"A": 0.90, "B": 0.70, "C": 0.60},
            reference_group="A",
        )
        assert result["reference_group"] == "A"
        assert result["reference_rate"] == 0.90

    def test_auto_selects_highest_pass_rate(self):
        result = compute_adverse_impact_ratio({"X": 0.40, "Y": 0.90, "Z": 0.70})
        assert result["reference_group"] == "Y"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_adverse_impact_ratio({})

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            compute_adverse_impact_ratio({"A": 1.5})

    def test_unknown_reference_group_raises(self):
        with pytest.raises(ValueError):
            compute_adverse_impact_ratio({"A": 0.5}, reference_group="Z")


class TestScoreParity:
    """Test score distribution parity across groups."""

    def test_equal_means(self):
        result = compute_score_parity({
            "A": [0.80, 0.82, 0.78],
            "B": [0.79, 0.81, 0.80],
        })
        assert result["flagged"] is False
        assert result["max_disparity"] < 0.10

    def test_flagged_when_large_gap(self):
        result = compute_score_parity({
            "High": [0.90, 0.88, 0.92],
            "Low": [0.50, 0.48, 0.52],
        })
        assert result["flagged"] is True
        assert result["max_disparity"] > 0.10

    def test_empty_groups(self):
        result = compute_score_parity({})
        assert result["flagged"] is False
        assert result["overall_mean"] == 0.0

    def test_single_group(self):
        result = compute_score_parity({"Only": [0.75, 0.80]})
        assert result["max_disparity"] == 0.0
        assert result["flagged"] is False
