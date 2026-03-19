"""Evaluation metrics for NER, classification, and ranking quality."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import numpy as np  # type: ignore

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    from sklearn.metrics import (  # type: ignore
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed — classification metrics unavailable.")


# ---------------------------------------------------------------------------
# NER metrics
# ---------------------------------------------------------------------------


def compute_ner_metrics(
    predicted: List[Dict[str, List[str]]],
    ground_truth: List[Dict[str, List[str]]],
    entity_types: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-entity-type precision, recall, and F1 using exact string match.

    Args:
        predicted: List of entity dicts (one per document), mapping
            entity-type label to list of extracted strings.
        ground_truth: List of entity dicts in the same format as ``predicted``.
        entity_types: Entity type labels to evaluate. If ``None``, all types
            present in either ``predicted`` or ``ground_truth`` are evaluated.

    Returns:
        Dict mapping each entity type to ``{"precision", "recall", "f1",
        "tp", "fp", "fn"}``.
    """
    if len(predicted) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predicted={len(predicted)}, ground_truth={len(ground_truth)}"
        )

    tp: Dict[str, int] = defaultdict(int)
    fp: Dict[str, int] = defaultdict(int)
    fn: Dict[str, int] = defaultdict(int)

    for pred_doc, true_doc in zip(predicted, ground_truth):
        all_labels = (
            entity_types
            or list(set(list(pred_doc.keys()) + list(true_doc.keys())))
        )
        for label in all_labels:
            pred_set = {s.lower().strip() for s in pred_doc.get(label, [])}
            true_set = {s.lower().strip() for s in true_doc.get(label, [])}
            tp[label] += len(pred_set & true_set)
            fp[label] += len(pred_set - true_set)
            fn[label] += len(true_set - pred_set)

    labels_to_eval = entity_types or sorted(set(list(tp) + list(fp) + list(fn)))
    results: Dict[str, Dict[str, float]] = {}

    for label in labels_to_eval:
        prec = tp[label] / max(tp[label] + fp[label], 1)
        rec = tp[label] / max(tp[label] + fn[label], 1)
        f1 = (
            2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        )
        results[label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp[label],
            "fp": fp[label],
            "fn": fn[label],
        }

    # Macro averages
    if results:
        macro_p = sum(v["precision"] for v in results.values()) / len(results)
        macro_r = sum(v["recall"] for v in results.values()) / len(results)
        macro_f1 = sum(v["f1"] for v in results.values()) / len(results)
        results["MACRO_AVG"] = {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1": round(macro_f1, 4),
        }

    return results


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_proba: Optional[Any] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute a full classification evaluation report.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices.
        y_proba: Predicted probability matrix (n_samples × n_classes), optional.
            Required for ROC-AUC computation.
        class_names: Human-readable class names for the report.

    Returns:
        Dict with ``accuracy``, ``f1_macro``, ``f1_weighted``,
        ``precision_macro``, ``recall_macro``, ``report`` (text), and
        optionally ``roc_auc_macro``.
    """
    if not _SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for classification metrics.")

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }

    if y_proba is not None and _NUMPY_AVAILABLE:
        try:
            proba_arr = np.array(y_proba)
            n_classes = proba_arr.shape[1] if proba_arr.ndim > 1 else 2
            if n_classes == 2:
                auc = roc_auc_score(y_true, proba_arr[:, 1])
            else:
                auc = roc_auc_score(
                    y_true, proba_arr, multi_class="ovr", average="macro"
                )
            metrics["roc_auc_macro"] = round(float(auc), 4)
        except Exception as exc:
            logger.warning("ROC-AUC computation failed: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------


def compute_ranking_metrics(
    rankings: List[str],
    relevant_ids: List[str],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Compute NDCG, MRR, and Precision@K for a ranked list.

    Args:
        rankings: Ordered list of candidate identifiers (best first).
        relevant_ids: Set of candidate identifiers considered relevant.
        k_values: Values of *K* for which to compute Precision@K.
            Defaults to ``[1, 3, 5, 10]``.

    Returns:
        Dict with ``ndcg``, ``mrr``, and ``precision_at_k`` entries.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    relevant_set = set(relevant_ids)

    # Binary relevance vector
    relevance = [1 if r in relevant_set else 0 for r in rankings]

    # NDCG@len(rankings)
    def _dcg(rel: List[int]) -> float:
        return sum(
            r / math.log2(i + 2) for i, r in enumerate(rel)
        )

    ideal = sorted(relevance, reverse=True)
    dcg_val = _dcg(relevance)
    idcg_val = _dcg(ideal)
    ndcg = dcg_val / idcg_val if idcg_val > 0 else 0.0

    # MRR
    mrr = 0.0
    for rank, r in enumerate(relevance, start=1):
        if r == 1:
            mrr = 1.0 / rank
            break

    # Precision@K
    precision_at_k: Dict[str, float] = {}
    for k in k_values:
        top_k = relevance[:k]
        precision_at_k[f"p@{k}"] = round(sum(top_k) / max(k, 1), 4)

    return {
        "ndcg": round(ndcg, 4),
        "mrr": round(mrr, 4),
        **precision_at_k,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_evaluation_report(results: Dict[str, Any]) -> str:
    """Render an evaluation results dict as a Markdown report string.

    Handles NER metrics, classification metrics, and ranking metrics in a
    single unified report.

    Args:
        results: Dict containing any combination of:
            - ``"ner_metrics"``: Output of :func:`compute_ner_metrics`
            - ``"classification_metrics"``: Output of :func:`compute_classification_metrics`
            - ``"ranking_metrics"``: Output of :func:`compute_ranking_metrics`
            - ``"metadata"``: Optional dict with run info (e.g. model name, date).

    Returns:
        Markdown-formatted evaluation report string.
    """
    from datetime import datetime, timezone

    lines: List[str] = [
        "# AI Resume Screener — Evaluation Report",
        f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
    ]

    meta = results.get("metadata", {})
    if meta:
        lines.append("## Run Metadata")
        for key, val in meta.items():
            lines.append(f"- **{key}:** {val}")
        lines.append("")

    # NER metrics
    ner = results.get("ner_metrics")
    if ner:
        lines.append("## NER Entity Metrics")
        lines.append(
            "| Entity Type  | Precision | Recall | F1    | TP | FP | FN |"
        )
        lines.append(
            "|--------------|-----------|--------|-------|----|----|-----|"
        )
        for label, m in ner.items():
            if label == "MACRO_AVG":
                continue
            tp = m.get("tp", "")
            fp = m.get("fp", "")
            fn = m.get("fn", "")
            lines.append(
                f"| {label:<12} | {m['precision']:.3f}     | {m['recall']:.3f}  "
                f"| {m['f1']:.3f} | {tp}  | {fp}  | {fn}  |"
            )
        if "MACRO_AVG" in ner:
            m = ner["MACRO_AVG"]
            lines.append(
                f"| **MACRO AVG**| {m['precision']:.3f}     | {m['recall']:.3f}  "
                f"| {m['f1']:.3f} | —  | —  | —  |"
            )
        lines.append("")

    # Classification metrics
    clf = results.get("classification_metrics")
    if clf:
        lines.append("## Classification Metrics")
        lines.append(f"- **Accuracy:** {clf.get('accuracy', 0):.4f}")
        lines.append(f"- **F1 (macro):** {clf.get('f1_macro', 0):.4f}")
        lines.append(f"- **F1 (weighted):** {clf.get('f1_weighted', 0):.4f}")
        lines.append(f"- **Precision (macro):** {clf.get('precision_macro', 0):.4f}")
        lines.append(f"- **Recall (macro):** {clf.get('recall_macro', 0):.4f}")
        if "roc_auc_macro" in clf:
            lines.append(f"- **ROC-AUC (macro):** {clf['roc_auc_macro']:.4f}")
        if "report" in clf:
            lines.append("\n```")
            lines.append(clf["report"])
            lines.append("```")
        lines.append("")

    # Ranking metrics
    rank = results.get("ranking_metrics")
    if rank:
        lines.append("## Ranking Metrics")
        lines.append(f"- **NDCG:** {rank.get('ndcg', 0):.4f}")
        lines.append(f"- **MRR:** {rank.get('mrr', 0):.4f}")
        for key, val in rank.items():
            if key.startswith("p@"):
                lines.append(f"- **Precision@{key[2:]}:** {val:.4f}")
        lines.append("")

    return "\n".join(lines)
