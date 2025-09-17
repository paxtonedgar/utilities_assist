from __future__ import annotations

"""Utility functions to evaluate retrieval and labeling impact."""

import logging
from typing import Dict, Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
except ImportError:  # pragma: no cover - degrade gracefully
    accuracy_score = f1_score = precision_recall_fscore_support = None  # type: ignore

logger = logging.getLogger(__name__)


def evaluate_label_stability(
    before_labels: Sequence[str],
    after_labels: Sequence[str],
    gold_labels: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Compute label stability and optional quality metrics."""

    if len(before_labels) != len(after_labels):
        raise ValueError("Input label sequences must be the same length")

    total = len(before_labels)
    if total == 0:
        return {"count": 0}

    unchanged = sum(1 for a, b in zip(before_labels, after_labels) if a == b)
    improved = 0
    degraded = 0
    if gold_labels and len(gold_labels) == total:
        for old, new, gold in zip(before_labels, after_labels, gold_labels):
            if new == gold and old != gold:
                improved += 1
            elif old == gold and new != gold:
                degraded += 1

    metrics = {
        "count": total,
        "stability_ratio": round(unchanged / total, 4),
        "improved": improved,
        "degraded": degraded,
    }

    if gold_labels and len(gold_labels) == total and accuracy_score is not None:
        metrics["accuracy_before"] = round(float(accuracy_score(gold_labels, before_labels)), 4)
        metrics["accuracy_after"] = round(float(accuracy_score(gold_labels, after_labels)), 4)
        metrics["macro_f1_before"] = round(float(f1_score(gold_labels, before_labels, average="macro")), 4)
        metrics["macro_f1_after"] = round(float(f1_score(gold_labels, after_labels, average="macro")), 4)
    else:
        logger.debug("Skipping accuracy/F1 metrics; sklearn not available or gold labels missing")

    return metrics


def evaluate_retrieval_recall(
    gold: Sequence[Iterable[str]],
    candidates_before: Sequence[Iterable[str]],
    candidates_after: Sequence[Iterable[str]],
) -> Dict[str, float]:
    """Compare recall@K before and after catalog enrichment."""

    if not (len(gold) == len(candidates_before) == len(candidates_after)):
        raise ValueError("Gold and candidate collections must align")

    def recall_at_k(g, c) -> float:
        gset = set(g)
        if not gset:
            return 1.0
        return len(gset.intersection(c)) / len(gset)

    recalls_before = [recall_at_k(g, c) for g, c in zip(gold, candidates_before)]
    recalls_after = [recall_at_k(g, c) for g, c in zip(gold, candidates_after)]

    if not recalls_before:
        return {"count": 0}

    lift = sum(recalls_after) / len(recalls_after) - sum(recalls_before) / len(recalls_before)
    return {
        "count": len(recalls_before),
        "recall_before": round(sum(recalls_before) / len(recalls_before), 4),
        "recall_after": round(sum(recalls_after) / len(recalls_after), 4),
        "recall_lift": round(lift, 4),
    }


def precision_recall_breakdown(
    gold_mentions: Sequence[str],
    predicted_mentions: Sequence[str],
) -> Dict[str, float]:
    """Simple precision/recall wrapper for gazetteer validation."""

    if precision_recall_fscore_support is None:
        logger.debug("scikit-learn not available; precision/recall not computed")
        return {}

    precision, recall, f1, _ = precision_recall_fscore_support(  # type: ignore
        gold_mentions,
        predicted_mentions,
        average="binary",
        zero_division=0,
    )
    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }

