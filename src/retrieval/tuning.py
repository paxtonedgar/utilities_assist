"""Retrieval tuning functions that use colors for KISS knob adjustment."""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def tune_for_colors(knobs: Dict[str, Any], colors) -> None:
    """
    Adjust retrieval knobs based on coloring attributes.

    Single tuning function with no DSL edits - mutates existing knobs dict
    that nodes already pass to their retrieval functions.

    Args:
        knobs: Mutable knobs dict (e.g., {"knn_k": 30, "bm25_size": 50})
        colors: Colors object from slotter

    Examples:
        >>> knobs = {"knn_k": 30, "bm25_size": 50, "ce_timeout_s": 1.8}
        >>> # Mock colors with high specificity
        >>> colors = Colors(specificity="high", ...)
        >>> tune_for_colors(knobs, colors)
        >>> knobs["knn_k"] <= 24  # Narrowed for high specificity
        True
    """
    from src.retrieval.config import TUNING_ADJUSTMENTS

    try:
        # Specificity-based tuning
        if colors.specificity == "high":
            # Narrow search for high specificity queries
            adj = TUNING_ADJUSTMENTS["specificity_high"]
            knobs["knn_k"] = min(knobs.get("knn_k", 30), adj["knn_k_max"])
            knobs["bm25_size"] = min(knobs.get("bm25_size", 50), adj["bm25_size_max"])

            logger.debug(
                f"High specificity: narrowed knn_k={knobs['knn_k']}, bm25_size={knobs['bm25_size']}"
            )

        elif colors.specificity == "low":
            # Widen search for low specificity queries
            adj = TUNING_ADJUSTMENTS["specificity_low"]
            knobs["knn_k"] = max(knobs.get("knn_k", 30), adj["knn_k_min"])
            knobs["bm25_size"] = max(knobs.get("bm25_size", 50), adj["bm25_size_min"])

            logger.debug(
                f"Low specificity: widened knn_k={knobs['knn_k']}, bm25_size={knobs['bm25_size']}"
            )

        # Suite affinity boosting
        if colors.suite_affinity:
            top_suite, top_score = _get_top_suite(colors.suite_affinity)

            if top_score >= 0.6:  # Strong suite affinity
                _add_suite_booster(knobs, top_suite, top_score)
                logger.debug(
                    f"Added suite booster: {top_suite} (score={top_score:.2f})"
                )

        # Time urgency adjustments
        if colors.time_urgency == "hard":
            adj = TUNING_ADJUSTMENTS["time_urgency_hard"]

            # Reduce CE timeout for urgent queries
            original_timeout = knobs.get("ce_timeout_s", 1.8)
            knobs["ce_timeout_s"] = original_timeout * adj["ce_timeout_factor"]

            # Skip reranking if time budget is tight
            if adj["skip_rerank_if_time_low"]:
                knobs["skip_rerank_if_time_low"] = True

            logger.debug(
                f"Hard urgency: reduced ce_timeout_s={knobs['ce_timeout_s']:.1f}s"
            )

        # Troubleshooting focus
        if colors.troubleshoot_flag:
            _add_troubleshoot_boosters(knobs)
            logger.debug("Added troubleshooting boosters")

    except Exception as e:
        logger.error(f"Tuning adjustment failed: {e}")
        # Don't fail the request - just skip tuning


def _get_top_suite(suite_affinity: Dict[str, float]) -> tuple:
    """Get the top suite by affinity score."""
    if not suite_affinity:
        return None, 0.0

    # suite_affinity is already sorted by SuiteAffinityCalculator
    top_suite = next(iter(suite_affinity.items()))
    return top_suite


def _add_suite_booster(knobs: Dict[str, Any], suite: str, score: float) -> None:
    """Add suite-specific booster to knobs."""
    # Add to existing boosters list or create new one
    boosters = knobs.get("soft_boosters", [])

    # Suite-specific booster (implementation depends on existing search architecture)
    booster = {
        "type": "suite_affinity",
        "suite": suite,
        "score_multiplier": min(1.0 + score * 0.5, 1.5),  # Cap at 1.5x boost
    }

    boosters.append(booster)
    knobs["soft_boosters"] = boosters


def _add_troubleshoot_boosters(knobs: Dict[str, Any]) -> None:
    """Add troubleshooting-specific boosters."""
    boosters = knobs.get("soft_boosters", [])

    # Prefer runbook/incident content
    troubleshoot_booster = {
        "type": "keyword_boost",
        "keywords": ["runbook", "incident", "troubleshoot", "error", "fix"],
        "score_multiplier": 1.3,
    }

    boosters.append(troubleshoot_booster)
    knobs["soft_boosters"] = boosters


def should_build_procedure_view(slot_result) -> bool:
    """
    Determine if procedure view should be built using colors for planning.

    Separates planning (use colors) from verdict (use extracted spans).
    This is the "planning" phase - final presenter choice uses spans.
    """
    from src.retrieval.config import VIEW_SETTINGS

    build_rules = VIEW_SETTINGS["build_procedure_if"]
    colors = slot_result.colors

    # Always build if doish or mixed intent
    if build_rules["doish"] and slot_result.doish:
        return True

    if build_rules["intent_mixed"] and slot_result.intent == "mixed":
        return True

    # Build if high actionability estimate
    if colors.actionability_est >= build_rules["actionability_threshold"]:
        return True

    # Build if strong suite affinity
    max_suite_score = (
        max(colors.suite_affinity.values()) if colors.suite_affinity else 0.0
    )
    if max_suite_score >= build_rules["suite_affinity_threshold"]:
        return True

    # Build if required artifacts are present (OR logic)
    required_artifacts = build_rules["required_artifacts"]
    if colors.artifact_types & required_artifacts:  # Set intersection
        return True

    return False
