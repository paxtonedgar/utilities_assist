# src/retrieval/actionability.py
"""
Actionability gate and presenter selection logic.
Determines whether to use procedure, info, or fallback presenter based on detected spans.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

from .suites.registry import detect_spans, actionable_score, actionable_count
from .suites.schemas import Span
from src.services.models import Passage

logger = logging.getLogger(__name__)

# Configuration thresholds
PROCEDURE_SCORE_THRESHOLD = 2.0  # Score needed for procedure presenter
MIN_ACTIONABLE_SPANS = 2  # Minimum spans for high confidence


@dataclass
class ViewResult:
    """Result from a search view (info or procedure)."""

    view: Literal["info", "procedure"]
    fused_top8: List[Passage]  # pre-rerank (dedup by doc_id)
    reranked_topk: Optional[List[Passage]]  # None if timed out or skipped
    metrics: Dict  # bm25_hits, knn_hits, rrf_candidates, ce_kept, timings


@dataclass
class Decision:
    """Presenter selection decision with supporting data."""

    presenter: Literal["procedure", "info", "fallback"]
    reason: str
    view_used: Optional[Literal["procedure", "info"]]
    spans: List[Span]  # if procedure
    sentences: List[Passage]  # if info
    fallback_paragraphs: List[Passage]  # if fallback
    actionable_score: float
    suite_counts: Dict[str, int]
    capability_counts: Dict[str, int]


def choose_presenter(
    info_view: Optional[ViewResult], proc_view: Optional[ViewResult]
) -> Decision:
    """
    Choose the best presenter based on view results and actionability analysis.

    Args:
        info_view: Result from info-focused search (what/definition queries)
        proc_view: Result from procedure-focused search (how/onboarding queries)

    Returns:
        Decision with presenter choice and supporting data
    """
    # Extract top passages for span detection
    passages_to_analyze = []

    if proc_view and proc_view.reranked_topk:
        passages_to_analyze = proc_view.reranked_topk[:8]
    elif proc_view and proc_view.fused_top8:
        passages_to_analyze = proc_view.fused_top8[:8]
    elif info_view and info_view.reranked_topk:
        passages_to_analyze = info_view.reranked_topk[:8]
    elif info_view and info_view.fused_top8:
        passages_to_analyze = info_view.fused_top8[:8]

    # Detect actionable spans
    spans = detect_spans(passages_to_analyze) if passages_to_analyze else []

    # Calculate actionability metrics
    action_score = actionable_score(spans)
    action_count = actionable_count(spans)

    # Count suites and capabilities for telemetry
    suite_counts = {}
    capability_counts = {}

    for span in spans:
        suite_counts[span.suite] = suite_counts.get(span.suite, 0) + 1
        capability_counts[span.capability] = (
            capability_counts.get(span.capability, 0) + 1
        )

    # Decision logic
    if (
        proc_view
        and action_score >= PROCEDURE_SCORE_THRESHOLD
        and action_count >= MIN_ACTIONABLE_SPANS
    ):
        # High-confidence procedure choice
        decision = Decision(
            presenter="procedure",
            reason=f"actionable_score={action_score:.1f}, spans={action_count}",
            view_used="procedure",
            spans=spans,
            sentences=[],
            fallback_paragraphs=[],
            actionable_score=action_score,
            suite_counts=suite_counts,
            capability_counts=capability_counts,
        )

    elif info_view and (info_view.reranked_topk or info_view.fused_top8):
        # Info presenter for definition/explanation queries
        info_passages = info_view.reranked_topk or info_view.fused_top8
        decision = Decision(
            presenter="info",
            reason=f"info_available={len(info_passages)}, low_actionability={action_score:.1f}",
            view_used="info",
            spans=[],
            sentences=info_passages[:6],  # Top sentences for info
            fallback_paragraphs=[],
            actionable_score=action_score,
            suite_counts=suite_counts,
            capability_counts=capability_counts,
        )

    else:
        # Fallback when neither view has good results
        fallback_passages = []
        if proc_view and proc_view.fused_top8:
            fallback_passages = proc_view.fused_top8[:3]
        elif info_view and info_view.fused_top8:
            fallback_passages = info_view.fused_top8[:3]

        decision = Decision(
            presenter="fallback",
            reason="insufficient_results_both_views",
            view_used=None,
            spans=spans,  # Include spans even in fallback for debugging
            sentences=[],
            fallback_paragraphs=fallback_passages,
            actionable_score=action_score,
            suite_counts=suite_counts,
            capability_counts=capability_counts,
        )

    # Log decision for observability
    logger.info(f"Presenter choice: {decision.presenter} ({decision.reason})")
    logger.debug(f"Suite counts: {suite_counts}")
    logger.debug(f"Capability breakdown: {capability_counts}")

    return decision


def materialize(decision: Decision) -> Dict:
    """
    Materialize the decision into structured content for the presenter.

    Args:
        decision: Presenter decision with supporting data

    Returns:
        Dictionary with structured content for the chosen presenter
    """
    if decision.presenter == "procedure":
        return _materialize_procedure(decision)
    elif decision.presenter == "info":
        return _materialize_info(decision)
    else:
        return _materialize_fallback(decision)


def _materialize_procedure(decision: Decision) -> Dict:
    """Materialize content for procedure presenter."""
    # Group spans by suite for organized presentation
    suite_groups = {}
    step_spans = []

    for span in decision.spans:
        if span.type == "step":
            step_spans.append(span)
        else:
            if span.suite not in suite_groups:
                suite_groups[span.suite] = []
            suite_groups[span.suite].append(span)

    # Find prerequisites in passages
    prereq_content = []
    for passage in decision.sentences or []:
        if any(
            word in passage.text.lower()
            for word in ["prereq", "requirement", "before", "first"]
        ):
            prereq_content.append(
                {
                    "text": passage.text[:200] + "..."
                    if len(passage.text) > 200
                    else passage.text,
                    "citation": f"[{passage.title} ▸ {passage.meta.get('section', 'Overview')}]",
                    "url": passage.url,
                }
            )

    return {
        "type": "procedure",
        "prerequisites": prereq_content,
        "steps": [
            {
                "text": span.text,
                "citation": f"[{_get_passage_title(span.doc_id, decision)} ▸ Step]",
                "url": span.url,
            }
            for span in step_spans
        ],
        "suite_actions": {
            suite: [
                {
                    "capability": span.capability,
                    "text": span.text,
                    "citation": f"[{_get_passage_title(span.doc_id, decision)} ▸ {span.capability}]",
                    "url": span.url,
                    "attrs": span.attrs,
                    "cta": _generate_cta(span),
                }
                for span in spans
            ]
            for suite, spans in suite_groups.items()
        },
        "metrics": {
            "actionable_score": decision.actionable_score,
            "suite_counts": decision.suite_counts,
            "capability_counts": decision.capability_counts,
        },
    }


def _materialize_info(decision: Decision) -> Dict:
    """Materialize content for info presenter."""
    sentences = decision.sentences[:6]  # Top 6 passages

    # Extract definition from first passage
    definition = ""
    if sentences:
        first_passage = sentences[0]
        definition = (
            first_passage.text[:300] + "..."
            if len(first_passage.text) > 300
            else first_passage.text
        )

    # Extract key facts from remaining passages
    key_facts = []
    for passage in sentences[1:5]:  # Next 4 passages
        fact_text = (
            passage.text[:150] + "..." if len(passage.text) > 150 else passage.text
        )
        key_facts.append(
            {
                "text": fact_text,
                "citation": f"[{passage.title} ▸ {passage.meta.get('section', 'Overview')}]",
                "url": passage.url,
            }
        )

    return {
        "type": "info",
        "definition": {
            "text": definition,
            "citation": f"[{sentences[0].title} ▸ Definition]" if sentences else "",
            "url": sentences[0].url if sentences else "",
        },
        "key_facts": key_facts,
        "metrics": {
            "actionable_score": decision.actionable_score,
            "suite_counts": decision.suite_counts,
        },
    }


def _materialize_fallback(decision: Decision) -> Dict:
    """Materialize content for fallback presenter."""
    paragraphs = []
    for passage in decision.fallback_paragraphs:
        paragraph_text = (
            passage.text[:400] + "..." if len(passage.text) > 400 else passage.text
        )
        paragraphs.append(
            {
                "text": paragraph_text,
                "citation": f"[{passage.title} ▸ {passage.meta.get('section', 'Content')}]",
                "url": passage.url,
            }
        )

    return {
        "type": "fallback",
        "banner": "Closest guidance found; structured checklist not available.",
        "paragraphs": paragraphs,
        "detected_spans": len(decision.spans),
        "metrics": {
            "actionable_score": decision.actionable_score,
            "suite_counts": decision.suite_counts,
        },
    }


def _get_passage_title(doc_id: str, decision: Decision) -> str:
    """Get passage title from doc_id."""
    # Look through all available passages to find title
    all_passages = decision.sentences + decision.fallback_paragraphs
    for passage in all_passages:
        if passage.doc_id == doc_id:
            return passage.title
    return "Document"


def _generate_cta(span: Span) -> str:
    """Generate call-to-action text based on capability."""
    CTA_MAP = {
        "jira": {
            "ticket.create": "Create ticket",
            "ticket.view": "Open ticket",
            "project.access": "Request access",
        },
        "servicenow": {
            "request.create": "Submit request",
            "request.view": "View request",
            "catalog.browse": "Browse catalog",
        },
        "api": {
            "api.try": "Try endpoint",
            "auth.setup": "Get API key",
            "api.docs": "View docs",
        },
        "teams": {
            "channel.open": "Open channel",
            "meeting.schedule": "Schedule meeting",
            "contact.open": "Contact team",
        },
        "outlook": {
            "email.send": "Send email",
            "calendar.create": "Schedule meeting",
            "contact.open": "Contact DL",
        },
        "global": {
            "step": "View step",
            "owner": "Contact owner",
            "table": "View details",
        },
    }

    suite_ctas = CTA_MAP.get(span.suite, {})
    return suite_ctas.get(span.capability, "Open link")
