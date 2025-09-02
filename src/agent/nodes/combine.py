"""Enhanced combine node with evidence-gated briefing composition."""

import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
from src.services.models import Passage
from .base_node import to_state_dict, from_state_dict

logger = logging.getLogger(__name__)


class BriefingSection(Enum):
    """Evidence-gated briefing sections."""
    INFO = "info"
    PROCEDURE = "procedure"
    API = "api"
    TROUBLESHOOT = "troubleshoot"
    LISTS = "lists"


@dataclass
class BriefingComposition:
    """Container for evidence-gated briefing composition."""
    sections: Dict[BriefingSection, List[Passage]]
    composition_strategy: str
    total_passages: int


async def combine_node(state, config=None, *, store=None):
    """
    Combine and merge results from searches with minimal complexity.

    Simplified from 477 lines to ~80 lines by removing:
    - Complex domain boosting logic
    - Elaborate quality gates
    - Multi-search fusion complexity
    - MMR diversification
    """
    incoming_type = type(state)

    try:
        # Convert to dict for processing
        s = to_state_dict(state)

        logger.info(
            "NODE_START combine | keys=%s | search_results_count=%d",
            list(s.keys()),
            len(s.get("search_results", [])),
        )

        all_results = s.get("search_results", [])
        query = s.get("normalized_query", s.get("original_query", ""))

        if not all_results:
            logger.warning("No search results to combine")
            return _create_no_results_response(s, incoming_type, query)

        # Simple deduplication and sorting - no complex fusion
        combined_results = _deduplicate_and_sort(all_results)

        # Limit to reasonable number of results
        if len(combined_results) > 12:
            combined_results = combined_results[:12]

        # Simple quality check
        if len(combined_results) < 2:
            logger.info(
                f"Low document count ({len(combined_results)}) for query: {query[:50]}..."
            )

        # Debug logging
        if combined_results:
            top_results_summary = [
                (r.meta.get("title", f"doc_{r.doc_id[:8]}"), round(float(r.score), 3))
                for r in combined_results[:5]
            ]
            logger.info(f"COMBINED_TOP: {top_results_summary}")

        # Build context
        final_context = _build_simple_context(combined_results)

        logger.info(
            f"NODE_END combine | combined_results={len(combined_results)} | context_length={len(final_context)}"
        )

        # Return state update
        merged = {
            **s,
            "combined_results": combined_results,
            "final_context": final_context,
            "workflow_path": s.get("workflow_path", []) + ["combine"],
        }
        return from_state_dict(incoming_type, merged)

    except Exception as e:
        logger.error(f"Combine node failed: {e}")
        return from_state_dict(
            incoming_type,
            {**s, "error_messages": s.get("error_messages", []) + [str(e)]},
        )


def _create_no_results_response(s: dict, incoming_type, query: str):
    """Create response when no results found."""
    # Generate search suggestions
    suggestions = [
        f"{query} setup documentation",
        f"{query} API reference",
        f"{query} integration guide",
    ]

    suggestion_text = "Try these searches: " + " | ".join(f'"{s}"' for s in suggestions)

    merged = {
        **s,
        "combined_results": [],
        "final_context": f"No documentation found for '{query}'.",
        "final_answer": f"No relevant documentation found. {suggestion_text}",
        "workflow_path": s.get("workflow_path", []) + ["combine", "no_results"],
    }
    return from_state_dict(incoming_type, merged)


def _deduplicate_and_sort(results: List[Passage]) -> List[Passage]:
    """Simple deduplication by doc_id and sort by score."""
    seen_ids = set()
    deduped = []

    for result in results:
        if result.doc_id not in seen_ids:
            seen_ids.add(result.doc_id)
            deduped.append(result)

    # Sort by relevance score (already set by reranker)
    return sorted(deduped, key=lambda x: x.score, reverse=True)


def _build_simple_context(results: List[Passage], max_length: int = 8000) -> str:
    """Build simple context without complex formatting."""
    if not results:
        return ""

    context_parts = []
    current_length = 0

    for i, result in enumerate(results, 1):
        # Simple format: [Source N: Title] Content
        title = result.meta.get("title", f"Document {i}")
        doc_context = f"[Source {i}: {title}]\n{result.text.strip()}\n"

        if current_length + len(doc_context) > max_length:
            break

        context_parts.append(doc_context)
        current_length += len(doc_context)

    context = "\n".join(context_parts)

    if current_length >= max_length:
        context += (
            f"\n[Note: Context truncated - showing top {len(context_parts)} results]"
        )

    return context


# Evidence-Gated Briefing Composer Functions
def compose_evidence_briefing(passages: List[Passage], query: str, max_sections: int = 4) -> BriefingComposition:
    """
    Compose briefing with evidence-gated sections.
    Only includes sections that have sufficient evidence.
    """
    if not passages:
        return BriefingComposition(sections={}, composition_strategy="no_evidence", total_passages=0)
    
    # Analyze passages for section evidence
    section_evidence = _analyze_section_evidence(passages, query)
    
    # Apply evidence gates (minimum 2 passages, confidence >= 0.4)
    gated_sections = _apply_evidence_gates(section_evidence)
    
    # Limit to max sections (prioritize by evidence strength)
    final_sections = _prioritize_sections(gated_sections, max_sections)
    
    strategy = _determine_composition_strategy(final_sections)
    
    logger.info(f"Evidence briefing: {len(final_sections)} sections from {len(passages)} passages")
    
    return BriefingComposition(
        sections=final_sections,
        composition_strategy=strategy,
        total_passages=len(passages)
    )


def _analyze_section_evidence(passages: List[Passage], query: str) -> Dict[BriefingSection, List[Passage]]:
    """Analyze passages to determine which briefing sections they support."""
    section_evidence = {section: [] for section in BriefingSection}
    
    for passage in passages:
        sections = _classify_passage_sections(passage, query)
        for section in sections:
            section_evidence[section].append(passage)
    
    return section_evidence


def _classify_passage_sections(passage: Passage, query: str) -> Set[BriefingSection]:
    """Classify which briefing sections a passage supports."""
    if not passage or not passage.text:
        return set()
    
    title = passage.meta.get("title", "").lower()
    content = passage.text.lower()
    combined_text = f"{title} {content}"
    
    sections = set()
    
    # API section patterns
    api_patterns = [
        "api", "endpoint", "swagger", "openapi", "rest", "/api/",
        "get ", "post ", "put ", "delete ", "http method"
    ]
    if any(pattern in combined_text for pattern in api_patterns):
        sections.add(BriefingSection.API)
    
    # Procedure section patterns
    procedure_patterns = [
        "step", "procedure", "how to", "setup", "configure", 
        "install", "create", "follow", "first", "then", "next"
    ]
    if any(pattern in combined_text for pattern in procedure_patterns):
        sections.add(BriefingSection.PROCEDURE)
    
    # Troubleshooting section patterns
    troubleshoot_patterns = [
        "error", "troubleshoot", "problem", "issue", "fix", "resolve",
        "failed", "not working", "warning", "exception"
    ]
    if any(pattern in combined_text for pattern in troubleshoot_patterns):
        sections.add(BriefingSection.TROUBLESHOOT)
    
    # Lists section patterns
    list_patterns = [
        "list of", "available", "supported", "options", "types",
        "all ", "following", "include:", "such as"
    ]
    if any(pattern in combined_text for pattern in list_patterns):
        sections.add(BriefingSection.LISTS)
    
    # Default to INFO if no specific section detected
    if not sections:
        sections.add(BriefingSection.INFO)
    
    return sections


def _apply_evidence_gates(section_evidence: Dict[BriefingSection, List[Passage]]) -> Dict[BriefingSection, List[Passage]]:
    """Apply evidence gates - only include sections with sufficient evidence."""
    MIN_PASSAGES = 1  # Lowered from 2 - allow single high-quality results
    MIN_CONFIDENCE = 0.005  # Lowered for ranx RRF scores (typically 0.01-0.02 range)
    
    gated_sections = {}
    
    for section, passages in section_evidence.items():
        if len(passages) < MIN_PASSAGES:
            continue
            
        # Check average confidence
        avg_confidence = sum(p.score for p in passages) / len(passages)
        if avg_confidence < MIN_CONFIDENCE:
            continue
            
        gated_sections[section] = passages
        logger.debug(f"Section {section.value}: {len(passages)} passages, conf={avg_confidence:.2f}")
    
    return gated_sections


def _prioritize_sections(sections: Dict[BriefingSection, List[Passage]], max_sections: int) -> Dict[BriefingSection, List[Passage]]:
    """Prioritize sections by evidence strength and limit to max_sections."""
    if len(sections) <= max_sections:
        return sections
    
    # Sort by evidence strength (passage count * average confidence)
    section_scores = []
    for section, passages in sections.items():
        avg_confidence = sum(p.score for p in passages) / len(passages)
        evidence_strength = len(passages) * avg_confidence
        section_scores.append((evidence_strength, section, passages))
    
    # Take top sections
    section_scores.sort(reverse=True)
    prioritized = {}
    for _, section, passages in section_scores[:max_sections]:
        prioritized[section] = passages
    
    return prioritized


def _determine_composition_strategy(sections: Dict[BriefingSection, List[Passage]]) -> str:
    """Determine composition strategy based on sections found."""
    if not sections:
        return "no_evidence"
    elif len(sections) == 1:
        return "single_section"
    elif BriefingSection.API in sections and BriefingSection.PROCEDURE in sections:
        return "api_procedure"
    elif BriefingSection.TROUBLESHOOT in sections:
        return "troubleshooting_focused"
    else:
        return "multi_section"


def render_briefing_markdown(composition: BriefingComposition) -> str:
    """Render evidence-gated briefing as markdown."""
    if not composition.sections:
        return "No sufficient evidence found to compose a structured briefing."
    
    lines = []
    
    for section, passages in composition.sections.items():
        # Section header
        section_title = _get_section_title(section)
        lines.append(f"## {section_title}\n")
        
        # Combine passage content for this section
        section_content = _combine_section_passages(passages)
        lines.append(section_content)
        lines.append("")  # Blank line between sections
    
    return "\n".join(lines)


def _get_section_title(section: BriefingSection) -> str:
    """Get display title for briefing section."""
    titles = {
        BriefingSection.INFO: "Overview",
        BriefingSection.PROCEDURE: "Steps & Procedures", 
        BriefingSection.API: "API Reference",
        BriefingSection.TROUBLESHOOT: "Troubleshooting",
        BriefingSection.LISTS: "Available Options"
    }
    return titles.get(section, section.value.title())


def _combine_section_passages(passages: List[Passage]) -> str:
    """Combine passages for a specific section."""
    if not passages:
        return "No information available."
    
    # Sort by relevance
    sorted_passages = sorted(passages, key=lambda x: x.score, reverse=True)
    
    # Combine top passages
    content_parts = []
    for i, passage in enumerate(sorted_passages[:3], 1):  # Limit to top 3 per section
        title = passage.meta.get("title", f"Source {i}")
        snippet = passage.text.strip()
        
        # Truncate if too long
        if len(snippet) > 500:
            snippet = snippet[:500] + "..."
            
        content_parts.append(f"**{title}:**\n{snippet}")
    
    return "\n\n".join(content_parts)
