# src/search/comparison_search.py
"""
Simplified Comparison Search: Handle obvious "X vs Y" patterns only.

Much simpler than the complex multi-search system. Just run two targeted 
searches and present side-by-side results using RRF fusion.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.agent.tools.search import search_index_tool
from src.agent.routing.micro_router import extract_comparison_entities
from src.services.models import Passage

logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result from comparison search with side-by-side findings."""
    entity1: str
    entity2: str
    entity1_results: List[Passage] 
    entity2_results: List[Passage]
    merged_results: List[Passage]  # RRF-fused results for unified briefing
    comparison_confidence: float


async def search_comparison(
    query: str,
    search_client,
    embed_client,
    embed_model: str,
    top_k_per_entity: int = 8
) -> Optional[ComparisonResult]:
    """
    Simple comparison search for obvious "X vs Y" queries.
    
    Extracts entities, runs separate searches, merges with RRF.
    Much simpler than the old multi-search complexity.
    
    Args:
        query: Comparison query like "CIU vs ETU" or "compare X and Y"
        search_client: OpenSearch client
        embed_client: Embedding client
        embed_model: Embedding model name
        top_k_per_entity: Results per entity before merging
        
    Returns:
        ComparisonResult with side-by-side findings, None if not a comparison
    """
    logger.info(f"Attempting comparison search for: '{query}'")
    
    # Extract comparison entities using micro-router logic
    entities = extract_comparison_entities(query)
    if not entities:
        logger.info(f"No clear X vs Y pattern found in: '{query}'")
        return None
    
    entity1, entity2 = entities
    logger.info(f"Extracted comparison: '{entity1}' vs '{entity2}'")
    
    # Search for each entity independently
    search_options = SearchOptions(top_k=top_k_per_entity)
    
    try:
        # Parallel searches for both entities
        entity1_search = search_docs(
            f"{entity1} utility service api documentation",
            search_client, embed_client, embed_model, search_options
        )
        
        entity2_search = search_docs(
            f"{entity2} utility service api documentation", 
            search_client, embed_client, embed_model, search_options
        )
        
        # Wait for both searches to complete
        entity1_result = await entity1_search
        entity2_result = await entity2_search
        
        # Tag results with entity labels
        for passage in entity1_result.passages:
            passage.meta["comparison_entity"] = entity1
            passage.meta["comparison_side"] = "entity1"
            
        for passage in entity2_result.passages:
            passage.meta["comparison_entity"] = entity2
            passage.meta["comparison_side"] = "entity2"
        
        # Merge results using simple RRF
        merged_results = _merge_comparison_results(
            entity1_result.passages, entity2_result.passages
        )
        
        # Calculate comparison confidence based on result quality
        confidence = _calculate_comparison_confidence(
            entity1_result.passages, entity2_result.passages, entity1, entity2
        )
        
        logger.info(
            f"Comparison search completed: {len(entity1_result.passages)} + "
            f"{len(entity2_result.passages)} results -> {len(merged_results)} merged"
        )
        
        return ComparisonResult(
            entity1=entity1,
            entity2=entity2,
            entity1_results=entity1_result.passages,
            entity2_results=entity2_result.passages,
            merged_results=merged_results,
            comparison_confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Comparison search failed: {e}")
        return None


def _merge_comparison_results(
    results1: List[Passage], results2: List[Passage], k: int = 60
) -> List[Passage]:
    """
    Merge comparison results using RRF to balance both entities.
    
    Ensures neither entity dominates the merged results completely.
    """
    all_results = []
    
    # Add results from entity1 (with slight boost for diversity)
    for i, passage in enumerate(results1):
        rrf_score = 1.0 / (k + i + 1)
        passage.score = rrf_score * 1.1  # Slight boost for first entity
        all_results.append(passage)
    
    # Add results from entity2  
    for i, passage in enumerate(results2):
        rrf_score = 1.0 / (k + i + 1)
        passage.score = rrf_score
        all_results.append(passage)
    
    # Sort by RRF score (descending)
    merged = sorted(all_results, key=lambda x: x.score, reverse=True)
    
    # Ensure both entities are represented in top results (diversity)
    return _ensure_entity_diversity(merged)


def _ensure_entity_diversity(merged_results: List[Passage]) -> List[Passage]:
    """Ensure both entities are represented in top results."""
    if len(merged_results) <= 6:
        return merged_results
    
    # Count entities in top 6 results  
    top_6 = merged_results[:6]
    entity1_count = sum(1 for p in top_6 if p.meta.get("comparison_side") == "entity1")
    entity2_count = sum(1 for p in top_6 if p.meta.get("comparison_side") == "entity2")
    
    # If one entity dominates completely, inject diversity
    if entity1_count == 0 and entity2_count > 0:
        # Find best entity1 result from remaining
        entity1_results = [p for p in merged_results[6:] if p.meta.get("comparison_side") == "entity1"]
        if entity1_results:
            # Replace lowest scoring result in top 6 with best entity1
            merged_results[5] = entity1_results[0]
            
    elif entity2_count == 0 and entity1_count > 0:
        # Find best entity2 result from remaining
        entity2_results = [p for p in merged_results[6:] if p.meta.get("comparison_side") == "entity2"]
        if entity2_results:
            merged_results[5] = entity2_results[0]
    
    return merged_results


def _calculate_comparison_confidence(
    results1: List[Passage], results2: List[Passage], entity1: str, entity2: str
) -> float:
    """Calculate confidence that this is a meaningful comparison."""
    
    # Base confidence on result counts
    if not results1 or not results2:
        return 0.2  # Low confidence if one side is empty
    
    if len(results1) < 2 or len(results2) < 2:
        return 0.4  # Medium-low if insufficient results
    
    # Check if results actually mention the entities
    entity1_mentions = sum(1 for p in results1 if entity1.lower() in p.text.lower())
    entity2_mentions = sum(1 for p in results2 if entity2.lower() in p.text.lower())
    
    mention_ratio = (entity1_mentions + entity2_mentions) / (len(results1) + len(results2))
    
    # Check average result quality (scores)
    avg_score1 = sum(p.score for p in results1) / len(results1)
    avg_score2 = sum(p.score for p in results2) / len(results2)
    avg_quality = (avg_score1 + avg_score2) / 2
    
    # Combine factors
    confidence = min(mention_ratio + avg_quality, 1.0)
    
    logger.debug(
        f"Comparison confidence: mention_ratio={mention_ratio:.2f}, "
        f"avg_quality={avg_quality:.2f}, final={confidence:.2f}"
    )
    
    return confidence


def render_comparison_briefing(result: ComparisonResult, query: str) -> str:
    """Render comparison result as structured briefing."""
    
    lines = [
        f"# {query}",
        "",
        f"**Comparing {result.entity1} vs {result.entity2}**",
        f"*Found {len(result.entity1_results)} sources for {result.entity1}, "
        f"{len(result.entity2_results)} sources for {result.entity2} "
        f"(confidence: {result.comparison_confidence:.1f})*",
        "",
    ]
    
    # Side-by-side comparison
    lines.extend([
        "## 📊 Side-by-Side Comparison",
        "",
        f"### {result.entity1}",
    ])
    
    if result.entity1_results:
        best1 = result.entity1_results[0]
        lines.append(f"**{best1.meta.get('title', 'Key Information')}**")
        lines.append(f"{best1.text[:200]}...")
        lines.append("")
    
    lines.append(f"### {result.entity2}")
    
    if result.entity2_results:
        best2 = result.entity2_results[0]
        lines.append(f"**{best2.meta.get('title', 'Key Information')}**")
        lines.append(f"{best2.text[:200]}...")
        lines.append("")
    
    # Overall findings
    lines.extend([
        "## 🔍 Additional Context",
        "",
        f"Found {len(result.merged_results)} total relevant documents covering both topics.",
        "",
    ])
    
    # List top sources
    for i, passage in enumerate(result.merged_results[:5], 1):
        title = passage.meta.get("title", f"Source {i}")
        entity = passage.meta.get("comparison_entity", "both")
        lines.append(f"{i}. **{title}** (*{entity}* - relevance: {passage.score:.2f})")
    
    return "\n".join(lines)


# Test function
if __name__ == "__main__":
    # Test entity extraction
    test_queries = [
        "compare CIU vs ETU",
        "CIU vs ETU differences",
        "difference between CIU and ETU",
        "what is better CIU or ETU",  # This won't match
        "general documentation"  # This won't match
    ]
    
    from src.agent.routing.micro_router import extract_comparison_entities
    
    for query in test_queries:
        entities = extract_comparison_entities(query)
        print(f"'{query}' -> {entities}")