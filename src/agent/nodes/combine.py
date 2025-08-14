"""Combine node - merging results from multiple searches."""

import logging
from typing import List, Dict, Any
from services.models import SearchResult, RetrievalResult
from services.retrieve import rrf_fuse_results, mmr_diversify

logger = logging.getLogger(__name__)


async def combine_node(state: dict, config, *, store=None) -> dict:
    """
    Combine and merge results from multiple searches.
    
    This reuses existing BM25+vector fusion logic but makes it callable
    for LangGraph workflows that perform multiple searches.
    
    Follows LangGraph pattern with config parameter and optional store injection.
    
    Args:
        state: Workflow state containing search_results
        config: RunnableConfig with user context and configuration
        store: Optional BaseStore for cross-thread user memory
        
    Returns:
        State update with combined and ranked results
    """
    try:
        all_results = state.get("search_results", [])
        query = state.get("normalized_query", state.get("original_query", ""))
        intent = state.get("intent")
        
        if not all_results:
            logger.warning("No search results to combine")
            return {
                "combined_results": [],
                "final_context": "No relevant information found.",
                "workflow_path": state.get("workflow_path", []) + ["combine"]
            }
        
        # Group results by search source/method for intelligent fusion
        grouped_results = {}
        for result in all_results:
            search_method = result.metadata.get("search_method", "unknown")
            search_id = result.metadata.get("search_id", search_method)
            
            if search_id not in grouped_results:
                grouped_results[search_id] = []
            grouped_results[search_id].append(result)
        
        logger.info(f"Combining results from {len(grouped_results)} search sources")
        
        # If we have results from multiple search methods, try to fuse them
        if len(grouped_results) >= 2:
            combined_results = await _fuse_multiple_search_results(
                grouped_results, query, intent
            )
        else:
            # Single search method - just merge and deduplicate
            combined_results = _deduplicate_results(all_results)
        
        # Apply MMR diversification to final results
        if len(combined_results) > 10:
            combined_results = await _apply_mmr_diversification(
                combined_results, query, top_k=10
            )
        
        # Build context from combined results
        final_context = _build_context_from_results(combined_results)
        
        return {
            "combined_results": combined_results,
            "final_context": final_context,
            "workflow_path": state.get("workflow_path", []) + ["combine"],
            "performance_metrics": {
                **state.get("performance_metrics", {}),
                "combined_results_count": len(combined_results),
                "source_groups": len(grouped_results)
            }
        }
        
    except Exception as e:
        logger.error(f"Combine node failed: {e}")
        # Fallback - just use raw results
        all_results = state.get("search_results", [])
        return {
            "combined_results": all_results[:10],  # Limit to top 10
            "final_context": _build_context_from_results(all_results[:10]),
            "workflow_path": state.get("workflow_path", []) + ["combine_error"],
            "error_messages": state.get("error_messages", []) + [f"Combine failed: {e}"]
        }


async def _fuse_multiple_search_results(
    grouped_results: Dict[str, List[SearchResult]], 
    query: str,
    intent
) -> List[SearchResult]:
    """Fuse results from multiple search methods using RRF."""
    try:
        # Convert to format expected by RRF fusion
        search_groups = list(grouped_results.items())
        
        if len(search_groups) == 2:
            # Two groups - can use standard RRF
            group1_name, group1_results = search_groups[0]
            group2_name, group2_results = search_groups[1]
            
            # Convert to (doc_id, score) tuples
            hits1 = [(r.doc_id, r.score) for r in group1_results]
            hits2 = [(r.doc_id, r.score) for r in group2_results]
            
            # Apply RRF fusion
            fused_hits = rrf_fuse_results(hits1, hits2, k_final=15, rrf_k=60)
            
            # Map back to SearchResult objects
            all_results_map = {}
            for result in group1_results + group2_results:
                all_results_map[result.doc_id] = result
            
            fused_results = []
            for doc_id, rrf_score in fused_hits:
                if doc_id in all_results_map:
                    result = all_results_map[doc_id]
                    # Update score to RRF score
                    result.score = rrf_score
                    fused_results.append(result)
            
            return fused_results
        
        else:
            # Multiple groups - use weighted combination
            return _weighted_combine_multiple_groups(grouped_results)
            
    except Exception as e:
        logger.warning(f"RRF fusion failed, using simple merge: {e}")
        # Fallback to simple merge
        all_results = []
        for results in grouped_results.values():
            all_results.extend(results)
        return _deduplicate_results(all_results)


def _weighted_combine_multiple_groups(grouped_results: Dict[str, List[SearchResult]]) -> List[SearchResult]:
    """Combine multiple search groups with weighted scoring."""
    # Assign weights to different search types
    weights = {
        "enhanced_rrf": 1.0,
        "bm25": 0.8,
        "knn": 0.9,
        "comparative": 1.0,
        "parallel": 0.9
    }
    
    weighted_results = {}
    
    for group_name, results in grouped_results.items():
        # Determine weight for this group
        weight = weights.get(group_name, 0.8)
        
        for result in results:
            doc_id = result.doc_id
            weighted_score = result.score * weight
            
            if doc_id in weighted_results:
                # Boost score for documents found in multiple searches
                existing = weighted_results[doc_id]
                existing.score = max(existing.score, weighted_score) * 1.1  # 10% boost
            else:
                # New document
                new_result = SearchResult(
                    doc_id=result.doc_id,
                    content=result.content,
                    score=weighted_score,
                    metadata={**result.metadata, "combined_from": group_name}
                )
                weighted_results[doc_id] = new_result
    
    # Sort by final weighted score
    return sorted(weighted_results.values(), key=lambda x: x.score, reverse=True)


def _deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    """Remove duplicate results based on doc_id."""
    seen_ids = set()
    deduped = []
    
    # Sort by score first to keep highest scoring duplicates
    sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
    
    for result in sorted_results:
        if result.doc_id not in seen_ids:
            seen_ids.add(result.doc_id)
            deduped.append(result)
    
    return deduped


async def _apply_mmr_diversification(results: List[SearchResult], query: str, top_k: int = 10) -> List[SearchResult]:
    """Apply MMR diversification to reduce redundant results."""
    try:
        if len(results) <= top_k:
            return results
        
        # Create document text lookup for MMR
        doc_text_lookup = {}
        candidates = []
        
        for result in results:
            candidates.append(result.doc_id)
            # Combine title and content for diversity analysis
            title = result.metadata.get("title", "")
            doc_text = f"{title} {result.content}"
            doc_text_lookup[result.doc_id] = doc_text
        
        # Apply MMR
        selected_ids, diagnostics = mmr_diversify(
            candidates=candidates,
            doc_text_lookup=doc_text_lookup,
            query=query,
            k=top_k,
            lambda_param=0.75
        )
        
        # Map back to SearchResult objects
        id_to_result = {r.doc_id: r for r in results}
        diversified_results = []
        
        for doc_id in selected_ids:
            if doc_id in id_to_result:
                diversified_results.append(id_to_result[doc_id])
        
        logger.info(f"MMR diversification: {len(results)} -> {len(diversified_results)} results")
        return diversified_results
        
    except Exception as e:
        logger.warning(f"MMR diversification failed: {e}")
        return results[:top_k]  # Simple truncation fallback


def _build_context_from_results(results: List[SearchResult], max_length: int = 6000) -> str:
    """Build human-readable briefing context from search results."""
    if not results:
        return "No relevant information found."
    
    context_parts = []
    current_length = 0
    
    for i, result in enumerate(results):
        # Extract meaningful title and utility info
        title = result.metadata.get("title", result.metadata.get("api_name", f"Document {i+1}"))
        utility_name = result.metadata.get("utility_name", "")
        
        # Clean and format the content for human consumption
        content = result.content.strip()
        if len(content) > 400:
            # Find a natural break point (sentence end, paragraph break)
            truncate_at = 400
            last_sentence = content.rfind('.', 0, truncate_at)
            last_paragraph = content.rfind('\n\n', 0, truncate_at)
            
            if last_sentence > 300:  # Good sentence break found
                content = content[:last_sentence + 1]
            elif last_paragraph > 200:  # Good paragraph break found
                content = content[:last_paragraph]
            else:
                content = content[:truncate_at] + "..."
        
        # Create a clean, scannable brief entry
        if utility_name and utility_name != title:
            part = f"\n**{title}** ({utility_name})\n{content}\n"
        else:
            part = f"\n**{title}**\n{content}\n"
        
        if current_length + len(part) > max_length:
            break
            
        context_parts.append(part)
        current_length += len(part)
    
    context = "".join(context_parts)
    
    if len(results) > len(context_parts):
        context += f"\n*({len(results) - len(context_parts)} additional sources available)*"
    
    return context