"""Rerank node for cross-encoder relevance scoring in LangGraph workflow.

Plugs into the graph between search and combine stages:
normalize → classify_intent → search_(bm25/knn/hybrid) → **rerank (local)** → combine → answer
"""

import logging
from typing import Dict, Any, Optional

from src.agent.nodes.base_node import BaseNodeHandler
from src.services.reranker import get_reranker, is_reranker_available, RerankResult
from src.services.models import RetrievalResult, SearchResult
from src.infra.resource_manager import get_resources
from src.telemetry.logger import log_event

logger = logging.getLogger(__name__)


class RerankNode(BaseNodeHandler):
    """Cross-encoder rerank node that scores and filters retrieved documents.
    
    Takes the current query and retrieved passages, assigns relevance scores,
    reorders/filters them, and exposes scores for confidence/coverage logic.
    """
    
    def __init__(self):
        super().__init__("rerank")
    
    async def execute(self, state: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute reranking on search results.
        
        Args:
            state: Graph state containing search_results and normalized_query
            config: Optional configuration
            
        Returns:
            Updated state with reranked search_results and rerank metrics
        """
        # Get current query and search results
        normalized_query = state.get("normalized_query", "")
        search_results = state.get("search_results", [])
        
        # Early exit if no query or results
        if not normalized_query or not search_results:
            logger.info("No query or search results to rerank")
            return {
                "search_results": search_results,
                "rerank_metrics": {
                    "rerank_enabled": False,
                    "reason": "no_query_or_results",
                    "original_count": len(search_results),
                    "final_count": len(search_results)
                }
            }
        
        # Get resources and reranker configuration
        resources = get_resources()
        if not resources:
            logger.warning("No resources available - skipping rerank")
            return {"search_results": search_results}
        
        reranker_config = resources.settings.reranker
        
        # Check if reranker is enabled and available
        if not reranker_config.enabled:
            logger.info("Reranker disabled in config")
            return {
                "search_results": search_results,
                "rerank_metrics": {
                    "rerank_enabled": False,
                    "reason": "disabled_in_config",
                    "original_count": len(search_results),
                    "final_count": len(search_results)
                }
            }
        
        if not is_reranker_available():
            logger.warning("Reranker dependencies not available - skipping rerank")
            return {
                "search_results": search_results,
                "rerank_metrics": {
                    "rerank_enabled": False,
                    "reason": "dependencies_not_available",
                    "original_count": len(search_results),
                    "final_count": len(search_results)
                }
            }
        
        # Get reranker instance
        reranker = get_reranker(
            model_id=reranker_config.model_id,
            device=reranker_config.device,
            batch_size=reranker_config.batch_size,
            max_length=reranker_config.max_length,
            use_fp16=reranker_config.use_fp16
        )
        
        if not reranker:
            logger.error("Failed to get reranker instance")
            return {
                "search_results": search_results,
                "rerank_metrics": {
                    "rerank_enabled": False,
                    "reason": "reranker_init_failed",
                    "original_count": len(search_results),
                    "final_count": len(search_results)
                }
            }
        
        try:
            # Perform reranking
            reranked_results, rerank_result = reranker.rerank(
                query=normalized_query,
                docs=search_results,
                min_score=reranker_config.min_score,
                top_k=reranker_config.top_k
            )
            
            # Gate before combine: check minimum documents threshold
            if len(reranked_results) < reranker_config.min_docs:
                # Not enough actionable documents - prepare "no actionable steps" response
                logger.info(
                    f"Insufficient actionable documents after rerank: "
                    f"{len(reranked_results)} < {reranker_config.min_docs}"
                )
                
                # Keep top raw results for "See sources" links
                top_raw_results = search_results[:reranker_config.top_k]
                
                # Log structured telemetry
                log_event(
                    stage="rerank",
                    event="insufficient_docs",
                    original_count=len(search_results),
                    reranked_count=len(reranked_results),
                    min_docs_threshold=reranker_config.min_docs,
                    top_scores=rerank_result.top_scores,
                    avg_score=rerank_result.avg_score,
                    device=rerank_result.device_used,
                    model_id=rerank_result.model_id,
                    took_ms=rerank_result.took_ms
                )
                
                return {
                    "search_results": top_raw_results,
                    "rerank_metrics": {
                        "rerank_enabled": True,
                        "insufficient_docs": True,
                        "reason": "below_min_docs_threshold",
                        "original_count": len(search_results),
                        "reranked_count": len(reranked_results),
                        "final_count": len(top_raw_results),
                        "min_docs_threshold": reranker_config.min_docs,
                        "top_scores": rerank_result.top_scores,
                        "avg_score": rerank_result.avg_score,
                        "device": rerank_result.device_used,
                        "model_id": rerank_result.model_id,
                        "took_ms": rerank_result.took_ms
                    },
                    "no_actionable_answer": True  # Signal to combine/answer nodes
                }
            
            # Success case - return reranked results
            logger.info(
                f"Reranked {len(search_results)} -> {len(reranked_results)} docs "
                f"(avg_score: {rerank_result.avg_score:.3f})"
            )
            
            # Log structured telemetry for successful reranking
            log_event(
                stage="rerank",
                event="success",
                original_count=len(search_results),
                final_count=len(reranked_results),
                dropped_count=rerank_result.dropped_count,
                top_scores=rerank_result.top_scores,
                avg_score=rerank_result.avg_score,
                device=rerank_result.device_used,
                model_id=rerank_result.model_id,
                took_ms=rerank_result.took_ms,
                min_score_threshold=reranker_config.min_score,
                top_k_limit=reranker_config.top_k
            )
            
            return {
                "search_results": reranked_results,
                "rerank_metrics": {
                    "rerank_enabled": True,
                    "successful": True,
                    "original_count": len(search_results),
                    "final_count": len(reranked_results),
                    "dropped_count": rerank_result.dropped_count,
                    "top_scores": rerank_result.top_scores,
                    "avg_score": rerank_result.avg_score,
                    "device": rerank_result.device_used,
                    "model_id": rerank_result.model_id,
                    "took_ms": rerank_result.took_ms,
                    "min_score_threshold": reranker_config.min_score,
                    "top_k_limit": reranker_config.top_k
                }
            }
            
        except Exception as e:
            # Log error and return original results
            logger.error(f"Reranking failed: {e}", exc_info=True)
            
            log_event(
                stage="rerank",
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                original_count=len(search_results)
            )
            
            return {
                "search_results": search_results,
                "rerank_metrics": {
                    "rerank_enabled": True,
                    "failed": True,
                    "error": str(e),
                    "original_count": len(search_results),
                    "final_count": len(search_results)
                }
            }


class RerankNodeV2(BaseNodeHandler):
    """Alternative rerank node implementation that preserves RetrievalResult structure.
    
    This version wraps the reranked documents back into a RetrievalResult for
    compatibility with downstream nodes that expect this structure.
    """
    
    def __init__(self):
        super().__init__("rerank_v2")
    
    async def execute(self, state: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute reranking preserving RetrievalResult structure."""
        # Get current query and retrieval result
        normalized_query = state.get("normalized_query", "")
        retrieval_result = state.get("retrieval_result")
        
        if not normalized_query or not retrieval_result:
            return {}
        
        if not isinstance(retrieval_result, RetrievalResult):
            logger.warning(f"Expected RetrievalResult, got {type(retrieval_result)}")
            return {}
        
        search_results = retrieval_result.results
        if not search_results:
            return {}
        
        # Get reranker configuration
        resources = get_resources()
        if not resources or not resources.settings.reranker.enabled:
            return {}
        
        reranker_config = resources.settings.reranker
        reranker = get_reranker(
            model_id=reranker_config.model_id,
            device=reranker_config.device,
            batch_size=reranker_config.batch_size
        )
        
        if not reranker:
            return {}
        
        try:
            # Perform reranking
            reranked_results, rerank_result = reranker.rerank(
                query=normalized_query,
                docs=search_results,
                min_score=reranker_config.min_score,
                top_k=reranker_config.top_k
            )
            
            # Create new RetrievalResult with reranked results
            new_retrieval_result = RetrievalResult(
                results=reranked_results,
                total_found=len(reranked_results),
                retrieval_time_ms=retrieval_result.retrieval_time_ms + rerank_result.took_ms,
                method=f"{retrieval_result.method}_reranked",
                diagnostics={
                    **retrieval_result.diagnostics,
                    "rerank_applied": True,
                    "rerank_device": rerank_result.device_used,
                    "rerank_model": rerank_result.model_id,
                    "rerank_took_ms": rerank_result.took_ms,
                    "rerank_dropped": rerank_result.dropped_count,
                    "rerank_avg_score": rerank_result.avg_score,
                    "rerank_top_scores": rerank_result.top_scores
                }
            )
            
            return {
                "retrieval_result": new_retrieval_result,
                "search_results": reranked_results  # Also update search_results for compatibility
            }
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return {}