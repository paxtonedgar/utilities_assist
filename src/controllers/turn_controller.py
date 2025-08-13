"""Turn controller - orchestrates conversation turns."""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from infra.config import Settings
from infra.resource_manager import get_resources, RAGResources
from services.models import TurnResult, IntentResult
from services.normalize import normalize_query
from services.intent import determine_intent
import services.retrieve as retrieve
from services.respond import build_context, generate_response, verify_answer, extract_source_chips
from embedding_creation import create_single_embedding, EmbeddingError
from infra.telemetry import (
    generate_request_id, log_overall_stage, log_normalize_stage, 
    log_intent_stage, log_embedding_stage, time_stage,
    log_retrieve_bm25_stage, log_retrieve_knn_stage, log_fuse_stage,
    log_llm_stage, log_verify_stage
)

logger = logging.getLogger(__name__)


async def handle_turn(
    user_input: str,
    resources: Optional[RAGResources] = None,
    chat_history: List[Dict[str, str]] = None,
    use_mock_corpus: bool = False
) -> AsyncGenerator[Dict[str, Any], None]:
    """Handle a complete conversation turn with streaming response.
    
    Phase 1 Performance Optimization: Uses shared resources instead of creating
    clients per turn, eliminating 25-50% response time overhead.
    
    Args:
        user_input: Raw user input
        resources: Pre-configured resource container (auto-fetched if None)
        chat_history: Recent conversation history
        use_mock_corpus: If True, use confluence_mock index instead of confluence_current
        
    Yields:
        Dict with turn progress updates and final result
    """
    start_time = time.time()
    turn_id = f"turn_{int(start_time)}"
    req_id = generate_request_id()
    
    try:
        # Get shared resources (eliminates client creation overhead)
        if resources is None:
            resources = get_resources()
            if resources is None:
                raise RuntimeError("Resources not initialized. Call initialize_resources() at startup.")
        
        # Use pre-configured clients (no creation overhead)
        settings = resources.settings
        chat_client = resources.chat_client
        embed_client = resources.embed_client
        search_client = resources.search_client
        
        logger.info(f"Using shared resources (age: {resources.get_age_seconds():.1f}s)")
        
        yield {"type": "status", "message": "Processing query...", "turn_id": turn_id, "req_id": req_id}
        
        # Step 1: Normalize query
        normalize_start = time.perf_counter()
        normalized_query = normalize_query(user_input)
        normalize_time = (time.perf_counter() - normalize_start) * 1000
        log_normalize_stage(req_id, user_input, normalized_query, normalize_time)
        logger.info(f"Normalized query: '{user_input}' -> '{normalized_query}'")
        
        yield {"type": "status", "message": "Understanding intent...", "turn_id": turn_id}
        
        # Step 2: Determine intent
        intent_start = time.perf_counter()
        utilities_list = _get_utilities_list()  # Get from config or data
        intent_error = None
        try:
            intent = await determine_intent(normalized_query, chat_client, utilities_list, settings.chat.model)
        except Exception as e:
            intent_error = repr(e)
            intent = IntentResult(intent="error", confidence=0.0)
            raise
        finally:
            intent_time = (time.perf_counter() - intent_start) * 1000
            log_intent_stage(req_id, intent.intent, intent.confidence, intent_time, intent_error)
        
        logger.info(f"Classified intent: {intent.intent} (confidence: {intent.confidence})")
        
        yield {
            "type": "intent", 
            "intent": intent.dict(),
            "turn_id": turn_id,
            "req_id": req_id
        }
        
        # Step 3: Handle restart intent specially
        if intent.intent == "restart":
            yield {
                "type": "response_chunk",
                "content": "Context has been cleared. How can I help you today?",
                "turn_id": turn_id
            }
            
            final_result = TurnResult(
                answer="Context has been cleared. How can I help you today?",
                sources=[],
                intent=intent,
                response_time_ms=(time.time() - start_time) * 1000
            )
            
            yield {
                "type": "complete",
                "result": final_result.dict(),
                "turn_id": turn_id
            }
            return
        
        # Step 4: Retrieve relevant information
        yield {"type": "status", "message": "Searching for relevant information...", "turn_id": turn_id}
        
        retrieval_result = await _perform_retrieval(
            normalized_query, 
            intent,
            resources,  # Pass entire resource container
            req_id,
            use_mock_corpus=use_mock_corpus
        )
        
        if retrieval_result:
            logger.info(f"Retrieved {len(retrieval_result.results)} results using {retrieval_result.method}")
            
            # Inform user if we fell back to BM25 due to embedding issues
            if retrieval_result.method == "bm25" and settings.embed.provider and intent.confidence > 0.7:
                yield {
                    "type": "status", 
                    "message": "Using text search (embedding service unavailable)...", 
                    "turn_id": turn_id
                }
        
        # Step 5: Build context
        context = build_context(retrieval_result.results if retrieval_result else [], intent)
        
        # Step 6: Generate response
        yield {"type": "status", "message": "Generating response...", "turn_id": turn_id}
        
        llm_start = time.perf_counter()
        full_answer = ""
        tokens_in = len(normalized_query.split()) + len(context.split()) if context else 0  # Rough estimate
        llm_error = None
        
        # Get LLM parameters from cached config (eliminates disk I/O overhead)
        temperature = float(resources.get_config_param('temperature', 0.2))
        max_tokens = int(resources.get_config_param('max_tokens_2k', 1500))
        
        try:
            async for response_chunk in generate_response(
                normalized_query,
                context,
                intent,
                chat_client,
                chat_history,
                settings.chat.model,  # Pass the deployment name for Azure OpenAI
                temperature,
                max_tokens
            ):
                full_answer += response_chunk
                yield {
                    "type": "response_chunk",
                    "content": response_chunk,
                    "turn_id": turn_id
                }
        except Exception as e:
            llm_error = repr(e)
            raise
        finally:
            llm_time = (time.perf_counter() - llm_start) * 1000
            tokens_out = len(full_answer.split()) if full_answer else 0  # Rough estimate
            log_llm_stage(req_id, settings.chat.model, tokens_in, tokens_out, llm_time, llm_error, context=context)
        
        # Step 7: Extract source chips
        source_chips = extract_source_chips(
            retrieval_result.results if retrieval_result else []
        )
        
        # Step 8: Verify answer quality
        verify_start = time.perf_counter()
        verification = verify_answer(full_answer, context, normalized_query)
        verify_time = (time.perf_counter() - verify_start) * 1000
        
        # Log verification stage
        log_verify_stage(
            req_id=req_id,
            verdict=verification.get('verdict', 'unknown'),
            unmatched_claims_count=verification.get('unmatched_claims_count', 0),
            ms=verify_time,
            confidence_score=verification.get('confidence_score')
        )
        
        logger.info(f"Answer verification: confidence={verification['confidence_score']:.2f}")
        
        # Step 9: Build final result
        final_result = TurnResult(
            answer=full_answer,
            sources=source_chips,
            intent=intent,
            retrieval=retrieval_result,
            response_time_ms=(time.time() - start_time) * 1000
        )
        
        yield {
            "type": "verification",
            "metrics": verification,
            "turn_id": turn_id,
            "req_id": req_id
        }
        
        # Log overall success
        total_latency = (time.time() - start_time) * 1000
        log_overall_stage(
            req_id=req_id,
            latency_ms=total_latency,
            success=True,
            result_count=len(retrieval_result.results) if retrieval_result else 0,
            method=retrieval_result.method if retrieval_result else "none"
        )
        
        yield {
            "type": "complete", 
            "result": final_result.dict(),
            "turn_id": turn_id,
            "req_id": req_id
        }
        
    except Exception as e:
        logger.error(f"Turn handling failed: {e}")
        
        # Log overall failure
        total_latency = (time.time() - start_time) * 1000
        log_overall_stage(
            req_id=req_id,
            latency_ms=total_latency,
            success=False,
            error=repr(e)
        )
        
        error_result = TurnResult(
            answer=f"I encountered an error processing your request: {str(e)}",
            sources=[],
            intent=IntentResult(intent="error", confidence=0.0),
            response_time_ms=total_latency,
            error=str(e)
        )
        
        yield {
            "type": "error",
            "result": error_result.dict(),
            "turn_id": turn_id,
            "req_id": req_id
        }


async def _perform_retrieval(
    query: str,
    intent: IntentResult,
    resources: RAGResources,
    req_id: str = None,
    use_mock_corpus: bool = False,
    top_k: int = 10
):
    """Perform retrieval using shared resources - no client creation overhead."""
    
    try:
        # Extract shared resources (no overhead)
        settings = resources.settings
        embed_client = resources.embed_client
        search_client = resources.search_client
        
        # Determine index based on intent and corpus selection
        if use_mock_corpus:
            index_name = "confluence_mock"  # Use mock corpus for evaluation
        elif intent.intent == "swagger":
            index_name = "khub-opensearch-swagger-index"
        else:
            # Use the index from cached configuration
            index_name = settings.search.index_alias
        
        # Build filters based on intent and user context
        filters = {}
        if intent.intent == "confluence":
            filters["content_type"] = "confluence"
        elif intent.intent == "swagger":
            filters["content_type"] = "api_spec"
        
        # Add ACL filter (would be determined by user session in production)
        # filters["acl_hash"] = user_acl_hash
        
        # Choose retrieval strategy based on intent
        if intent.intent == "list":
            # For list queries, prefer BM25 for broader coverage
            bm25_start = time.perf_counter()
            try:
                result = await retrieve.bm25_search(
                    query=query,
                    search_client=search_client,
                    index_name=index_name,
                    filters=filters,
                    top_k=top_k,
                    time_decay_days=120  # 4-month half-life for recency
                )
                bm25_time = (time.perf_counter() - bm25_start) * 1000
                top_ids = [r.doc_id for r in result.results[:5]] if result and result.results else []
                log_retrieve_bm25_stage(req_id, top_k, len(result.results) if result else 0, bm25_time, top_ids)
                return result
            except Exception as e:
                bm25_time = (time.perf_counter() - bm25_start) * 1000
                log_retrieve_bm25_stage(req_id, top_k, 0, bm25_time, error=repr(e))
                raise
        
        elif embed_client and intent.confidence > 0.7:  # Check client availability directly
            # For high-confidence queries, use enhanced RRF search with MMR diversification
            try:
                # Get query embedding with proper validation
                embedding_start = time.perf_counter()
                expected_dims = 1536  # Standard for text-embedding-ada-002 and text-embedding-3-small
                try:
                    query_embedding = await create_single_embedding(
                        embed_client=embed_client,
                        text=query,
                        model=settings.embed.model,
                        expected_dims=expected_dims
                    )
                    embedding_time = (time.perf_counter() - embedding_start) * 1000
                    log_embedding_stage(req_id, 1, 1, expected_dims, embedding_time)
                except Exception as e:
                    embedding_time = (time.perf_counter() - embedding_start) * 1000
                    log_embedding_stage(req_id, 1, 1, expected_dims, embedding_time, repr(e))
                    raise
                
                # Use enhanced RRF search with MMR diversification
                hybrid_start = time.perf_counter()
                try:
                    result, diagnostics = await retrieve.enhanced_rrf_search(
                        query=query,
                        query_embedding=query_embedding,
                        search_client=search_client,
                        index_name=index_name,
                        filters=filters,
                        top_k=top_k,
                        rrf_k=60,
                        use_mmr=True,
                        lambda_param=0.75  # 75% relevance, 25% diversity
                    )
                    hybrid_time = (time.perf_counter() - hybrid_start) * 1000
                    
                    # Log enhanced diagnostics
                    logger.info(f"Enhanced RRF: BM25({diagnostics['bm25_count']}) + KNN({diagnostics['knn_count']}) → RRF({diagnostics['rrf_count']}) → Final({len(result.results)})")
                    logger.info(f"MMR applied: {diagnostics['mmr_applied']}, Generic penalties: {diagnostics['generic_penalties']}")
                    
                    # Log fusion stage with enhanced metrics
                    log_fuse_stage(
                        req_id=req_id,
                        method="enhanced_rrf_mmr",
                        k_final=top_k,
                        bm25_count=diagnostics['bm25_count'],
                        knn_count=diagnostics['knn_count'],
                        ms=hybrid_time
                    )
                    
                    # Diagnostics now included in RetrievalResult object
                    return result
                    
                except Exception as e:
                    hybrid_time = (time.perf_counter() - hybrid_start) * 1000
                    log_fuse_stage(
                        req_id=req_id,
                        method="enhanced_rrf_mmr",
                        k_final=top_k,
                        bm25_count=0,
                        knn_count=0,
                        ms=hybrid_time,
                        error=repr(e)
                    )
                    raise
                
            except EmbeddingError as e:
                logger.error(f"Embedding service error: {e}")
                # Log telemetry for embedding misconfiguration
                logger.warning(f"Falling back to BM25-only due to embedding error: {e}")
                # Fallback BM25 with timing
                bm25_start = time.perf_counter()
                try:
                    result = await retrieve.bm25_search(query, search_client, index_name, filters, top_k)
                    bm25_time = (time.perf_counter() - bm25_start) * 1000
                    top_ids = [r.doc_id for r in result.results[:5]] if result and result.results else []
                    log_retrieve_bm25_stage(req_id, top_k, len(result.results) if result else 0, bm25_time, top_ids)
                    return result
                except Exception as fallback_e:
                    bm25_time = (time.perf_counter() - bm25_start) * 1000
                    log_retrieve_bm25_stage(req_id, top_k, 0, bm25_time, error=repr(fallback_e))
                    raise
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to BM25: {e}")
                # Fallback BM25 with timing
                bm25_start = time.perf_counter()
                try:
                    result = await retrieve.bm25_search(query, search_client, index_name, filters, top_k)
                    bm25_time = (time.perf_counter() - bm25_start) * 1000
                    top_ids = [r.doc_id for r in result.results[:5]] if result and result.results else []
                    log_retrieve_bm25_stage(req_id, top_k, len(result.results) if result else 0, bm25_time, top_ids)
                    return result
                except Exception as fallback_e:
                    bm25_time = (time.perf_counter() - bm25_start) * 1000
                    log_retrieve_bm25_stage(req_id, top_k, 0, bm25_time, error=repr(fallback_e))
                    raise
        
        else:
            # Fallback to BM25 with timing
            bm25_start = time.perf_counter()
            try:
                result = await retrieve.bm25_search(query, search_client, index_name, filters, top_k)
                bm25_time = (time.perf_counter() - bm25_start) * 1000
                top_ids = [r.doc_id for r in result.results[:5]] if result and result.results else []
                log_retrieve_bm25_stage(req_id, top_k, len(result.results) if result else 0, bm25_time, top_ids)
                return result
            except Exception as e:
                bm25_time = (time.perf_counter() - bm25_start) * 1000
                log_retrieve_bm25_stage(req_id, top_k, 0, bm25_time, error=repr(e))
                raise
            
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return None


def _get_utilities_list() -> List[str]:
    """Get list of known utilities for intent classification."""
    # This could be loaded from config or data files
    return [
        "Customer Summary Utility",
        "Enhanced Transaction Utility", 
        "Account Utility",
        "Customer Interaction Utility",
        "Digital Events",
        "Product Catalog Utility",
        "Global Customer Platform"
    ]