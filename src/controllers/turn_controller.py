"""Turn controller - orchestrates conversation turns."""

import logging
import time
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
from ..infra.config import Settings
from ..infra.clients import make_chat_client, make_embed_client, make_search_session
from ..infra.azure_auth import azure_token_provider
from ..services.models import TurnResult, IntentResult
from ..services.normalize import normalize_query
from ..services.intent import determine_intent
from ..services.retrieve import bm25_search, knn_search, rrf_fuse
from ..services.respond import build_context, generate_response, verify_answer, extract_source_chips

logger = logging.getLogger(__name__)


async def handle_turn(
    user_input: str,
    settings: Settings,
    chat_history: List[Dict[str, str]] = None,
    token_provider: Optional[callable] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """Handle a complete conversation turn with streaming response.
    
    Args:
        user_input: Raw user input
        settings: Application settings
        chat_history: Recent conversation history
        token_provider: Optional token provider for Azure (auto-detected for JPMC)
        
    Yields:
        Dict with turn progress updates and final result
    """
    start_time = time.time()
    turn_id = f"turn_{int(start_time)}"
    
    try:
        # Auto-detect Azure token provider for JPMC profile
        if settings.profile == "jpmc_azure" and token_provider is None:
            token_provider = azure_token_provider
            logger.info("Using Azure certificate authentication for JPMC profile")
        
        # Initialize clients
        chat_client = make_chat_client(settings.chat, token_provider)
        embed_client = make_embed_client(settings.embed, token_provider)
        search_session = make_search_session(settings.search)
        
        yield {"type": "status", "message": "Processing query...", "turn_id": turn_id}
        
        # Step 1: Normalize query
        normalized_query = normalize_query(user_input)
        logger.info(f"Normalized query: '{user_input}' -> '{normalized_query}'")
        
        yield {"type": "status", "message": "Understanding intent...", "turn_id": turn_id}
        
        # Step 2: Determine intent
        utilities_list = _get_utilities_list()  # Get from config or data
        intent = await determine_intent(normalized_query, chat_client, utilities_list)
        logger.info(f"Classified intent: {intent.intent} (confidence: {intent.confidence})")
        
        yield {
            "type": "intent", 
            "intent": intent.dict(),
            "turn_id": turn_id
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
            embed_client,
            search_session,
            settings
        )
        
        if retrieval_result:
            logger.info(f"Retrieved {len(retrieval_result.results)} results using {retrieval_result.method}")
        
        # Step 5: Build context
        context = build_context(retrieval_result.results if retrieval_result else [], intent)
        
        # Step 6: Generate response
        yield {"type": "status", "message": "Generating response...", "turn_id": turn_id}
        
        full_answer = ""
        async for response_chunk in generate_response(
            normalized_query,
            context,
            intent,
            chat_client,
            chat_history
        ):
            full_answer += response_chunk
            yield {
                "type": "response_chunk",
                "content": response_chunk,
                "turn_id": turn_id
            }
        
        # Step 7: Extract source chips
        source_chips = extract_source_chips(
            retrieval_result.results if retrieval_result else []
        )
        
        # Step 8: Verify answer quality
        verification = verify_answer(full_answer, context, normalized_query)
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
            "turn_id": turn_id
        }
        
        yield {
            "type": "complete", 
            "result": final_result.dict(),
            "turn_id": turn_id
        }
        
    except Exception as e:
        logger.error(f"Turn handling failed: {e}")
        
        error_result = TurnResult(
            answer=f"I encountered an error processing your request: {str(e)}",
            sources=[],
            intent=IntentResult(intent="error", confidence=0.0),
            response_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )
        
        yield {
            "type": "error",
            "result": error_result.dict(),
            "turn_id": turn_id
        }


async def _perform_retrieval(
    query: str,
    intent: IntentResult,
    embed_client: Any,
    search_session,
    settings: Settings,
    top_k: int = 10
):
    """Perform retrieval based on intent and settings."""
    
    try:
        # Determine index based on intent
        if intent.intent == "swagger":
            index_name = "khub-opensearch-swagger-index"
        else:
            index_name = "khub-test-md"  # Default confluence index
        
        # Choose retrieval strategy based on intent
        if intent.intent == "list":
            # For list queries, prefer BM25 for broader coverage
            return await bm25_search(
                query=query,
                session=search_session,
                index_name=index_name,
                top_k=top_k
            )
        
        elif settings.embed.provider and intent.confidence > 0.7:
            # For high-confidence queries, try hybrid search
            try:
                # Get query embedding
                embedding_response = await embed_client.embeddings.create(
                    model=settings.embed.model,
                    input=query
                )
                query_embedding = embedding_response.data[0].embedding
                
                # Perform both searches in parallel
                bm25_task = bm25_search(query, search_session, index_name, top_k)
                knn_task = knn_search(query_embedding, search_session, index_name, top_k)
                
                bm25_result, knn_result = await asyncio.gather(bm25_task, knn_task)
                
                # Fuse results
                return await rrf_fuse(bm25_result, knn_result, top_k=top_k)
                
            except Exception as e:
                logger.warning(f"Hybrid search failed, falling back to BM25: {e}")
                return await bm25_search(query, search_session, index_name, top_k)
        
        else:
            # Fallback to BM25
            return await bm25_search(query, search_session, index_name, top_k)
            
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