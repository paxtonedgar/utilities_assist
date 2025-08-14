# src/services/embed.py
"""
Embedding service with comprehensive logging, error handling, and per-turn caching.
"""

import logging
import time
import hashlib
from typing import List, Optional, Any, Dict, Tuple

from src.telemetry.logger import stage, log_event

logger = logging.getLogger(__name__)

# Per-turn embedding cache to avoid recomputing embeddings for search rewrites
# Format: {query_hash: (embedding_vector, timestamp)}
_EMBEDDING_CACHE: Dict[str, Tuple[List[float], float]] = {}
_CACHE_TTL_SECONDS = 300  # 5 minutes TTL per turn


@stage("embed")
async def create_embeddings(
    texts: List[str], 
    embed_client: Any = None,
    model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """
    Create embeddings for a list of texts with comprehensive logging.
    
    Args:
        texts: List of text strings to embed
        embed_client: Embedding client instance
        model: Embedding model to use
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        log_event(
            stage="embed",
            event="skip",
            reason="empty_texts",
            texts_count=0
        )
        return []
    
    # Log embedding request
    log_event(
        stage="embed",
        event="start",
        texts_count=len(texts),
        total_chars=sum(len(text) for text in texts),
        model=model,
        avg_text_length=sum(len(text) for text in texts) / len(texts)
    )
    
    try:
        if not embed_client:
            raise ValueError("embed_client is required")
        
        # Create embeddings
        embeddings = []
        for i, text in enumerate(texts):
            try:
                # Use existing embedding creation logic
                from embedding_creation import create_single_embedding
                embedding = await create_single_embedding(text, embed_client, model)
                embeddings.append(embedding)
                
                # Log individual embedding success
                if i == 0:  # Log details for first embedding only
                    log_event(
                        stage="embed",
                        event="individual_success",
                        text_length=len(text),
                        vector_dims=len(embedding),
                        index=i
                    )
                
            except Exception as e:
                log_event(
                    stage="embed",
                    event="individual_error",
                    err=True,
                    error_type=type(e).__name__,
                    error_message=str(e)[:200],
                    index=i,
                    text_length=len(text)
                )
                # Continue with next text, don't fail entire batch
                continue
        
        # Log batch completion
        log_event(
            stage="embed",
            event="success",
            embeddings_created=len(embeddings),
            texts_processed=len(texts),
            success_rate=len(embeddings) / len(texts) if texts else 0,
            vector_dims=len(embeddings[0]) if embeddings else 0
        )
        
        return embeddings
        
    except Exception as e:
        # Log batch error
        log_event(
            stage="embed",
            event="error",
            err=True,
            error_type=type(e).__name__,
            error_message=str(e)[:200],
            texts_count=len(texts)
        )
        raise


def _get_cache_key(query: str, model: str) -> str:
    """Generate cache key from query and model."""
    content = f"{query}|{model}"
    return hashlib.md5(content.encode()).hexdigest()


def _clean_expired_cache():
    """Remove expired cache entries to prevent memory bloat."""
    current_time = time.time()
    expired_keys = [
        key for key, (_, timestamp) in _EMBEDDING_CACHE.items()
        if current_time - timestamp > _CACHE_TTL_SECONDS
    ]
    for key in expired_keys:
        del _EMBEDDING_CACHE[key]
    
    if expired_keys:
        logger.debug(f"Cleaned {len(expired_keys)} expired embedding cache entries")


@stage("embed")
async def create_query_embedding(
    query: str,
    embed_client: Any = None,
    model: str = "text-embedding-3-small",
    enable_cache: bool = True
) -> List[float]:
    """
    Create embedding for a single query string with per-turn caching.
    
    Avoids recomputing embeddings for search rewrites as per user requirements.
    
    Args:
        query: Query text to embed
        embed_client: Embedding client instance 
        model: Embedding model to use
        enable_cache: Whether to use caching (default True)
        
    Returns:
        Embedding vector
    """
    if not query or not query.strip():
        log_event(
            stage="embed",
            event="skip",
            reason="empty_query"
        )
        return []
    
    # QUERY EMBEDDING CACHING: Check cache first to avoid recomputing for search rewrites
    if enable_cache:
        cache_key = _get_cache_key(query.strip(), model)
        current_time = time.time()
        
        # Check if we have a valid cached embedding
        if cache_key in _EMBEDDING_CACHE:
            cached_embedding, timestamp = _EMBEDDING_CACHE[cache_key]
            if current_time - timestamp <= _CACHE_TTL_SECONDS:
                log_event(
                    stage="embed",
                    event="cache_hit",
                    query_length=len(query),
                    model=model,
                    cache_age_seconds=current_time - timestamp
                )
                return cached_embedding
            else:
                # Remove expired entry
                del _EMBEDDING_CACHE[cache_key]
        
        # Clean expired entries periodically
        if len(_EMBEDDING_CACHE) > 50:  # Clean when cache gets large
            _clean_expired_cache()
    
    # Cache miss - compute embedding
    log_event(
        stage="embed",
        event="cache_miss" if enable_cache else "cache_disabled",
        query_length=len(query),
        model=model
    )
    
    embeddings = await create_embeddings([query.strip()], embed_client, model)
    embedding = embeddings[0] if embeddings else []
    
    # Store in cache for future use
    if enable_cache and embedding:
        cache_key = _get_cache_key(query.strip(), model)
        _EMBEDDING_CACHE[cache_key] = (embedding, time.time())
        
        log_event(
            stage="embed", 
            event="cached",
            query_length=len(query),
            vector_dims=len(embedding),
            cache_size=len(_EMBEDDING_CACHE)
        )
    
    return embedding


def clear_embedding_cache():
    """Clear the embedding cache. Useful for testing or memory management."""
    global _EMBEDDING_CACHE
    cache_size = len(_EMBEDDING_CACHE)
    _EMBEDDING_CACHE.clear()
    logger.info(f"Cleared embedding cache: {cache_size} entries removed")


def get_cache_stats() -> Dict[str, Any]:
    """Get embedding cache statistics for monitoring."""
    current_time = time.time()
    valid_entries = sum(
        1 for _, timestamp in _EMBEDDING_CACHE.values()
        if current_time - timestamp <= _CACHE_TTL_SECONDS
    )
    
    return {
        "total_entries": len(_EMBEDDING_CACHE),
        "valid_entries": valid_entries,
        "expired_entries": len(_EMBEDDING_CACHE) - valid_entries,
        "cache_ttl_seconds": _CACHE_TTL_SECONDS
    }