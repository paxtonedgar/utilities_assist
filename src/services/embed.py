# src/services/embed.py
"""
Embedding service with comprehensive logging and error handling.
"""

import logging
from typing import List, Optional, Any

from src.telemetry.logger import stage, log_event

logger = logging.getLogger(__name__)


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


@stage("embed")
async def create_query_embedding(
    query: str,
    embed_client: Any = None,
    model: str = "text-embedding-3-small"
) -> List[float]:
    """
    Create embedding for a single query string.
    
    Args:
        query: Query text to embed
        embed_client: Embedding client instance 
        model: Embedding model to use
        
    Returns:
        Embedding vector
    """
    embeddings = await create_embeddings([query], embed_client, model)
    return embeddings[0] if embeddings else []