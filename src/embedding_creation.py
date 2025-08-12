# Selection between local OpenAI vs JPMC Azure is driven by cfg.embed.provider and env.
# No code changes required to switch.

"""
Pure embedding creation with request-scoped clients and strict validation.

Features:
- Pure function approach with no singletons
- Request-scoped clients passed from caller
- Strict dimension validation with detailed errors
- Retry logic with exponential backoff
- Batching with configurable chunk sizes
- Text normalization and length limits
- Comprehensive error handling for chat integration
"""

import asyncio
import logging
import re
from typing import Iterable, List, Tuple, Any, Optional

from tenacity import retry, wait_exponential_jitter, stop_after_attempt, before_sleep_log

logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    """Raised when embedding creation fails or validation errors occur."""
    pass


def assert_dims(vecs: List[List[float]], expected: int) -> None:
    """
    Assert all vectors have expected dimensions.
    
    Args:
        vecs: List of embedding vectors
        expected: Expected dimension count
        
    Raises:
        EmbeddingError: If any vector has wrong dimensions
    """
    for i, v in enumerate(vecs):
        if len(v) != expected:
            raise EmbeddingError(
                f"Dimension mismatch at index {i}: got {len(v)}, expected {expected}"
            )


def _normalize_text(text: str, max_length: int = 8192) -> str:
    """
    Normalize text for embedding creation.
    
    Args:
        text: Raw input text
        max_length: Maximum allowed character length
        
    Returns:
        Normalized text, possibly truncated
    """
    # Strip whitespace and normalize line endings
    normalized = re.sub(r'\s+', ' ', text.strip())
    
    # Truncate if too long and log warning
    if len(normalized) > max_length:
        logger.warning(f"Text truncated from {len(normalized)} to {max_length} chars")
        normalized = normalized[:max_length]
    
    return normalized


async def _create_batch_embeddings(
    embed_client: Any,
    texts: List[str], 
    model: str
) -> List[List[float]]:
    """
    Create embeddings for a batch of texts.
    
    Args:
        embed_client: OpenAI/Azure client instance
        texts: List of text strings to embed
        model: Embedding model name
        
    Returns:
        List of embedding vectors
        
    Raises:
        EmbeddingError: If API call fails or returns invalid data
    """
    try:
        response = embed_client.embeddings.create(
            model=model,
            input=texts
        )
        
        # Extract embeddings from response
        embeddings = []
        for data_item in response.data:
            embedding = data_item.embedding
            if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                raise EmbeddingError(f"Invalid embedding format received from API")
            embeddings.append(embedding)
        
        if len(embeddings) != len(texts):
            raise EmbeddingError(
                f"Embedding count mismatch: sent {len(texts)} texts, got {len(embeddings)} embeddings"
            )
        
        return embeddings
        
    except Exception as e:
        if isinstance(e, EmbeddingError):
            raise
        raise EmbeddingError(f"API call failed: {str(e)}") from e


@retry(
    wait=wait_exponential_jitter(initial=0.2, max=1.2, jitter=0.1),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True
)
async def create_embeddings_with_retry(
    embed_client: Any, 
    texts: Iterable[str], 
    model: str, 
    expected_dims: int,
    batch_size: int = 64,
    max_text_length: int = 8192
) -> List[List[float]]:
    """
    Create embeddings for texts with retry logic and validation.
    
    Args:
        embed_client: OpenAI/Azure client instance (request-scoped)
        texts: Iterable of text strings to embed
        model: Embedding model name 
        expected_dims: Expected embedding dimensions for validation
        batch_size: Number of texts per API batch (default: 64)
        max_text_length: Maximum text length before truncation
        
    Returns:
        List of embedding vectors, one per input text
        
    Raises:
        EmbeddingError: If embedding creation fails, API errors, or dimension validation fails
    """
    if not embed_client:
        raise EmbeddingError("Embed client is None - client factory may have failed")
    
    if expected_dims <= 0:
        raise EmbeddingError(f"Invalid expected_dims: {expected_dims}")
    
    # Convert to list and normalize texts
    text_list = []
    for text in texts:
        if not text or not text.strip():
            logger.warning("Empty text found in embedding batch, using placeholder")
            normalized = "[empty]"
        else:
            normalized = _normalize_text(text, max_text_length)
        text_list.append(normalized)
    
    if not text_list:
        logger.info("No texts to embed, returning empty list")
        return []
    
    logger.info(f"Creating embeddings for {len(text_list)} texts using model {model}")
    
    # Process in batches
    all_embeddings = []
    
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        batch_start = i + 1
        batch_end = min(i + batch_size, len(text_list))
        
        logger.debug(f"Processing batch {batch_start}-{batch_end} of {len(text_list)}")
        
        try:
            batch_embeddings = await _create_batch_embeddings(embed_client, batch, model)
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            error_msg = f"Batch {batch_start}-{batch_end} failed: {str(e)}"
            logger.error(error_msg)
            raise EmbeddingError(error_msg) from e
    
    # Validate dimensions for all embeddings
    try:
        assert_dims(all_embeddings, expected_dims)
    except EmbeddingError as e:
        logger.error(f"Dimension validation failed: {e}")
        raise
    
    logger.info(f"Successfully created {len(all_embeddings)} embeddings with {expected_dims} dimensions")
    return all_embeddings


async def create_single_embedding(
    embed_client: Any,
    text: str,
    model: str,
    expected_dims: int,
    max_text_length: int = 8192
) -> List[float]:
    """
    Create embedding for a single text with validation.
    
    Args:
        embed_client: OpenAI/Azure client instance
        text: Text string to embed
        model: Embedding model name
        expected_dims: Expected embedding dimensions
        max_text_length: Maximum text length before truncation
        
    Returns:
        Single embedding vector
        
    Raises:
        EmbeddingError: If embedding creation fails or validation fails
    """
    embeddings = await create_embeddings_with_retry(
        embed_client=embed_client,
        texts=[text],
        model=model,
        expected_dims=expected_dims,
        batch_size=1,
        max_text_length=max_text_length
    )
    
    return embeddings[0]


# Utility functions for common embedding operations

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First embedding vector
        vec2: Second embedding vector
        
    Returns:
        Cosine similarity score between -1 and 1
        
    Raises:
        EmbeddingError: If vectors have different dimensions
    """
    if len(vec1) != len(vec2):
        raise EmbeddingError(f"Vector dimension mismatch: {len(vec1)} vs {len(vec2)}")
    
    # Calculate dot product
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    
    # Calculate magnitudes
    magnitude1 = sum(a * a for a in vec1) ** 0.5
    magnitude2 = sum(b * b for b in vec2) ** 0.5
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def validate_embedding_batch(
    embeddings: List[List[float]], 
    expected_dims: int,
    min_magnitude: float = 0.1,
    max_magnitude: float = 2.0
) -> Tuple[bool, List[str]]:
    """
    Validate a batch of embeddings for quality issues.
    
    Args:
        embeddings: List of embedding vectors
        expected_dims: Expected dimension count
        min_magnitude: Minimum acceptable vector magnitude
        max_magnitude: Maximum acceptable vector magnitude
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    for i, embedding in enumerate(embeddings):
        # Check dimensions
        if len(embedding) != expected_dims:
            issues.append(f"Vector {i}: wrong dimensions ({len(embedding)} vs {expected_dims})")
        
        # Check for NaN or infinite values
        if not all(isinstance(x, (int, float)) and not (x != x or abs(x) == float('inf')) for x in embedding):
            issues.append(f"Vector {i}: contains NaN or infinite values")
        
        # Check magnitude
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude < min_magnitude:
            issues.append(f"Vector {i}: magnitude too small ({magnitude:.4f})")
        elif magnitude > max_magnitude:
            issues.append(f"Vector {i}: magnitude too large ({magnitude:.4f})")
    
    return len(issues) == 0, issues