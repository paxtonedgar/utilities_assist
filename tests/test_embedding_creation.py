#!/usr/bin/env python3
"""
Tests for embedding creation with error scenarios and retry logic.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, Mock
from typing import List

# Add project to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding_creation import (
    create_embeddings_with_retry,
    create_single_embedding,
    EmbeddingError,
    assert_dims,
    cosine_similarity,
    validate_embedding_batch,
    _normalize_text
)


class TestEmbeddingCreation:
    """Test embedding creation functions."""
    
    @pytest.fixture
    def mock_embed_client(self):
        """Create mock embedding client."""
        client = Mock()
        client.embeddings = AsyncMock()
        return client
    
    @pytest.fixture 
    def valid_embedding_response(self):
        """Mock valid embedding API response."""
        response = Mock()
        response.data = [
            Mock(embedding=[0.1] * 1024),
            Mock(embedding=[0.2] * 1024),
            Mock(embedding=[0.3] * 1024)
        ]
        return response
    
    def test_normalize_text(self):
        """Test text normalization."""
        # Normal text
        assert _normalize_text("  hello   world  \n") == "hello world"
        
        # Long text truncation
        long_text = "x" * 10000
        normalized = _normalize_text(long_text, max_length=100)
        assert len(normalized) == 100
        
        # Empty text
        assert _normalize_text("") == ""
        assert _normalize_text("   \n\t  ") == ""
    
    def test_assert_dims_valid(self):
        """Test dimension validation with valid vectors."""
        vecs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        assert_dims(vecs, 3)  # Should not raise
    
    def test_assert_dims_invalid(self):
        """Test dimension validation with invalid vectors."""
        vecs = [[1.0, 2.0], [4.0, 5.0, 6.0]]  # Mismatched dimensions
        
        with pytest.raises(EmbeddingError, match="Dimension mismatch at index 0"):
            assert_dims(vecs, 3)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        vec3 = [1.0, 0.0, 0.0]
        
        # Orthogonal vectors
        assert abs(cosine_similarity(vec1, vec2)) < 1e-6
        
        # Identical vectors
        assert abs(cosine_similarity(vec1, vec3) - 1.0) < 1e-6
        
        # Different dimensions should raise error
        with pytest.raises(EmbeddingError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
    
    def test_validate_embedding_batch(self):
        """Test embedding batch validation."""
        # Valid batch
        valid_embeddings = [[0.5] * 3, [0.7] * 3]
        is_valid, issues = validate_embedding_batch(valid_embeddings, 3)
        assert is_valid
        assert len(issues) == 0
        
        # Invalid dimensions
        invalid_embeddings = [[0.5] * 2, [0.7] * 3]
        is_valid, issues = validate_embedding_batch(invalid_embeddings, 3)
        assert not is_valid
        assert "wrong dimensions" in issues[0]
        
        # Zero magnitude (invalid)
        zero_embeddings = [[0.0] * 3]
        is_valid, issues = validate_embedding_batch(zero_embeddings, 3)
        assert not is_valid
        assert "magnitude too small" in issues[0]
    
    @pytest.mark.asyncio
    async def test_create_embeddings_success(self, mock_embed_client, valid_embedding_response):
        """Test successful embedding creation."""
        mock_embed_client.embeddings.create.return_value = valid_embedding_response
        
        texts = ["hello", "world", "test"]
        embeddings = await create_embeddings_with_retry(
            embed_client=mock_embed_client,
            texts=texts,
            model="text-embedding-ada-002",
            expected_dims=1024
        )
        
        assert len(embeddings) == 3
        assert all(len(emb) == 1024 for emb in embeddings)
        mock_embed_client.embeddings.create.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_embeddings_wrong_dimensions(self, mock_embed_client):
        """Test embedding creation with wrong dimensions."""
        # Mock response with wrong dimensions
        response = Mock()
        response.data = [Mock(embedding=[0.1] * 512)]  # Wrong size
        mock_embed_client.embeddings.create.return_value = response
        
        with pytest.raises(EmbeddingError, match="Dimension mismatch"):
            await create_embeddings_with_retry(
                embed_client=mock_embed_client,
                texts=["test"],
                model="text-embedding-ada-002",
                expected_dims=1024
            )
    
    @pytest.mark.asyncio
    async def test_create_embeddings_api_failure(self, mock_embed_client):
        """Test embedding creation with API failure."""
        mock_embed_client.embeddings.create.side_effect = Exception("API Error")
        
        with pytest.raises(EmbeddingError, match="API call failed"):
            await create_embeddings_with_retry(
                embed_client=mock_embed_client,
                texts=["test"],
                model="text-embedding-ada-002",
                expected_dims=1024
            )
    
    @pytest.mark.asyncio
    async def test_create_embeddings_batch_processing(self, mock_embed_client):
        """Test batching with multiple API calls."""
        # Create responses for multiple batches
        def mock_create(model, input):
            response = Mock()
            response.data = [Mock(embedding=[0.1] * 1024) for _ in input]
            return response
        
        mock_embed_client.embeddings.create.side_effect = mock_create
        
        # Test with 5 texts and batch_size=2
        texts = ["text1", "text2", "text3", "text4", "text5"]
        embeddings = await create_embeddings_with_retry(
            embed_client=mock_embed_client,
            texts=texts,
            model="text-embedding-ada-002",
            expected_dims=1024,
            batch_size=2
        )
        
        assert len(embeddings) == 5
        # Should have made 3 API calls (2+2+1)
        assert mock_embed_client.embeddings.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_create_single_embedding(self, mock_embed_client):
        """Test single embedding creation."""
        response = Mock()
        response.data = [Mock(embedding=[0.5] * 1024)]
        mock_embed_client.embeddings.create.return_value = response
        
        embedding = await create_single_embedding(
            embed_client=mock_embed_client,
            text="test text",
            model="text-embedding-ada-002",
            expected_dims=1024
        )
        
        assert len(embedding) == 1024
        assert embedding == [0.5] * 1024
    
    @pytest.mark.asyncio
    async def test_create_embeddings_empty_texts(self, mock_embed_client):
        """Test embedding creation with empty texts."""
        embeddings = await create_embeddings_with_retry(
            embed_client=mock_embed_client,
            texts=[],
            model="text-embedding-ada-002",
            expected_dims=1024
        )
        
        assert embeddings == []
        mock_embed_client.embeddings.create.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_create_embeddings_none_client(self):
        """Test with None embed client."""
        with pytest.raises(EmbeddingError, match="Embed client is None"):
            await create_embeddings_with_retry(
                embed_client=None,
                texts=["test"],
                model="text-embedding-ada-002",
                expected_dims=1024
            )
    
    @pytest.mark.asyncio
    async def test_create_embeddings_invalid_expected_dims(self, mock_embed_client):
        """Test with invalid expected dimensions."""
        with pytest.raises(EmbeddingError, match="Invalid expected_dims"):
            await create_embeddings_with_retry(
                embed_client=mock_embed_client,
                texts=["test"],
                model="text-embedding-ada-002",
                expected_dims=0
            )
    
    @pytest.mark.asyncio 
    async def test_create_embeddings_with_empty_strings(self, mock_embed_client):
        """Test handling of empty/whitespace strings."""
        response = Mock()
        response.data = [
            Mock(embedding=[0.1] * 1024),
            Mock(embedding=[0.2] * 1024),
            Mock(embedding=[0.3] * 1024)
        ]
        mock_embed_client.embeddings.create.return_value = response
        
        texts = ["valid text", "   ", ""]
        embeddings = await create_embeddings_with_retry(
            embed_client=mock_embed_client,
            texts=texts,
            model="text-embedding-ada-002",
            expected_dims=1024
        )
        
        # Should get 3 embeddings (valid text + 2 placeholders for empty strings)
        assert len(embeddings) == 3
        
        # Check that empty strings were replaced with placeholder
        call_args = mock_embed_client.embeddings.create.call_args[1]
        input_texts = call_args['input']
        assert "[empty]" in input_texts
        assert input_texts.count("[empty]") == 2


class TestRetryLogic:
    """Test retry logic and failure scenarios."""
    
    @pytest.mark.asyncio
    async def test_retry_logic(self):
        """Test that retry logic works with exponential backoff."""
        mock_client = Mock()
        mock_client.embeddings = AsyncMock()
        
        # First two calls fail, third succeeds
        mock_client.embeddings.create.side_effect = [
            Exception("Temporary failure"),
            Exception("Another failure"),
            Mock(data=[Mock(embedding=[0.1] * 1024)])
        ]
        
        embeddings = await create_embeddings_with_retry(
            embed_client=mock_client,
            texts=["test"],
            model="text-embedding-ada-002", 
            expected_dims=1024
        )
        
        assert len(embeddings) == 1
        assert mock_client.embeddings.create.call_count == 3
    
    @pytest.mark.asyncio
    async def test_retry_exhaustion(self):
        """Test that retries are exhausted after max attempts."""
        mock_client = Mock()
        mock_client.embeddings = AsyncMock()
        mock_client.embeddings.create.side_effect = Exception("Persistent failure")
        
        with pytest.raises(EmbeddingError):
            await create_embeddings_with_retry(
                embed_client=mock_client,
                texts=["test"],
                model="text-embedding-ada-002",
                expected_dims=1024
            )
        
        # Should have tried 3 times (initial + 2 retries)
        assert mock_client.embeddings.create.call_count == 3


# Mock client that returns wrong dimensions for integration testing
class WrongDimensionClient:
    def __init__(self, wrong_dims=512):
        self.wrong_dims = wrong_dims
        self.embeddings = AsyncMock()
        self.embeddings.create = self._create_wrong_dims
    
    async def _create_wrong_dims(self, model, input):
        response = Mock()
        response.data = [Mock(embedding=[0.1] * self.wrong_dims) for _ in input]
        return response


# Slow client for testing timeouts and retries
class SlowClient:
    def __init__(self, delay_seconds=0.1, fail_count=2):
        self.delay_seconds = delay_seconds
        self.fail_count = fail_count
        self.call_count = 0
        self.embeddings = AsyncMock()
        self.embeddings.create = self._slow_create
    
    async def _slow_create(self, model, input):
        self.call_count += 1
        await asyncio.sleep(self.delay_seconds)
        
        if self.call_count <= self.fail_count:
            raise Exception(f"Slow failure #{self.call_count}")
        
        response = Mock()
        response.data = [Mock(embedding=[0.1] * 1024) for _ in input]
        return response


@pytest.mark.asyncio
async def test_wrong_dimension_integration():
    """Integration test with wrong dimension client."""
    client = WrongDimensionClient(wrong_dims=768)
    
    with pytest.raises(EmbeddingError, match="Dimension mismatch"):
        await create_embeddings_with_retry(
            embed_client=client,
            texts=["test"],
            model="text-embedding-ada-002",
            expected_dims=1024
        )


@pytest.mark.asyncio 
async def test_slow_client_integration():
    """Integration test with slow client that eventually succeeds."""
    client = SlowClient(delay_seconds=0.01, fail_count=2)
    
    embeddings = await create_embeddings_with_retry(
        embed_client=client,
        texts=["test"],
        model="text-embedding-ada-002", 
        expected_dims=1024
    )
    
    assert len(embeddings) == 1
    assert client.call_count == 3  # Failed 2 times, succeeded on 3rd


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])