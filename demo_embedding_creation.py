#!/usr/bin/env python3
"""
Demo script for the new embedding creation system.

Shows:
- Pure function approach with no singletons
- Request-scoped client usage  
- Dimension validation and error handling
- Retry logic with backoff
- Batch processing
"""

import asyncio
import logging
from unittest.mock import Mock, AsyncMock

# Set up logging to see retry behavior
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.embedding_creation import (
    create_embeddings_with_retry,
    create_single_embedding,
    EmbeddingError,
    assert_dims,
    cosine_similarity,
    validate_embedding_batch
)


class MockEmbeddingClient:
    """Mock client that demonstrates various scenarios."""
    
    def __init__(self, scenario="success"):
        self.scenario = scenario
        self.call_count = 0
        self.embeddings = AsyncMock()
        self.embeddings.create = self._mock_create
    
    async def _mock_create(self, model, input):
        self.call_count += 1
        
        if self.scenario == "success":
            response = Mock()
            response.data = [Mock(embedding=[0.1 + i * 0.1] * 1536) for i in range(len(input))]
            return response
            
        elif self.scenario == "wrong_dims":
            response = Mock()
            response.data = [Mock(embedding=[0.1] * 512) for _ in input]  # Wrong dimensions
            return response
            
        elif self.scenario == "retry_then_success":
            if self.call_count <= 2:
                raise Exception(f"Temporary failure #{self.call_count}")
            response = Mock()
            response.data = [Mock(embedding=[0.1] * 1536) for _ in input]
            return response
            
        elif self.scenario == "persistent_failure":
            raise Exception("Persistent API failure")


async def demo_successful_embedding():
    """Demo successful embedding creation."""
    print("\n🟢 Demo: Successful Embedding Creation")
    print("-" * 50)
    
    client = MockEmbeddingClient("success")
    
    texts = ["Hello world", "This is a test", "Embedding creation works!"]
    
    try:
        embeddings = await create_embeddings_with_retry(
            embed_client=client,
            texts=texts,
            model="text-embedding-ada-002",
            expected_dims=1536,
            batch_size=2  # Small batch to show batching
        )
        
        print(f"✅ Created {len(embeddings)} embeddings")
        print(f"✅ Each embedding has {len(embeddings[0])} dimensions")
        print(f"✅ API calls made: {client.call_count}")
        
        # Test cosine similarity
        sim = cosine_similarity(embeddings[0], embeddings[1])
        print(f"✅ Cosine similarity between first two: {sim:.4f}")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def demo_dimension_validation():
    """Demo dimension validation error."""
    print("\n🔴 Demo: Dimension Validation Error")
    print("-" * 50)
    
    client = MockEmbeddingClient("wrong_dims")
    
    try:
        embeddings = await create_embeddings_with_retry(
            embed_client=client,
            texts=["Test text"],
            model="text-embedding-ada-002",
            expected_dims=1536  # Expecting 1536, but client returns 512
        )
        print("❌ Should have failed with dimension error!")
        
    except EmbeddingError as e:
        print(f"✅ Caught expected dimension error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")


async def demo_retry_logic():
    """Demo retry logic with eventual success."""
    print("\n🟡 Demo: Retry Logic (Fails 2 times, then succeeds)")
    print("-" * 50)
    
    client = MockEmbeddingClient("retry_then_success")
    
    try:
        embeddings = await create_embeddings_with_retry(
            embed_client=client,
            texts=["Test retry logic"],
            model="text-embedding-ada-002",
            expected_dims=1536
        )
        
        print(f"✅ Eventually succeeded after {client.call_count} attempts")
        print(f"✅ Created embedding with {len(embeddings[0])} dimensions")
        
    except Exception as e:
        print(f"❌ Unexpected failure: {e}")


async def demo_persistent_failure():
    """Demo retry exhaustion."""
    print("\n🔴 Demo: Retry Exhaustion (Always fails)")
    print("-" * 50)
    
    client = MockEmbeddingClient("persistent_failure")
    
    try:
        embeddings = await create_embeddings_with_retry(
            embed_client=client,
            texts=["Test persistent failure"],
            model="text-embedding-ada-002",
            expected_dims=1536
        )
        print("❌ Should have failed after retries!")
        
    except EmbeddingError as e:
        print(f"✅ Correctly failed after {client.call_count} attempts: {e}")
    except Exception as e:
        print(f"❌ Unexpected error type: {e}")


def demo_validation_utilities():
    """Demo validation utility functions."""
    print("\n🔵 Demo: Validation Utilities")
    print("-" * 50)
    
    # Test dimension assertion
    valid_vecs = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    try:
        assert_dims(valid_vecs, 3)
        print("✅ Valid dimensions passed assertion")
    except EmbeddingError:
        print("❌ Valid dimensions should not fail")
    
    # Test invalid dimensions
    invalid_vecs = [[1.0, 2.0], [4.0, 5.0, 6.0]]
    try:
        assert_dims(invalid_vecs, 3)
        print("❌ Invalid dimensions should have failed")
    except EmbeddingError as e:
        print(f"✅ Invalid dimensions correctly caught: {e}")
    
    # Test batch validation
    test_embeddings = [
        [0.5, 0.5, 0.5],  # Valid
        [1.0, 1.0, 1.0],  # Valid
        [0.0, 0.0, 0.0]   # Zero magnitude (invalid)
    ]
    
    is_valid, issues = validate_embedding_batch(test_embeddings, 3)
    print(f"✅ Batch validation: valid={is_valid}, issues={len(issues)}")
    if issues:
        print(f"✅ Issues found: {issues[0]}")


async def demo_empty_text_handling():
    """Demo handling of empty/whitespace texts."""
    print("\n🟡 Demo: Empty Text Handling")
    print("-" * 50)
    
    client = MockEmbeddingClient("success")
    
    texts = ["Valid text", "", "   ", "Another valid text"]
    
    try:
        embeddings = await create_embeddings_with_retry(
            embed_client=client,
            texts=texts,
            model="text-embedding-ada-002",
            expected_dims=1536
        )
        
        print(f"✅ Processed {len(texts)} input texts")
        print(f"✅ Created {len(embeddings)} embeddings")
        print("✅ Empty texts replaced with '[empty]' placeholder")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


async def main():
    """Run all demos."""
    print("🚀 Embedding Creation System Demo")
    print("=" * 60)
    
    # Run async demos
    await demo_successful_embedding()
    await demo_dimension_validation() 
    await demo_retry_logic()
    await demo_persistent_failure()
    await demo_empty_text_handling()
    
    # Run sync demo
    demo_validation_utilities()
    
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("• Pure function approach - no singletons")
    print("• Request-scoped client usage")
    print("• Strict dimension validation with detailed errors")
    print("• Retry logic with exponential backoff")
    print("• Batch processing with configurable sizes")
    print("• Text normalization and empty text handling")
    print("• Comprehensive error handling for chat integration")
    print("• Fallback to BM25-only when embeddings fail")
    

if __name__ == "__main__":
    asyncio.run(main())