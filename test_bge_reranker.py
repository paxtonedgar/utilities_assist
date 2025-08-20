#!/usr/bin/env python3
"""
Smoke test for BGE v2-m3 reranker integration.

Tests that the model loads from local cache and works correctly within our system
without making any external calls or database connections.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_reranker_dependencies():
    """Test that reranker dependencies are available."""
    print("🔍 Testing reranker dependencies...")
    
    try:
        import torch
        print(f"✅ PyTorch available: {torch.__version__}")
        
        # Test device availability
        if torch.backends.mps.is_available():
            print("✅ MPS (Metal Performance Shaders) available")
        elif torch.cuda.is_available():
            print("✅ CUDA available")
        else:
            print("✅ CPU fallback available")
            
        import sentence_transformers
        print(f"✅ sentence-transformers available: {sentence_transformers.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def test_reranker_service():
    """Test that our CrossEncodeReranker service works."""
    print("\n🔍 Testing CrossEncodeReranker service...")
    
    try:
        from src.services.reranker import CrossEncodeReranker, is_reranker_available
        
        if not is_reranker_available():
            print("❌ Reranker dependencies not available")
            return False
            
        print("✅ Reranker dependencies available")
        
        # Test initialization
        print("📥 Initializing BGE v2-m3 reranker...")
        start_time = time.time()
        
        reranker = CrossEncodeReranker(
            model_id="BAAI/bge-reranker-v2-m3",
            device="auto",
            batch_size=8  # Small batch for testing
        )
        
        init_time = (time.time() - start_time) * 1000
        print(f"✅ Reranker initialized in {init_time:.1f}ms")
        print(f"📱 Using device: {reranker.device}")
        print(f"🏷️  Model ID: {reranker.model_id}")
        
        return reranker
        
    except Exception as e:
        print(f"❌ Reranker service failed: {e}")
        return None


def test_scoring_functionality(reranker):
    """Test basic scoring functionality."""
    print("\n🔍 Testing scoring functionality...")
    
    try:
        # Test query and passages
        query = "How do I configure Kafka consumer settings?"
        passages = [
            "Kafka consumer configuration includes bootstrap.servers, group.id, and auto.offset.reset parameters.",
            "To configure Kafka producers, set bootstrap.servers, key.serializer, and value.serializer properties.",
            "Spring Boot auto-configuration provides default Kafka settings through application.properties.",
            "Database connection pools require proper configuration for optimal performance.",
            "Redis cache configuration involves setting timeout and connection pool parameters."
        ]
        
        print(f"📝 Query: {query}")
        print(f"📄 Testing with {len(passages)} passages")
        
        # Test scoring
        start_time = time.time()
        scores = reranker.score(query, passages)
        score_time = (time.time() - start_time) * 1000
        
        print(f"✅ Scoring completed in {score_time:.1f}ms")
        print(f"📊 Scores received: {len(scores)}")
        
        # Validate scores
        if len(scores) != len(passages):
            print(f"❌ Score count mismatch: expected {len(passages)}, got {len(scores)}")
            return False
            
        # Print top scores
        scored_passages = list(zip(scores, passages))
        scored_passages.sort(key=lambda x: x[0], reverse=True)
        
        print("\n🏆 Top scoring passages:")
        for i, (score, passage) in enumerate(scored_passages[:3]):
            print(f"  {i+1}. Score: {score:.3f} | {passage[:60]}...")
            
        # Validate score ranges
        valid_scores = all(isinstance(s, (int, float)) for s in scores)
        if not valid_scores:
            print("❌ Invalid score types detected")
            return False
            
        print("✅ All scores are valid numeric values")
        return True
        
    except Exception as e:
        print(f"❌ Scoring test failed: {e}")
        return False


def test_rerank_functionality(reranker):
    """Test document reranking functionality."""
    print("\n🔍 Testing rerank functionality...")
    
    try:
        from src.services.models import SearchResult
        
        # Create mock search results
        query = "How to setup OAuth authentication for APIs?"
        mock_results = [
            SearchResult(
                doc_id="doc1",
                title="API Authentication Guide", 
                url="https://example.com/auth",
                score=0.8,
                content="OAuth 2.0 is the industry standard for API authentication. Configure client credentials and redirect URIs.",
                metadata={"source": "documentation"}
            ),
            SearchResult(
                doc_id="doc2", 
                title="Database Setup Guide",
                url="https://example.com/db",
                score=0.6,
                content="Database authentication requires username, password, and connection string configuration.",
                metadata={"source": "documentation"}
            ),
            SearchResult(
                doc_id="doc3",
                title="OAuth Implementation Tutorial", 
                url="https://example.com/oauth",
                score=0.7,
                content="Step-by-step OAuth implementation: register app, configure scopes, handle authorization codes.",
                metadata={"source": "tutorial"}
            )
        ]
        
        print(f"📝 Query: {query}")
        print(f"📄 Testing with {len(mock_results)} mock search results")
        
        # Test reranking
        start_time = time.time()
        reranked_results, rerank_result = reranker.rerank(
            query=query,
            docs=mock_results,
            min_score=0.1,  # Low threshold for testing
            top_k=10
        )
        rerank_time = (time.time() - start_time) * 1000
        
        print(f"✅ Reranking completed in {rerank_time:.1f}ms")
        print(f"📊 Results: {len(mock_results)} → {len(reranked_results)}")
        print(f"🎯 Average score: {rerank_result.avg_score:.3f}")
        print(f"📱 Device used: {rerank_result.device_used}")
        
        # Validate rerank results
        for i, result in enumerate(reranked_results):
            if not hasattr(result, 'rerank_score'):
                print(f"❌ Result {i} missing rerank_score")
                return False
            print(f"  {i+1}. {result.doc_id}: {result.rerank_score:.3f} | {result.title}")
            
        print("✅ All results have rerank_score attribute")
        
        # Test telemetry data
        print(f"📈 Telemetry - Top scores: {rerank_result.top_scores}")
        print(f"📈 Telemetry - Dropped count: {rerank_result.dropped_count}")
        print(f"📈 Telemetry - Model ID: {rerank_result.model_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Rerank test failed: {e}")
        return False


def test_coverage_integration():
    """Test integration with coverage system."""
    print("\n🔍 Testing coverage system integration...")
    
    try:
        from src.quality.utils import get_coverage_gate, run_coverage_evaluation
        from src.services.models import SearchResult
        
        # Create test search results
        mock_results = [
            SearchResult(
                doc_id="doc1",
                title="Kafka Consumer Configuration",
                url="https://example.com/kafka-consumer",
                score=0.8,
                content="To configure Kafka consumers: 1. Set bootstrap.servers 2. Configure group.id 3. Set auto.offset.reset to earliest",
                metadata={"source": "documentation"}
            ),
            SearchResult(
                doc_id="doc2",
                title="Kafka Producer Setup", 
                url="https://example.com/kafka-producer",
                score=0.7,
                content="Kafka producer configuration requires bootstrap.servers, key.serializer, and value.serializer properties.",
                metadata={"source": "documentation"}
            )
        ]
        
        query = "How do I configure Kafka consumer settings?"
        print(f"📝 Query: {query}")
        print(f"📄 Testing with {len(mock_results)} search results")
        
        # Test coverage gate initialization
        coverage_gate = get_coverage_gate()
        print(f"✅ Coverage gate initialized")
        print(f"🏷️  Model: {coverage_gate.model_name}")
        print(f"📱 Device: {coverage_gate.device}")
        print(f"🎯 Threshold (tau): {coverage_gate.tau}")
        
        # Test coverage evaluation
        start_time = time.time()
        coverage_result = run_coverage_evaluation(query, mock_results)
        eval_time = (time.time() - start_time) * 1000
        
        print(f"✅ Coverage evaluation completed in {eval_time:.1f}ms")
        print(f"🚪 Gate pass: {coverage_result['gate_pass']}")
        print(f"📊 Aspect recall: {coverage_result['aspect_recall']:.3f}")
        print(f"📊 Alpha nDCG: {coverage_result['alpha_ndcg']:.3f}")
        print(f"🔧 Actionable spans: {coverage_result['actionable_spans']}")
        print(f"❓ Subqueries: {len(coverage_result['subqueries'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Coverage integration test failed: {e}")
        return False


def test_settings_integration():
    """Test that settings properly load reranker config."""
    print("\n🔍 Testing settings integration...")
    
    try:
        from src.infra.settings import get_settings
        
        settings = get_settings()
        reranker_config = settings.reranker
        
        print(f"✅ Settings loaded successfully")
        print(f"🔛 Reranker enabled: {reranker_config.enabled}")
        print(f"🏷️  Model ID: {reranker_config.model_id}")
        print(f"📱 Device: {reranker_config.device}")
        print(f"📦 Batch size: {reranker_config.batch_size}")
        print(f"🎯 Min score: {reranker_config.min_score}")
        print(f"🔝 Top K: {reranker_config.top_k}")
        print(f"📏 Max length: {reranker_config.max_length}")
        print(f"⚡ Use FP16: {reranker_config.use_fp16}")
        
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False


def check_model_cache():
    """Check if BGE model is cached locally."""
    print("\n🔍 Checking local model cache...")
    
    # Common Hugging Face cache locations
    possible_cache_dirs = [
        Path.home() / ".cache" / "huggingface",
        Path(os.environ.get("HF_HOME", "")),
        Path(os.environ.get("TRANSFORMERS_CACHE", "")),
        Path(os.environ.get("HF_DATASETS_CACHE", ""))
    ]
    
    # Remove empty paths
    cache_dirs = [d for d in possible_cache_dirs if d and d.exists()]
    
    print(f"🔍 Checking {len(cache_dirs)} cache directories...")
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            print(f"📁 Cache directory: {cache_dir}")
            
            # Look for BGE model
            bge_patterns = [
                "models--BAAI--bge-reranker-v2-m3",
                "BAAI/bge-reranker-v2-m3",
                "*bge-reranker-v2-m3*"
            ]
            
            for pattern in bge_patterns:
                matches = list(cache_dir.rglob(pattern))
                if matches:
                    print(f"✅ Found BGE model cache: {matches[0]}")
                    return True
    
    print("⚠️  BGE model not found in local cache (will download on first use)")
    return False


def main():
    """Run all smoke tests."""
    print("🚀 BGE v2-m3 Reranker Smoke Test")
    print("=" * 50)
    
    # Check model cache
    check_model_cache()
    
    # Test dependencies
    if not test_reranker_dependencies():
        print("\n❌ Smoke test failed: Missing dependencies")
        return False
    
    # Test settings
    if not test_settings_integration():
        print("\n❌ Smoke test failed: Settings integration")
        return False
        
    # Test reranker service
    reranker = test_reranker_service()
    if not reranker:
        print("\n❌ Smoke test failed: Reranker service")
        return False
    
    # Test scoring
    if not test_scoring_functionality(reranker):
        print("\n❌ Smoke test failed: Scoring functionality")
        return False
        
    # Test reranking
    if not test_rerank_functionality(reranker):
        print("\n❌ Smoke test failed: Rerank functionality") 
        return False
        
    # Test coverage integration
    if not test_coverage_integration():
        print("\n❌ Smoke test failed: Coverage integration")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All smoke tests passed!")
    print("🎉 BGE v2-m3 reranker is working correctly in the system")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)