#!/usr/bin/env python3
"""
Quick test script for the new OpenSearch client with ACL filters and RRF.
"""

import os
import asyncio
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project to path
import sys
sys.path.insert(0, '.')

from src.infra.config import get_settings
from src.infra.opensearch_client import create_search_client, SearchFilters


async def test_search_client():
    """Test the new OpenSearch client functionality."""
    
    print("🔍 Testing OpenSearch Client with ACL Filters + Time Decay + RRF")
    print("=" * 60)
    
    # Load settings
    settings = get_settings()
    print(f"📊 Profile: {settings.profile}")
    print(f"🔗 OpenSearch Host: {settings.search.host}")
    
    # Create client
    search_client = create_search_client(settings.search)
    
    # Test 1: Health check
    print("\n1️⃣ Health Check")
    health = search_client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Authentication: {health['authentication']}")
    if health['status'] == 'healthy':
        print(f"   Cluster: {health['cluster_name']} ({health['cluster_status']})")
        print(f"   Nodes: {health['node_count']} total, {health['data_nodes']} data")
        print(f"   Index exists: {health['index_exists']}")
    
    # Test 2: BM25 search with filters
    print("\n2️⃣ BM25 Search with Time Decay")
    try:
        filters = SearchFilters(
            content_type="confluence",
            updated_after=datetime.now() - timedelta(days=365)  # Last year
        )
        
        bm25_response = search_client.bm25_search(
            query="customer utility API",
            filters=filters,
            k=5,
            time_decay_half_life_days=120
        )
        
        print(f"   Results: {len(bm25_response.results)}/{bm25_response.total_hits}")
        print(f"   Time: {bm25_response.took_ms}ms")
        for i, result in enumerate(bm25_response.results[:3]):
            print(f"   {i+1}. {result.title[:60]}... (score: {result.score:.3f})")
            
    except Exception as e:
        print(f"   ❌ BM25 failed: {e}")
    
    # Test 3: Mock kNN search (would need embeddings in real test)
    print("\n3️⃣ kNN Search (Mock)")
    try:
        # Generate mock embedding vector (1536 dimensions)
        mock_vector = [0.1] * 1536
        
        knn_response = search_client.knn_search(
            query_vector=mock_vector,
            filters=filters,
            k=5,
            ef_search=128
        )
        
        print(f"   Results: {len(knn_response.results)}/{knn_response.total_hits}")
        print(f"   Time: {knn_response.took_ms}ms")
        for i, result in enumerate(knn_response.results[:3]):
            print(f"   {i+1}. {result.title[:60]}... (score: {result.score:.3f})")
            
    except Exception as e:
        print(f"   ❌ kNN failed: {e}")
        knn_response = None
    
    # Test 4: RRF fusion
    if 'bm25_response' in locals() and knn_response and len(bm25_response.results) > 0 and len(knn_response.results) > 0:
        print("\n4️⃣ RRF Fusion")
        try:
            rrf_response = search_client.rrf_fuse(
                bm25_response=bm25_response,
                knn_response=knn_response,
                k=5,
                rrf_k=60
            )
            
            print(f"   Fused Results: {len(rrf_response.results)}")
            print(f"   Total Time: {rrf_response.took_ms}ms")
            for i, result in enumerate(rrf_response.results):
                print(f"   {i+1}. {result.title[:60]}... (RRF: {result.score:.6f})")
                
        except Exception as e:
            print(f"   ❌ RRF failed: {e}")
    else:
        print("\n4️⃣ RRF Fusion")
        print("   ⚠️  Skipped - need both BM25 and kNN results")
    
    # Test 5: Filter variations
    print("\n5️⃣ ACL Filter Test")
    try:
        acl_filters = SearchFilters(
            acl_hash="public",
            space_key="UTILITIESPRODUCT"
        )
        
        acl_response = search_client.bm25_search(
            query="transaction",
            filters=acl_filters,
            k=3
        )
        
        print(f"   ACL-filtered Results: {len(acl_response.results)}")
        if acl_response.results:
            print(f"   First result: {acl_response.results[0].title[:50]}...")
            
    except Exception as e:
        print(f"   ❌ ACL filter failed: {e}")
    
    print("\n✅ OpenSearch Client Test Complete!")


if __name__ == "__main__":
    asyncio.run(test_search_client())