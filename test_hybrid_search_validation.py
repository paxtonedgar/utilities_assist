#!/usr/bin/env python3
"""
Test script to validate hybrid RRF search functionality.
This script tests that both BM25 and kNN components work together correctly.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from infra.opensearch_client import OpenSearchClient
from infra.search_config import OpenSearchConfig
from services.models import SearchFilters
import numpy as np

def test_hybrid_search():
    """Test hybrid RRF search with both BM25 and kNN components."""
    print("Testing hybrid RRF search functionality...")
    
    # Initialize config and client
    config = OpenSearchConfig()
    client = OpenSearchClient(config)
    
    # Test parameters
    test_query = "API documentation"
    test_vector = np.random.rand(10).tolist()  # Small test vector
    index_name = "test_index"
    k = 5
    ef_search = 100
    filters = SearchFilters()
    time_decay_half_life_days = 365
    
    print(f"\n1. Testing hybrid search query generation...")
    
    try:
        # Test hybrid search query generation
        hybrid_query = client._build_hybrid_search_query(
            query=test_query,
            query_vector=test_vector,
            filters=filters,
            k=k,
            ef_search=ef_search,
            time_decay_half_life_days=time_decay_half_life_days,
            index=index_name
        )
        
        print("✓ Hybrid search query generated successfully")
        
        # Validate query structure
        assert "query" in hybrid_query, "Query should contain 'query' field"
        assert "rank" in hybrid_query, "Query should contain 'rank' field for RRF"
        
        # Check RRF configuration
        rank_config = hybrid_query["rank"]
        assert "rrf" in rank_config, "Rank should use RRF (Reciprocal Rank Fusion)"
        assert "window_size" in rank_config["rrf"], "RRF should have window_size"
        
        print("✓ RRF configuration validated")
        
        # Check that query contains both BM25 and kNN components
        query_part = hybrid_query["query"]
        
        # Should be a bool query with multiple should clauses
        if "bool" in query_part:
            bool_query = query_part["bool"]
            assert "should" in bool_query, "Bool query should have 'should' clauses"
            should_clauses = bool_query["should"]
            
            # Look for BM25 and vector search components
            has_text_search = False
            has_vector_search = False
            
            for clause in should_clauses:
                # Check for text search (BM25)
                if any(key in clause for key in ["multi_match", "match", "query_string"]):
                    has_text_search = True
                    print("✓ Found BM25 text search component")
                
                # Check for vector search (script_score or function_score)
                if "script_score" in clause or "function_score" in clause:
                    has_vector_search = True
                    print("✓ Found kNN vector search component")
                    
                    # Validate vector search uses cosineSimilarity
                    if "script_score" in clause:
                        script = clause["script_score"].get("script", {})
                        source = script.get("source", "")
                        assert "cosineSimilarity" in source, "Vector search should use cosineSimilarity"
                        print("✓ Vector search uses cosineSimilarity")
                    elif "function_score" in clause:
                        # Check nested script_score within function_score
                        func_query = clause["function_score"].get("query", {})
                        if "script_score" in func_query:
                            script = func_query["script_score"].get("script", {})
                            source = script.get("source", "")
                            assert "cosineSimilarity" in source, "Vector search should use cosineSimilarity"
                            print("✓ Vector search uses cosineSimilarity (within function_score)")
            
            assert has_text_search, "Hybrid search should include BM25 text search component"
            assert has_vector_search, "Hybrid search should include kNN vector search component"
            
        else:
            raise AssertionError(f"Unexpected query structure: {query_part}")
        
        print("\n✓ All hybrid search validation tests passed!")
        print("✓ Hybrid RRF search correctly combines BM25 and kNN components")
        print("✓ Vector search uses cosineSimilarity for dense_vector fields")
        print("✓ RRF ranking is properly configured")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Hybrid search validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("=== Hybrid RRF Search Validation ===")
    
    success = test_hybrid_search()
    
    if success:
        print("\n🎉 All tests passed! Hybrid search is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed! Hybrid search needs attention.")
        sys.exit(1)

if __name__ == "__main__":
    main()