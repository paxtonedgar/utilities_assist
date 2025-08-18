#!/usr/bin/env python3
"""
Test script to verify that the vector search fixes work correctly.
This script tests the updated kNN query structure that uses script_score
with cosineSimilarity to be compatible with both dense_vector and knn_vector fields.
"""

import sys
import os
import json
from typing import List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_knn_query_structure():
    """Test that the kNN query structure is correctly formatted for script_score."""
    try:
        from src.infra.opensearch_client import OpenSearchClient
        from src.infra.settings import get_settings
        
        print("✅ Successfully imported OpenSearchClient and settings")
        
        # Create a mock client to test query building
        settings = get_settings()
        client = OpenSearchClient(settings)
        
        # Test vector (smaller for cleaner output)
        test_vector = [0.1] * 10
        
        # Test simple kNN query building
        print("\n🔍 Testing _build_simple_knn_query method...")
        simple_query = client._build_simple_knn_query(test_vector, k=10)
        
        print("Simple kNN Query Structure: [Validating structure...]")
        
        # Verify the query uses script_score instead of knn
        assert "query" in simple_query, "No 'query' field in simple_query"
        
        # Check if it's using script_score directly or within bool/must
        if "script_score" in simple_query["query"]:
            script_score = simple_query["query"]["script_score"]
            assert "query" in script_score, "No 'query' field in script_score"
            assert "script" in script_score, "No 'script' field in script_score"
            assert "cosineSimilarity" in script_score["script"]["source"], "cosineSimilarity not found in script source"
            print("✅ Found script_score query with cosineSimilarity")
        elif "bool" in simple_query["query"] and "must" in simple_query["query"]["bool"]:
            script_score_found = False
            for clause in simple_query["query"]["bool"]["must"]:
                if "script_score" in clause:
                    script_score_found = True
                    script_score = clause["script_score"]
                    assert "query" in script_score, "No 'query' field in script_score"
                    assert "script" in script_score, "No 'script' field in script_score"
                    assert "cosineSimilarity" in script_score["script"]["source"], "cosineSimilarity not found in script source"
                    print("✅ Found script_score query with cosineSimilarity")
                    break
            
            assert script_score_found, "script_score query not found in simple kNN query"
        else:
            raise AssertionError(f"Unexpected query structure: {simple_query['query']}")
        
        # Test full kNN query building
        print("\n🔍 Testing _build_knn_query method...")
        from src.infra.opensearch_client import SearchFilters
        filters = SearchFilters()  # Empty filters for testing
        full_query = client._build_knn_query(test_vector, filters, k=10, ef_search=256)
        
        print("Full kNN Query Structure: [Validating structure...]")
        
        # Verify the query uses script_score (may be wrapped in function_score for time decay)
        assert "query" in full_query, "No 'query' field in full_query"
        
        # Check if it's wrapped in function_score (for time decay)
        if "function_score" in full_query["query"]:
            function_score = full_query["query"]["function_score"]
            assert "query" in function_score, "No 'query' field in function_score"
            
            # The inner query should be script_score
            if "script_score" in function_score["query"]:
                script_score = function_score["query"]["script_score"]
                assert "query" in script_score, "No 'query' field in script_score"
                assert "script" in script_score, "No 'script' field in script_score"
                assert "cosineSimilarity" in script_score["script"]["source"], "cosineSimilarity not found in script source"
                print("✅ Found script_score query with cosineSimilarity (wrapped in function_score)")
            else:
                raise AssertionError(f"Expected script_score in function_score query: {function_score['query']}")
        
        # Check if it's using script_score directly
        elif "script_score" in full_query["query"]:
            script_score = full_query["query"]["script_score"]
            assert "query" in script_score, "No 'query' field in script_score"
            assert "script" in script_score, "No 'script' field in script_score"
            assert "cosineSimilarity" in script_score["script"]["source"], "cosineSimilarity not found in script source"
            print("✅ Found script_score query with cosineSimilarity")
        
        # Check if it's within bool/must
        elif "bool" in full_query["query"] and "must" in full_query["query"]["bool"]:
            script_score_found = False
            for clause in full_query["query"]["bool"]["must"]:
                if "script_score" in clause:
                    script_score_found = True
                    script_score = clause["script_score"]
                    assert "query" in script_score, "No 'query' field in script_score"
                    assert "script" in script_score, "No 'script' field in script_score"
                    assert "cosineSimilarity" in script_score["script"]["source"], "cosineSimilarity not found in script source"
                    print("✅ Found script_score query with cosineSimilarity")
                    break
            
            assert script_score_found, "script_score query not found in full kNN query"
        else:
            raise AssertionError(f"Unexpected query structure: {full_query['query']}")
        
        print("\n✅ All query structure tests passed!")
        print("✅ Vector search queries are now compatible with dense_vector fields")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_vector_field_configuration():
    """Test that the vector field configuration is correct."""
    try:
        from src.infra.search_config import OpenSearchConfig
        
        print("\n🔍 Testing vector field configuration...")
        
        # Test vector field name (requires index name parameter)
        index_name = "khub-opensearch-index"
        vector_field = OpenSearchConfig.get_vector_field(index_name)
        print(f"Vector field name: {vector_field}")
        assert vector_field == "embedding", f"Expected 'embedding', got '{vector_field}'"
        
        # Test embedding dimensions
        dimensions = OpenSearchConfig.EMBEDDING_DIMENSIONS
        print(f"Embedding dimensions: {dimensions}")
        assert dimensions == 1536, f"Expected 1536, got {dimensions}"
        
        print("✅ Vector field configuration is correct")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Vector Search Fixes")
    print("=" * 50)
    
    # Test vector field configuration
    config_test = test_vector_field_configuration()
    
    # Test query structure
    query_test = test_knn_query_structure()
    
    print("\n" + "=" * 50)
    if config_test and query_test:
        print("🎉 All tests passed! Vector search fixes are working correctly.")
        print("✅ The application should now work with K-hub cluster's dense_vector fields.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        sys.exit(1)