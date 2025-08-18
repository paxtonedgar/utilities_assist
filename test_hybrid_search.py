#!/usr/bin/env python3
"""
Test script for hybrid search functionality
"""
import json
import requests
import numpy as np

def generate_query_embedding():
    """Generate a test query embedding vector"""
    vector = np.random.normal(0, 1, 1024)
    vector = vector / np.linalg.norm(vector)
    return vector.tolist()

def test_hybrid_search():
    """Test the hybrid search functionality"""
    base_url = "http://localhost:9200"
    index_name = "khub-opensearch-index"
    
    # Test query
    query_text = "vector search optimization"
    query_embedding = generate_query_embedding()
    
    # Hybrid search query (based on the fixed search_config.py)
    hybrid_query = {
        "size": 10,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "dis_max": {
                            "queries": [
                                {
                                    "match": {
                                        "title": {
                                            "query": query_text,
                                            "boost": 4
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "section": {
                                            "query": query_text,
                                            "boost": 2
                                        }
                                    }
                                },
                                {
                                    "match": {
                                        "body": {
                                            "query": query_text,
                                            "boost": 1
                                        }
                                    }
                                }
                            ],
                            "tie_breaker": 0.3
                        }
                    },
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding,
                                "k": 10
                            }
                        }
                    }
                ]
            }
        }
    }
    
    # Execute the search
    search_url = f"{base_url}/{index_name}/_search"
    response = requests.post(search_url, json=hybrid_query, headers={"Content-Type": "application/json"})
    
    print(f"Hybrid Search Test for query: '{query_text}'")
    print("=" * 50)
    
    if response.status_code == 200:
        results = response.json()
        hits = results.get('hits', {}).get('hits', [])
        
        print(f"✓ Search successful! Found {len(hits)} results")
        print(f"Total hits: {results.get('hits', {}).get('total', {}).get('value', 0)}")
        
        for i, hit in enumerate(hits, 1):
            source = hit.get('_source', {})
            score = hit.get('_score', 0)
            print(f"\n{i}. {source.get('title', 'No title')} (Score: {score:.4f})")
            print(f"   Section: {source.get('section', 'N/A')}")
            print(f"   Body: {source.get('body', 'No body')[:100]}...")
            
    else:
        print(f"✗ Search failed with status {response.status_code}")
        print(f"Error: {response.text}")
        
    return response.status_code == 200

def test_bm25_only():
    """Test BM25-only search for comparison"""
    base_url = "http://localhost:9200"
    index_name = "khub-opensearch-index"
    
    query_text = "vector search optimization"
    
    bm25_query = {
        "size": 10,
        "query": {
            "dis_max": {
                "queries": [
                    {
                        "match": {
                            "title": {
                                "query": query_text,
                                "boost": 4
                            }
                        }
                    },
                    {
                        "match": {
                            "section": {
                                "query": query_text,
                                "boost": 2
                            }
                        }
                    },
                    {
                        "match": {
                            "body": {
                                "query": query_text,
                                "boost": 1
                            }
                        }
                    }
                ],
                "tie_breaker": 0.3
            }
        }
    }
    
    search_url = f"{base_url}/{index_name}/_search"
    response = requests.post(search_url, json=bm25_query, headers={"Content-Type": "application/json"})
    
    print(f"\nBM25-Only Search Test for query: '{query_text}'")
    print("=" * 50)
    
    if response.status_code == 200:
        results = response.json()
        hits = results.get('hits', {}).get('hits', [])
        
        print(f"✓ BM25 search successful! Found {len(hits)} results")
        
        for i, hit in enumerate(hits, 1):
            source = hit.get('_source', {})
            score = hit.get('_score', 0)
            print(f"{i}. {source.get('title', 'No title')} (Score: {score:.4f})")
            
    else:
        print(f"✗ BM25 search failed: {response.text}")

if __name__ == "__main__":
    # Test hybrid search
    hybrid_success = test_hybrid_search()
    
    # Test BM25 for comparison
    test_bm25_only()
    
    if hybrid_success:
        print("\n🎉 Hybrid search is working correctly!")
    else:
        print("\n❌ Hybrid search test failed")