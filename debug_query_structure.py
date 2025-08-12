#!/usr/bin/env python3
"""
Debug query structures to understand why simple queries aren't matching.
"""

import json
import requests
from src.infra.config import get_settings
from eval.os_eval_client import OpenSearchEvalClient
from src.infra.opensearch_client import SearchFilters

def test_query_structures():
    """Compare different query structures."""
    
    settings = get_settings()
    
    # Test query and expected results
    test_query = "start utility service"
    acl_filter = "public"
    
    # Test 1: Current simple_query_string approach (from evaluation client)
    simple_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "simple_query_string": {
                            "query": test_query,
                            "fields": ["title^2", "section^1.5", "body^1"], 
                            "default_operator": "and",
                            "minimum_should_match": "70%"
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {"acl_hash": acl_filter}
                    }
                ]
            }
        }
    }
    
    # Test 2: More lenient multi_match approach
    lenient_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": test_query,
                            "fields": ["title^2", "section^1.5", "body^1"],
                            "type": "best_fields"
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {"acl_hash": acl_filter}
                    }
                ]
            }
        }
    }
    
    # Test 3: Even more lenient with OR operator
    very_lenient_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "simple_query_string": {
                            "query": test_query,
                            "fields": ["title^2", "section^1.5", "body^1"],
                            "default_operator": "or",  # Changed to OR
                            "minimum_should_match": "50%"  # Reduced threshold
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {"acl_hash": acl_filter}
                    }
                ]
            }
        }
    }
    
    # Test against OpenSearch
    session = requests.Session()
    session.auth = ("admin", "admin")
    base_url = "http://localhost:9200"
    
    test_queries = [
        ("Current simple_query_string (AND + 70%)", simple_query),
        ("Multi_match (best_fields)", lenient_query), 
        ("Lenient simple_query_string (OR + 50%)", very_lenient_query)
    ]
    
    for name, query in test_queries:
        print(f"\n🔍 Testing: {name}")
        print("=" * 60)
        print(json.dumps(query, indent=2))
        print("=" * 60)
        
        try:
            url = f"{base_url}/confluence_current/_search"
            response = session.post(url, json=query, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            total_hits = data.get('hits', {}).get('total', {}).get('value', 0)
            
            print(f"📊 Results: {total_hits} hits")
            
            for hit in data.get('hits', {}).get('hits', []):
                doc_id = hit['_id']
                score = hit['_score']
                title = hit['_source'].get('title', 'No title')
                acl = hit['_source'].get('acl_hash', 'No ACL')
                print(f"   • {doc_id} (score: {score:.3f}, ACL: {acl})")
                print(f"     {title}")
            
        except Exception as e:
            print(f"❌ Query failed: {e}")


if __name__ == "__main__":
    test_query_structures()