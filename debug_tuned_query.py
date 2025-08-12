#!/usr/bin/env python3
"""
Debug script to test individual tuned BM25 queries and identify 400 errors.
"""

import json
import sys
from src.infra.opensearch_client import OpenSearchClient, SearchFilters
from src.infra.config import SearchCfg

def test_tuned_query():
    """Test a single tuned BM25 query to debug 400 errors."""
    
    # Create search configuration
    config = SearchCfg(
        host="http://localhost:9200",
        username="admin",
        password="admin", 
        timeout_s=30
    )
    
    client = OpenSearchClient(config)
    
    # Test query that's failing
    test_query = "How do I start new utility service?"
    filters = SearchFilters(acl_hash="public")
    
    print(f"🔍 Testing tuned BM25 query: '{test_query}'")
    print(f"🔒 ACL Filter: {filters.acl_hash}")
    
    try:
        # Build the query manually to inspect it
        search_body = client._build_bm25_query(
            query=test_query,
            filters=filters,
            k=10,
            time_decay_half_life_days=120
        )
        
        print(f"\n📋 Generated Query Structure:")
        print("=" * 60)
        print(json.dumps(search_body, indent=2))
        print("=" * 60)
        
        # Test the query against OpenSearch
        url = f"{client.base_url}/confluence_current/_search"
        response = client.session.post(url, json=search_body, timeout=30)
        
        print(f"\n📊 Response Status: {response.status_code}")
        
        if response.status_code == 400:
            print(f"❌ 400 Bad Request Error:")
            try:
                error_detail = response.json()
                print(json.dumps(error_detail, indent=2))
            except:
                print(f"Raw response text: {response.text}")
        else:
            print(f"✅ Query executed successfully")
            data = response.json()
            print(f"📈 Total hits: {data.get('hits', {}).get('total', {}).get('value', 0)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def test_simple_query():
    """Test a simple query for comparison."""
    config = SearchCfg(
        host="http://localhost:9200",
        username="admin",
        password="admin",
        timeout_s=30
    )
    
    client = OpenSearchClient(config)
    
    print(f"\n🔍 Testing simple BM25 query for comparison...")
    
    # Simple query structure (what works)
    simple_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": "How do I start new utility service?",
                            "fields": ["title^2", "body^1"],
                            "type": "best_fields"
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {"acl_hash": "public"}
                    }
                ]
            }
        },
        "size": 10
    }
    
    print(f"📋 Simple Query Structure:")
    print(json.dumps(simple_query, indent=2))
    
    try:
        url = f"{client.base_url}/confluence_current/_search"
        response = client.session.post(url, json=simple_query, timeout=30)
        
        print(f"\n📊 Simple Query Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Simple query works - hits: {data.get('hits', {}).get('total', {}).get('value', 0)}")
        else:
            print(f"❌ Even simple query failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Simple query error: {e}")


if __name__ == "__main__":
    test_tuned_query()
    test_simple_query()