#!/usr/bin/env python3
"""
Minimal smoke test to verify basic multi_match queries work.
This strips away all complexity and tests pure term matching.
"""

import requests
import json
from eval.os_eval_client import OpenSearchEvalClient

def minimal_smoke_test():
    """Test basic multi_match queries without any complexity."""
    
    # OpenSearch direct connection
    session = requests.Session()
    session.auth = ("admin", "admin")
    base_url = "http://localhost:9200"
    
    # Test queries that should work based on our document content
    test_cases = [
        {
            "name": "Credit Score Query",
            "query": "credit score service",
            "expected_doc": "UTILS:CUST:START_SERVICE:v4#eligibility",
            "acl": "grp_care"
        },
        {
            "name": "Start Service Query", 
            "query": "start utility service",
            "expected_doc": "UTILS:CUST:START_SERVICE:v4#overview",
            "acl": "public"
        },
        {
            "name": "Stop Service Query",
            "query": "stop utility service",
            "expected_doc": "UTILS:CUST:STOP_SERVICE:v3#overview", 
            "acl": "public"
        },
        {
            "name": "TOU Rates Query",
            "query": "time use rates",
            "expected_doc": "UTILS:RATE:TOU_SWITCH:v3#overview",
            "acl": "public"
        },
        {
            "name": "Power Outage Query",
            "query": "power outage reporting",
            "expected_doc": "UTILS:EMERGENCY:OUTAGE:v1#reporting",
            "acl": "public"
        }
    ]
    
    print("🔥 MINIMAL SMOKE TEST - Basic Multi-Match Queries")
    print("=" * 60)
    
    for test in test_cases:
        print(f"\n🔍 Testing: {test['name']}")
        print(f"   Query: '{test['query']}'")
        print(f"   Expected: {test['expected_doc']}")
        print(f"   ACL: {test['acl']}")
        
        # Minimal multi_match query
        minimal_query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": test['query'],
                                "fields": ["title^5", "body"]
                            }
                        }
                    ],
                    "filter": [
                        {
                            "term": {"acl_hash": test['acl']}
                        }
                    ]
                }
            },
            "size": 3,
            "_source": ["title", "canonical_id", "acl_hash"]
        }
        
        try:
            url = f"{base_url}/confluence_current/_search"
            response = session.post(url, json=minimal_query, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            hits = data.get('hits', {}).get('hits', [])
            total = data.get('hits', {}).get('total', {}).get('value', 0)
            
            print(f"   📊 Results: {total} total hits")
            
            if hits:
                found_expected = False
                for i, hit in enumerate(hits):
                    doc_id = hit.get('_id', 'Unknown')
                    score = hit.get('_score', 0)
                    title = hit.get('_source', {}).get('title', 'No title')
                    
                    is_expected = "✅" if doc_id == test['expected_doc'] else "◦"
                    if doc_id == test['expected_doc']:
                        found_expected = True
                        
                    print(f"      {i+1}. {doc_id} (score: {score:.2f}) {is_expected}")
                    print(f"         {title}")
                
                if found_expected:
                    print(f"   ✅ SUCCESS: Found expected document!")
                else:
                    print(f"   ⚠️  PARTIAL: Got results but not expected document")
            else:
                print(f"   ❌ FAILED: No results found")
                
        except Exception as e:
            print(f"   💥 ERROR: {e}")
    
    print(f"\n" + "=" * 60)
    print("🏁 Smoke test complete. If any queries show SUCCESS, basic matching works!")


if __name__ == "__main__":
    minimal_smoke_test()