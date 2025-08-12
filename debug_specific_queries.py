#!/usr/bin/env python3
"""
Debug specific queries to understand why they're not matching.
"""

from eval.os_eval_client import OpenSearchEvalClient

def test_specific_queries():
    """Test specific queries that should be matching."""
    
    client = OpenSearchEvalClient()
    
    test_cases = [
        {
            "query": "How do I start new utility service?",  # Q001
            "acl": "public", 
            "expected": ["UTILS:CUST:START_SERVICE:v4#overview", "UTILS:CUST:START_SERVICE:v4#eligibility", "UTILS:CUST:START_SERVICE:v1#eligibility"]
        },
        {
            "query": "Walk me through stopping utility service",  # Q008  
            "acl": "public",
            "expected": ["UTILS:CUST:STOP_SERVICE:v3#overview", "UTILS:CUST:STOP_SERVICE:v3#final_bill"]
        },
        {
            "query": "How do I switch to time-of-use rates?",  # Q011
            "acl": "public", 
            "expected": ["UTILS:RATE:TOU_SWITCH:v3#overview", "UTILS:RATE:TOU_SWITCH:v3#eligibility"]
        },
        {
            "query": "What credit score is needed to start service?",  # Q002
            "acl": "grp_care",
            "expected": ["UTILS:CUST:START_SERVICE:v4#eligibility", "UTILS:CUST:START_SERVICE:v1#eligibility"]
        }
    ]
    
    for i, test in enumerate(test_cases):
        print(f"\n🔍 Test {i+1}: {test['query']}")
        print(f"🔒 ACL: {test['acl']}")
        print(f"📋 Expected: {test['expected']}")
        
        # Test simple mode
        results = client.rank_docs(
            query_text=test['query'],
            mode="simple",
            acl_hash=test['acl'],
            index_alias="confluence_current"
        )
        
        print(f"📊 Results ({len(results)} found):")
        if results:
            for j, (doc_id, score) in enumerate(results):
                is_expected = "✅" if doc_id in test['expected'] else "❌"
                print(f"   {j+1}. {doc_id} (score: {score:.3f}) {is_expected}")
        else:
            print("   No results found")
        
        # Calculate precision at 5
        if results:
            top_5_ids = [doc_id for doc_id, _ in results[:5]]
            hits = sum(1 for doc_id in top_5_ids if doc_id in test['expected'])
            p5 = hits / 5.0
            print(f"📈 Precision@5: {p5:.3f}")


if __name__ == "__main__":
    test_specific_queries()