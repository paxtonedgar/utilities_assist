#!/usr/bin/env python3
"""
Debug why the evaluation client isn't working the same as direct queries.
"""

from eval.os_eval_client import OpenSearchEvalClient

def test_evaluation_client():
    """Test evaluation client vs direct query."""
    
    client = OpenSearchEvalClient()
    
    print("🔍 Testing evaluation client with grp_care query...")
    
    # Test the exact same query that worked directly
    results = client.rank_docs(
        query_text="credit score service",
        mode="simple", 
        acl_hash="grp_care",
        index_alias="confluence_current"
    )
    
    print(f"📊 Evaluation client results ({len(results)} found):")
    for i, (doc_id, score) in enumerate(results):
        print(f"   {i+1}. {doc_id} (score: {score:.3f})")
    
    # Also test what query was actually built
    filters = client.os_client.filters if hasattr(client.os_client, 'filters') else None
    if not filters:
        from src.infra.opensearch_client import SearchFilters
        filters = SearchFilters(acl_hash="grp_care")
    
    query_body = client._build_bm25_simple("credit score service", filters)
    
    print(f"\n📋 Generated query body:")
    import json
    print(json.dumps(query_body, indent=2))


if __name__ == "__main__":
    test_evaluation_client()