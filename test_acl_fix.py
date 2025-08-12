#!/usr/bin/env python3
"""
Test ACL filtering fix.
"""

from eval.os_eval_client import OpenSearchEvalClient

def test_acl_filtering():
    """Test that ACL filtering works correctly with public queries."""
    
    client = OpenSearchEvalClient()
    
    # Test public query - should only match public documents
    print("🔍 Testing public ACL query...")
    results = client.rank_docs(
        query_text="start utility service",
        mode="simple",
        acl_hash="public",  # Should explicitly filter for public documents
        index_alias="confluence_current"
    )
    
    print(f"📊 Results for public ACL:")
    for i, (doc_id, score) in enumerate(results):
        print(f"   {i+1}. {doc_id} (score: {score:.3f})")
    
    # Test grp_care query
    print(f"\n🔍 Testing grp_care ACL query...")  
    results = client.rank_docs(
        query_text="credit score service",
        mode="simple",
        acl_hash="grp_care",  # Should only match grp_care documents
        index_alias="confluence_current"
    )
    
    print(f"📊 Results for grp_care ACL:")
    for i, (doc_id, score) in enumerate(results):
        print(f"   {i+1}. {doc_id} (score: {score:.3f})")
    
    # Test grp_ops query
    print(f"\n🔍 Testing grp_ops ACL query...")
    results = client.rank_docs(
        query_text="deposits corporate accounts",
        mode="simple", 
        acl_hash="grp_ops",  # Should only match grp_ops documents
        index_alias="confluence_current"
    )
    
    print(f"📊 Results for grp_ops ACL:")
    for i, (doc_id, score) in enumerate(results):
        print(f"   {i+1}. {doc_id} (score: {score:.3f})")


if __name__ == "__main__":
    test_acl_filtering()