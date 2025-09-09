#!/usr/bin/env python3
"""Test scroll using the working resource manager approach."""

from src.infra.resource_manager import get_cached_resources
from src.infra.settings import get_settings
from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
import requests

def test_scroll_with_working_setup():
    """Test scroll using the same approach that works in ontology pipeline."""
    print("🔍 Testing scroll with working resource manager setup...")
    
    # Use the same setup that works in build_queue
    settings = get_settings()
    _setup_jpmc_proxy()
    auth = _get_aws_auth()
    base_url = settings.opensearch_host.rstrip("/")
    index = "khub-opensearch-swagger-index"
    
    print(f"📍 Host: {base_url}")
    print(f"📍 Index: {index}")
    
    # Test scroll exactly like the working iterate_ids method
    try:
        body = {"size": 50, "sort": ["_doc"], "query": {"match_all": {}}}
        url = f"{base_url}/{index}/_search?scroll=2m"
        headers = {"Content-Type": "application/json"}
        
        print(f"🔗 Request URL: {url}")
        
        response = requests.post(url, json=body, auth=auth, timeout=30, headers=headers)
        
        print(f"📊 Status: {response.status_code}")
        
        if response.ok:
            data = response.json()
            hits = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", {})
            scroll_id = data.get("_scroll_id")
            
            if isinstance(total, dict):
                total_count = total.get("value", 0)
            else:
                total_count = total
            
            print(f"✅ SUCCESS!")
            print(f"   📄 First batch: {len(hits)} documents")
            print(f"   📊 Total available: {total_count} documents")
            print(f"   🔄 Scroll ID: {'Present' if scroll_id else 'Missing'}")
            
            # Test scroll continuation if we have a scroll_id
            if scroll_id and len(hits) > 0:
                print(f"\n🔄 Testing scroll continuation...")
                scroll_url = f"{base_url}/_search/scroll"
                scroll_body = {"scroll": "2m", "scroll_id": scroll_id}
                
                scroll_response = requests.post(scroll_url, json=scroll_body, auth=auth, timeout=30, headers=headers)
                
                if scroll_response.ok:
                    scroll_data = scroll_response.json()
                    scroll_hits = scroll_data.get("hits", {}).get("hits", [])
                    print(f"   ✅ Continuation: {len(scroll_hits)} more documents")
                    
                    if len(scroll_hits) == 0:
                        print("   ⚠️  No more documents (expected for small index)")
                else:
                    print(f"   ❌ Continuation failed: {scroll_response.status_code}")
                    print(f"   Response: {scroll_response.text[:100]}...")
                    
        else:
            print(f"❌ FAILED: {response.status_code}")
            print(f"Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ EXCEPTION: {e}")

if __name__ == "__main__":
    test_scroll_with_working_setup()