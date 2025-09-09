#!/usr/bin/env python3
"""Simple scroll API test without resource manager."""

from src.infra.settings import get_settings
from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
import requests

def simple_scroll_test():
    """Simple test of scroll API."""
    print("🔍 Simple scroll test...")
    
    settings = get_settings()
    _setup_jpmc_proxy()
    auth = _get_aws_auth()
    base_url = settings.opensearch_host.rstrip("/")
    index = "khub-opensearch-swagger-index"
    
    print(f"📍 Testing: {base_url}/{index}")
    
    # Test scroll
    try:
        body = {"size": 10, "sort": ["_doc"], "query": {"match_all": {}}}
        url = f"{base_url}/{index}/_search?scroll=2m"
        
        response = requests.post(
            url,
            json=body,
            auth=auth,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status: {response.status_code}")
        
        if response.ok:
            data = response.json()
            hits = data.get("hits", {}).get("hits", [])
            total = data.get("hits", {}).get("total", {})
            scroll_id = data.get("_scroll_id")
            
            print(f"✅ SUCCESS!")
            print(f"   - Got {len(hits)} docs in first batch")
            print(f"   - Total docs: {total}")
            print(f"   - Scroll ID: {'Yes' if scroll_id else 'No'}")
            
        else:
            print(f"❌ FAILED: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    simple_scroll_test()