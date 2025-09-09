#!/usr/bin/env python3
"""Test if OpenSearch cluster supports scroll API."""

from src.infra.resource_manager import initialize_resources
from src.infra.settings import get_settings
from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
import requests
import json

def test_scroll_support():
    """Test scroll API on your working index."""
    print("🔍 Testing scroll API support...")
    
    # Use the same working pattern as ontology pipeline
    settings = get_settings()
    _setup_jpmc_proxy()
    auth = _get_aws_auth()
    base_url = settings.opensearch_host.rstrip("/")
    
    # Test on your working index
    index = "khub-opensearch-swagger-index"
    
    print(f"📍 Testing index: {index}")
    print(f"🔗 OpenSearch host: {base_url}")
    
    # Test 1: Try scroll API
    print("\n📜 TEST 1: Scroll API")
    try:
        scroll_body = {
            "size": 5,
            "sort": ["_doc"],
            "query": {"match_all": {}}
        }
        
        response = requests.post(
            f"{base_url}/{index}/_search?scroll=2m",
            json=scroll_body,
            auth=auth,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.ok:
            data = response.json()
            scroll_id = data.get("_scroll_id")
            hits_count = len(data.get("hits", {}).get("hits", []))
            total = data.get("hits", {}).get("total", {})
            
            if isinstance(total, dict):
                total_docs = total.get("value", 0)
            else:
                total_docs = total
                
            print(f"✅ Scroll API works!")
            print(f"   - Got scroll_id: {scroll_id[:20]}..." if scroll_id else "   - No scroll_id")
            print(f"   - First batch: {hits_count} docs")
            print(f"   - Total docs: {total_docs}")
            
            # Test scroll continuation
            if scroll_id:
                print("\n📜 TEST 2: Scroll Continuation")
                scroll_response = requests.post(
                    f"{base_url}/_search/scroll",
                    json={"scroll": "2m", "scroll_id": scroll_id},
                    auth=auth,
                    timeout=30,
                    headers={"Content-Type": "application/json"}
                )
                
                if scroll_response.ok:
                    scroll_data = scroll_response.json()
                    scroll_hits = len(scroll_data.get("hits", {}).get("hits", []))
                    print(f"✅ Scroll continuation works! Got {scroll_hits} more docs")
                else:
                    print(f"❌ Scroll continuation failed: {scroll_response.status_code}")
                    print(f"   Response: {scroll_response.text[:200]}...")
            
        else:
            print(f"❌ Scroll API failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except Exception as e:
        print(f"❌ Scroll API exception: {e}")
    
    # Test 3: Regular search_after pagination
    print("\n🔍 TEST 3: search_after pagination")
    try:
        search_body = {
            "size": 10,
            "sort": [{"_id": "asc"}],
            "query": {"match_all": {}}
        }
        
        response = requests.post(
            f"{base_url}/{index}/_search",
            json=search_body,
            auth=auth,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        if response.ok:
            data = response.json()
            hits = data.get("hits", {}).get("hits", [])
            print(f"✅ search_after pagination works! Got {len(hits)} docs")
            
            if hits:
                last_sort = hits[-1].get("sort")
                print(f"   - Last sort value: {last_sort}")
        else:
            print(f"❌ search_after failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ search_after exception: {e}")
    
    # Test 4: Check cluster settings
    print("\n⚙️ TEST 4: Cluster Settings")
    try:
        response = requests.get(
            f"{base_url}/_cluster/settings",
            auth=auth,
            timeout=30
        )
        
        if response.ok:
            settings_data = response.json()
            print("✅ Can read cluster settings")
            
            # Look for scroll-related settings
            persistent = settings_data.get("persistent", {})
            transient = settings_data.get("transient", {})
            
            scroll_settings = {}
            for section_name, section in [("persistent", persistent), ("transient", transient)]:
                for key, value in section.items():
                    if "scroll" in key.lower():
                        scroll_settings[f"{section_name}.{key}"] = value
            
            if scroll_settings:
                print("   Scroll-related settings:")
                for key, value in scroll_settings.items():
                    print(f"   - {key}: {value}")
            else:
                print("   No explicit scroll settings found (likely using defaults)")
                
        else:
            print(f"❌ Cannot read cluster settings: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Cluster settings exception: {e}")

if __name__ == "__main__":
    test_scroll_support()