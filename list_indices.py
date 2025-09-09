#!/usr/bin/env python3
"""List all available indices and aliases in OpenSearch cluster."""

from src.infra.resource_manager import initialize_resources
from src.infra.settings import get_settings
from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
import requests
import json

def list_indices_and_aliases():
    """List all indices and aliases using the working resource manager authentication."""
    print("🔍 Listing OpenSearch indices and aliases...")
    
    # Initialize using same pattern as ontology module
    settings = get_settings()
    resources = initialize_resources(settings)
    
    # Use the same auth utilities that work
    _setup_jpmc_proxy()
    auth = _get_aws_auth()
    base_url = settings.opensearch_host.rstrip("/")
    
    print(f"📍 OpenSearch host: {base_url}")
    print(f"🔑 Using profile: {settings.cloud_profile}")
    print()
    
    # List all indices
    print("📊 INDICES:")
    print("-" * 80)
    try:
        response = requests.get(f"{base_url}/_cat/indices?format=json&s=index", 
                              auth=auth, timeout=30)
        if response.ok:
            indices = response.json()
            if indices:
                print(f"{'INDEX NAME':<40} {'DOCS':<10} {'SIZE':<10} {'STATUS':<10}")
                print("-" * 80)
                for idx in indices:
                    name = idx.get('index', 'N/A')
                    docs = idx.get('docs.count', '0')
                    size = idx.get('store.size', 'N/A')
                    status = idx.get('status', 'N/A')
                    print(f"{name:<40} {docs:<10} {size:<10} {status:<10}")
            else:
                print("No indices found")
        else:
            print(f"❌ Error listing indices: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception listing indices: {e}")
    
    print()
    
    # List all aliases  
    print("🏷️  ALIASES:")
    print("-" * 80)
    try:
        response = requests.get(f"{base_url}/_cat/aliases?format=json&s=alias", 
                              auth=auth, timeout=30)
        if response.ok:
            aliases = response.json()
            if aliases:
                print(f"{'ALIAS NAME':<30} {'INDEX NAME':<40}")
                print("-" * 80)
                for alias in aliases:
                    alias_name = alias.get('alias', 'N/A')
                    index_name = alias.get('index', 'N/A')
                    print(f"{alias_name:<30} {index_name:<40}")
            else:
                print("No aliases found")
        else:
            print(f"❌ Error listing aliases: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Exception listing aliases: {e}")
    
    print()
    print("✅ Index listing complete!")

if __name__ == "__main__":
    list_indices_and_aliases()