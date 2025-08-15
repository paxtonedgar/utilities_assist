#!/usr/bin/env python3
"""
OpenSearch Discovery Audit Script

Purpose: Discover exactly what indices, mappings, and data exist in your OpenSearch cluster.
This will help identify why CIU searches return medical results instead of utilities.

Usage: python audit_opensearch.py
"""

import json
import os
import sys
import time
import requests
from requests_aws4auth import AWS4Auth
import boto3

# Add src to path for settings
sys.path.insert(0, 'src')

def load_settings():
    """Load OpenSearch endpoint and authentication from your settings."""
    # Check for command line argument first
    if len(sys.argv) > 1:
        endpoint = sys.argv[1]
        if not endpoint.startswith('http'):
            endpoint = f"https://{endpoint}"
        print(f"📌 Using endpoint from command line: {endpoint}")
        return {
            'endpoint': endpoint,
            'region': os.environ.get('AWS_REGION', 'us-east-1'),
            'search_index': os.environ.get('SEARCH_INDEX', 'khub-opensearch-index')
        }
    
    # Check environment variables
    env_endpoint = os.environ.get('OPENSEARCH_HOST') or os.environ.get('OPENSEARCH_ENDPOINT')
    if env_endpoint:
        if not env_endpoint.startswith('http'):
            env_endpoint = f"https://{env_endpoint}"
        print(f"📌 Using endpoint from environment: {env_endpoint}")
        return {
            'endpoint': env_endpoint,
            'region': os.environ.get('AWS_REGION', 'us-east-1'),
            'search_index': os.environ.get('SEARCH_INDEX', 'khub-opensearch-index')
        }
    
    # Try to load from application settings
    try:
        from src.infra.settings import get_settings
        settings = get_settings()
        
        endpoint = settings.opensearch_host
        if endpoint and endpoint != 'http://localhost:9200':
            if not endpoint.startswith('http'):
                endpoint = f"https://{endpoint}"
            print(f"📌 Using endpoint from settings: {endpoint}")
            return {
                'endpoint': endpoint,
                'region': getattr(settings.aws_info, 'aws_region', 'us-east-1') if settings.aws_info else 'us-east-1',
                'search_index': settings.search_index_alias
            }
    except Exception as e:
        print(f"⚠️  Could not load application settings: {e}")
    
    # Show usage and exit if no valid endpoint found
    print(f"❌ No OpenSearch endpoint configured!")
    print(f"")
    print(f"Usage options:")
    print(f"  1. Command line: python audit_opensearch.py <endpoint>")
    print(f"     Example: python audit_opensearch.py your-cluster.us-east-1.es.amazonaws.com")
    print(f"")
    print(f"  2. Environment: export OPENSEARCH_HOST=<endpoint>")
    print(f"     Example: export OPENSEARCH_HOST=your-cluster.us-east-1.es.amazonaws.com")
    print(f"")
    print(f"  3. Update config.local.ini [opensearch] endpoint = https://your-endpoint")
    print(f"")
    sys.exit(1)

# Load configuration
config = load_settings()
ENDPOINT = config['endpoint'].rstrip("/")  # Normalize endpoint (strip trailing slash)
REGION = config['region']
SEARCH_INDEX = config['search_index']

# Common index candidates to check
INDEX_CANDIDATES = [
    SEARCH_INDEX,                      # From your settings
    "khub-opensearch-index",           # Main content (common name)
    "khub-opensearch-swagger-index",   # Swagger/API specs
    "confluence-kb-index",             # Alternative confluence name
    "confluence_current",              # Another common pattern
    "swagger_index",                   # Alternative swagger name
]

# Optional: corp proxy (matches your service)
PROXIES = {
    "http": os.environ.get("HTTP_PROXY", ""),
    "https": os.environ.get("HTTPS_PROXY", "")
}
TIMEOUT = 30

def setup_session():
    """Setup authenticated requests session."""
    session = requests.Session()
    session.proxies.update({k: v for k, v in PROXIES.items() if v})
    
    try:
        # Use AWS credentials for authentication
        creds = boto3.Session().get_credentials().get_frozen_credentials()
        awsauth = AWS4Auth(creds.access_key, creds.secret_key, REGION, 'es', 
                          session_token=creds.token)
        session.auth = awsauth
        print(f"✅ AWS authentication configured for region: {REGION}")
    except Exception as e:
        print(f"⚠️  AWS authentication setup failed: {e}")
        print("   Proceeding without auth (may fail for secured endpoints)")
    
    return session

def safe_request(session, method, path, data=None, params=None):
    """Make a safe request with error handling."""
    try:
        if method.upper() == 'GET':
            r = session.get(f"{ENDPOINT}{path}", params=params, timeout=TIMEOUT)
        elif method.upper() == 'POST':
            headers = {"Content-Type": "application/json"} if data else {}
            r = session.post(f"{ENDPOINT}{path}", 
                           data=json.dumps(data) if data else None,
                           headers=headers, timeout=TIMEOUT)
        
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP {r.status_code} {method} {path}")
        if r.status_code == 404:
            print(f"   Resource not found: {path}")
        elif r.status_code == 403:
            print(f"   Access denied: {path}")
        else:
            print(f"   Error: {r.text[:200]}")
        return None
    except Exception as e:
        print(f"❌ Request failed {method} {path}: {e}")
        return None

def check_index_exists(session, index_name):
    """Check if an index exists and is accessible using HEAD (faster, fewer logs)."""
    try:
        r = session.head(f"{ENDPOINT}/{index_name}", timeout=TIMEOUT)
        return r.status_code == 200
    except Exception:
        return False

def pretty_print(title, data, max_length=3000):
    """Pretty print JSON data with length limits."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    if data is None:
        print("❌ No data available")
        return
    
    try:
        json_str = json.dumps(data, indent=2)
        if len(json_str) > max_length:
            json_str = json_str[:max_length] + f"\n... [truncated, {len(json_str)} total chars]"
        print(json_str)
    except:
        print(str(data)[:max_length])

def discover_vector_field(mapping_data):
    """Discover vector fields in the mapping."""
    vector_fields = []
    
    try:
        for index_name, index_info in mapping_data.items():
            properties = index_info.get("mappings", {}).get("properties", {})
            
            for field_name, field_info in properties.items():
                field_type = field_info.get("type", "")
                dimension = field_info.get("dimension", 0)
                
                if field_type in ("knn_vector", "dense_vector"):
                    vector_fields.append({
                        "field": field_name,
                        "type": field_type,
                        "dimension": dimension
                    })
                
                # Check nested properties too
                if "properties" in field_info:
                    for nested_field, nested_info in field_info["properties"].items():
                        nested_type = nested_info.get("type", "")
                        nested_dim = nested_info.get("dimension", 0)
                        
                        if nested_type in ("knn_vector", "dense_vector"):
                            vector_fields.append({
                                "field": f"{field_name}.{nested_field}",
                                "type": nested_type,
                                "dimension": nested_dim
                            })
    except Exception as e:
        print(f"⚠️  Error discovering vector fields: {e}")
    
    return vector_fields

def vector_space(settings_json):
    """Detect vector space type from index settings."""
    try:
        settings = list(settings_json.values())[0]["settings"]["index"]
        # OpenSearch 2.x: 'knn.space_type' OR per-field 'space_type' in mapping
        return settings.get("knn.space_type") or settings.get("space_type")
    except Exception:
        return None

def main():
    """Main discovery function."""
    print("\n" + "🔍" * 20 + " OPENSEARCH DISCOVERY AUDIT " + "🔍" * 20)
    print(f"Endpoint: {ENDPOINT}")
    print(f"Region: {REGION}")
    print(f"Configured Search Index: {SEARCH_INDEX}")
    
    session = setup_session()
    
    # Early auth check
    pong = safe_request(session, 'GET', '/')
    if pong is None:
        print("❌ Could not reach OpenSearch root with current auth/proxy. Fix this first.")
        sys.exit(2)
    
    # 1. Cluster overview
    print(f"\n{'🏥' * 60}")
    print("  CLUSTER HEALTH & OVERVIEW")
    print(f"{'🏥' * 60}")
    
    cluster_health = safe_request(session, 'GET', '/_cluster/health')
    pretty_print("Cluster Health", cluster_health, 1000)
    
    # 2. List all indices
    indices_info = safe_request(session, 'GET', '/_cat/indices?v&format=json')
    pretty_print("All Indices", indices_info, 2000)
    
    # 3. List all aliases
    aliases_info = safe_request(session, 'GET', '/_cat/aliases?v&format=json')
    pretty_print("All Aliases", aliases_info, 2000)
    
    # 4. Get full aliases mapping
    aliases_map = safe_request(session, 'GET', '/_aliases')
    pretty_print("Aliases Mapping", aliases_map, 2000)
    
    # 5. Check kNN plugin (if OpenSearch)
    knn_stats = safe_request(session, 'GET', '/_plugins/_knn/stats')
    pretty_print("kNN Plugin Stats", knn_stats, 1500)
    
    # 6. Inspect candidate indices
    print(f"\n{'📚' * 60}")
    print("  DETAILED INDEX INSPECTION")
    print(f"{'📚' * 60}")
    
    for idx in INDEX_CANDIDATES:
        print(f"\n🔍 Checking index: {idx}")
        
        if not check_index_exists(session, idx):
            print(f"❌ Index does not exist or is not accessible: {idx}")
            continue
        
        print(f"✅ Index exists: {idx}")
        
        # Get settings
        settings = safe_request(session, 'GET', f'/{idx}/_settings')
        pretty_print(f"{idx} Settings", settings, 2000)
        
        # Get mapping
        mapping = safe_request(session, 'GET', f'/{idx}/_mapping')
        pretty_print(f"{idx} Mapping", mapping, 3000)
        
        # Get document count
        count = safe_request(session, 'GET', f'/{idx}/_count')
        pretty_print(f"{idx} Document Count", count, 500)
        
        # Get field capabilities
        field_caps = safe_request(session, 'POST', f'/{idx}/_field_caps?fields=*', {})
        pretty_print(f"{idx} Field Capabilities", field_caps, 2000)
        
        # Sample documents (deterministic, avoid scoring)
        sample_docs = safe_request(session, 'POST', f'/{idx}/_search', {
            "size": 5,
            "sort": ["_doc"],
            "_source": ["title", "path", "url", "doc_type", "acronyms", "space", "h1", "h2", "content"],
            "query": {"match_all": {}}
        })
        pretty_print(f"{idx} Sample Documents", sample_docs, 2000)
        
        # Print concise field inventory
        if field_caps and "fields" in field_caps:
            fields = sorted(field_caps["fields"].keys())
            pretty_print(f"{idx} Field Inventory (count={len(fields)})", fields[:300], 2000)
        
        # Test CIU search (BM25) - robust field checking
        print(f"\n🎯 Testing CIU Search on {idx}")
        
        # Build robust CIU search that doesn't assume .keyword exists
        ciu_should = [
            {"term": {"acronyms.keyword": "CIU"}},
            {"term": {"acronyms.keyword": "ciu"}},
            {"term": {"acronyms": "CIU"}},                 # if not keyworded
            {"match_phrase": {"title": "Customer Interaction Utility"}},
            {"match": {"title": {"query": "CIU", "boost": 3}}},
            {"match": {"content": {"query": "Customer Interaction Utility", "boost": 2}}}
        ]
        
        # Add wildcard only if title.keyword exists (discovered via field_caps)
        title_has_keyword = False
        if field_caps and isinstance(field_caps, dict) and "fields" in field_caps:
            title_has_keyword = "title.keyword" in field_caps["fields"]
        
        if title_has_keyword:
            ciu_should.append({"wildcard": {"title.keyword": "*CIU*"}})
        
        ciu_search = safe_request(session, 'POST', f'/{idx}/_search', {
            "size": 5,
            "_source": ["title", "path", "url", "acronyms", "space", "utility_name"],
            "query": {
                "bool": {
                    "should": ciu_should,
                    "minimum_should_match": 1
                }
            }
        })
        pretty_print(f"{idx} CIU Search Results", ciu_search, 2000)
        
        # Test vector search if available (guard against cosine similarity)
        if mapping:
            vector_fields = discover_vector_field(mapping)
            
            # Get vector space type from settings
            idx_settings = settings or safe_request(session, 'GET', f'/{idx}/_settings') or {}
            space = vector_space(idx_settings)
            
            if vector_fields:
                print(f"\n🧮 Found vector fields in {idx}: {vector_fields} (space: {space or 'unknown'})")
                
                # Only probe if space type is safe for dummy vectors
                if (space or "").lower() in ("l2", "euclidean", "dotproduct", "ip", "innerproduct"):
                    probe_val = 1.0 if (space or "").lower() in ("dotproduct", "ip", "innerproduct") else 0.0
                    test_dim = vector_fields[0]["dimension"] or 1536
                    vec = [probe_val] * test_dim
                    
                    knn_search = safe_request(session, 'POST', f'/{idx}/_search', {
                        "size": 3,
                        "_source": ["title", "path", "url"],
                        "knn": {
                            "field": vector_fields[0]["field"],
                            "query_vector": vec,
                            "k": 3,
                            "num_candidates": 100
                        }
                    })
                    pretty_print(f"{idx} kNN Test on {vector_fields[0]['field']}", knn_search, 1500)
                else:
                    print("ℹ️  Cosine/unknown space detected; skipping dummy-vector probe to avoid invalid-vector errors.")
            else:
                print(f"ℹ️  No vector fields found in {idx}")
    
    # 7. Summary and recommendations
    print(f"\n{'🎯' * 60}")
    print("  DISCOVERY SUMMARY & RECOMMENDATIONS")
    print(f"{'🎯' * 60}")
    
    print("\n📋 Key Findings:")
    print(f"   • Configured search index: {SEARCH_INDEX}")
    print(f"   • Endpoint accessible: {ENDPOINT}")
    
    if indices_info:
        valid_indices = [idx for idx in INDEX_CANDIDATES if check_index_exists(session, idx)]
        print(f"   • Valid indices found: {valid_indices}")
        
        missing_indices = [idx for idx in INDEX_CANDIDATES if not check_index_exists(session, idx)]
        if missing_indices:
            print(f"   • Missing indices (causing 404s): {missing_indices}")
    
    print("\n🔧 Next Steps:")
    print("   1. Review the mapping structures above")
    print("   2. Check which indices contain CIU vs medical content")
    print("   3. Update routing to avoid missing indices")
    print("   4. Adjust boosting based on actual field names")
    
    print(f"\n{'✅' * 60}")
    print("  DISCOVERY AUDIT COMPLETE")
    print(f"{'✅' * 60}")

if __name__ == "__main__":
    main()