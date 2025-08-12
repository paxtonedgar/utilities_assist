#!/usr/bin/env python3
"""
Create test documents for A/B evaluation testing.

Creates documents that match the expected_doc_ids from the golden set
to enable realistic evaluation of BM25 simple vs tuned modes.
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
from urllib.parse import quote_plus

# OpenSearch configuration
OPENSEARCH_URL = "http://localhost:9200"
USERNAME = "admin"
PASSWORD = "admin"
INDEX_NAME = "confluence_current"

# Test documents matching golden set expected_doc_ids
TEST_DOCUMENTS = [
    # Service start documents
    {
        "_id": "UTILS:CUST:START_SERVICE:v4#overview",
        "title": "Starting New Utility Service - Overview",
        "section": "overview",
        "body": "To start new utility service, customers need valid ID, proof of residence, and credit check. Online application available 24/7. Most applications processed within 24 hours.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "START_SERVICE_v4",
            "version": "v4"
        },
        "updated_at": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "public",
        "embedding": [0.1] * 1024  # Dummy embedding (1024 dims per mapping)
    },
    {
        "_id": "UTILS:CUST:START_SERVICE:v4#eligibility",
        "title": "Service Eligibility Requirements",
        "section": "eligibility", 
        "body": "Credit score above 650 required for new service. Customers with lower scores may need security deposit. Corporate accounts have different requirements.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "START_SERVICE_v4",
            "version": "v4"
        },
        "updated_at": (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "grp_care",
        "embedding": [0.2] * 1024
    },
    {
        "_id": "UTILS:CUST:START_SERVICE:v1#eligibility",
        "title": "Legacy Service Eligibility",
        "section": "eligibility",
        "body": "Security deposits required for credit scores below 700. Corporate accounts are always exempt from deposits regardless of credit score. Deposit calculation based on average monthly usage.",
        "metadata": {
            "space_key": "UTILS", 
            "page_id": "START_SERVICE_v1",
            "version": "v1"
        },
        "updated_at": (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "grp_ops",
        "embedding": [0.3] * 1024
    },
    {
        "_id": "UTILS:CUST:START_SERVICE:v1#deposits",
        "title": "Deposit Calculation Methodology",
        "section": "deposits",
        "body": "Deposit = (Average monthly usage × 2.5) + base fee $50. Usage calculated from similar properties in same zip code. Corporate accounts exempt. Refunded after 12 months good payment history.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "START_SERVICE_v1", 
            "version": "v1"
        },
        "updated_at": (datetime.now() - timedelta(days=400)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "grp_ops",
        "embedding": [0.4] * 1024
    },
    
    # Reconnection documents
    {
        "_id": "UTILS:CUST:RECONNECT:v2#process",
        "title": "Service Reconnection Process",
        "section": "process",
        "body": "To reconnect service: 1) Pay outstanding balance, 2) Pay reconnection fee, 3) Schedule appointment online or by phone. Same-day reconnection available for additional fee.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "RECONNECT_v2",
            "version": "v2"
        },
        "updated_at": (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "grp_ops",
        "embedding": [0.5] * 1024
    },
    {
        "_id": "UTILS:CUST:RECONNECT:v1#fees",
        "title": "Reconnection Fees and Charges",
        "section": "fees",
        "body": "Standard reconnection fee: $45. Same-day service: additional $75. After hours (weekends/holidays): additional $100. Late payment surcharge: 1.5% per month on outstanding balance.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "RECONNECT_v1", 
            "version": "v1"
        },
        "updated_at": (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "grp_ops",
        "embedding": [0.6] * 1024
    },
    
    # Outage documents
    {
        "_id": "UTILS:OPS:OUTAGES:v3#emergency",
        "title": "Emergency Outage Response",
        "section": "emergency",
        "body": "Emergency outages require immediate escalation to on-call manager. Customer communications via automated system. Estimated restoration time updated every 30 minutes.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "OUTAGES_v3",
            "version": "v3"
        },
        "updated_at": (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
        "content_type": "confluence", 
        "acl_hash": "grp_ops",
        "embedding": [0.7] * 1024
    },
    {
        "_id": "UTILS:OPS:OUTAGES:v2#planned",
        "title": "Planned Maintenance Procedures", 
        "section": "planned",
        "body": "Planned maintenance notifications sent 48 hours in advance. Maintenance windows: Tuesday-Thursday 2-6am. Customer portal updated with real-time progress.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "OUTAGES_v2",
            "version": "v2"
        },
        "updated_at": (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "grp_ops", 
        "embedding": [0.8] * 1024
    },
    
    # Generic/low-quality documents (should rank lower)
    {
        "_id": "UTILS:GEN:PLATFORM#overview",
        "title": "Platform Overview",
        "section": "overview", 
        "body": "This is a general overview of our platform. The platform provides various services. Please see specific documentation for details.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "PLATFORM",
            "version": "current"
        },
        "updated_at": (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "public",
        "embedding": [0.9] * 1024
    },
    {
        "_id": "UTILS:GEN:WELCOME#intro",
        "title": "Welcome to Utilities",
        "section": "introduction",
        "body": "Welcome to our utilities platform. We provide excellent service. This introduction covers general information about our company.",
        "metadata": {
            "space_key": "UTILS", 
            "page_id": "WELCOME",
            "version": "current"
        },
        "updated_at": (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "public",
        "embedding": [1.0] * 1024
    }
]


def create_documents():
    """Create test documents in OpenSearch index."""
    print(f"🔧 Creating {len(TEST_DOCUMENTS)} test documents in {INDEX_NAME}...")
    
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)
    
    success_count = 0
    
    for doc in TEST_DOCUMENTS:
        doc_id = doc.pop("_id")  # Remove _id from doc body
        
        try:
            # URL encode the document ID to handle special characters
            encoded_doc_id = quote_plus(doc_id)
            url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_doc/{encoded_doc_id}"
            response = session.put(url, json=doc, timeout=30)
            response.raise_for_status()
            
            success_count += 1
            print(f"✅ Created: {doc_id}")
            
        except Exception as e:
            print(f"❌ Failed to create {doc_id}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"   Error details: {error_detail}")
                except:
                    print(f"   Response text: {e.response.text}")
                    pass
    
    print(f"\n📊 Created {success_count}/{len(TEST_DOCUMENTS)} documents successfully")
    
    # Refresh index to make documents searchable
    try:
        refresh_url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_refresh"
        response = session.post(refresh_url, timeout=10)
        response.raise_for_status()
        print("✅ Index refreshed - documents ready for search")
    except Exception as e:
        print(f"⚠️  Index refresh failed: {e}")


def verify_documents():
    """Verify documents were created successfully."""
    print(f"\n🔍 Verifying documents in {INDEX_NAME}...")
    
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)
    
    try:
        # Count total documents
        count_url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_count"
        response = session.get(count_url, timeout=10)
        response.raise_for_status()
        count_data = response.json()
        
        print(f"📊 Total documents in index: {count_data['count']}")
        
        # Test search functionality
        search_url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_search"
        search_query = {
            "query": {"match": {"title": "service"}},
            "size": 5
        }
        
        response = session.post(search_url, json=search_query, timeout=10)
        response.raise_for_status()
        search_data = response.json()
        
        print(f"🔍 Test search found {search_data['hits']['total']['value']} documents")
        
        if search_data['hits']['hits']:
            first_doc = search_data['hits']['hits'][0]
            print(f"📄 Example document: {first_doc['_id']}")
            print(f"   Title: {first_doc['_source'].get('title', 'N/A')}")
            print(f"   Score: {first_doc['_score']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Setting up test documents for A/B evaluation")
    print("=" * 60)
    
    create_documents()
    
    if verify_documents():
        print("\n🎉 Test document setup complete!")
        print("Ready to run A/B evaluation with live OpenSearch data")
    else:
        print("\n❌ Setup verification failed")