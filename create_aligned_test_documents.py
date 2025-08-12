#!/usr/bin/env python3
"""
Create test documents aligned with golden set ACL expectations and document IDs.

This ensures ACL filtering works correctly and queries return expected results.
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

# Test documents aligned with golden_set.yaml expectations
ALIGNED_DOCUMENTS = [
    # Service start documents (Q001, Q002, Q003, Q004)
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
        "acl_hash": "public",  # Q001 expects public
        "embedding": [0.1] * 1024
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
        "acl_hash": "grp_care",  # Q002 expects grp_care
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
        "acl_hash": "grp_ops",  # Q003 expects grp_ops
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
        "acl_hash": "grp_ops",  # Q004 expects grp_ops
        "embedding": [0.4] * 1024
    },
    
    # Reconnection documents (Q005, Q012)
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
        "acl_hash": "grp_care",  # Q012 expects grp_care
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
        "acl_hash": "grp_ops",  # Q005 expects grp_ops
        "embedding": [0.6] * 1024
    },
    
    # Payment plan documents (Q006, Q009, Q028)
    {
        "_id": "UTILS:BILLING:PAYMENT_PLAN:v2#overview",
        "title": "Payment Plan Overview",
        "section": "overview",
        "body": "Payment plans available for customers facing financial hardship. Extended payment terms up to 12 months. No setup fees for qualified customers.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "PAYMENT_PLAN_v2",
            "version": "v2"
        },
        "updated_at": (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "public",  # Q028 expects public (ACL boundary test)
        "embedding": [0.7] * 1024
    },
    {
        "_id": "UTILS:BILLING:PAYMENT_PLAN:v2#eligibility",
        "title": "Payment Plan Eligibility",
        "section": "eligibility", 
        "body": "Customer must demonstrate financial hardship. Income verification required. Medical emergencies qualify for expedited approval. Past payment history considered.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "PAYMENT_PLAN_v2",
            "version": "v2"
        },
        "updated_at": (datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "grp_care",  # Q006, Q009 expect grp_care
        "embedding": [0.8] * 1024
    },
    {
        "_id": "UTILS:BILLING:PAYMENT_PLAN:v1#hardship",
        "title": "Hardship Payment Plans",
        "section": "hardship",
        "body": "Special payment plan provisions for medical emergencies, job loss, and natural disasters. Reduced fees and extended terms available. Social services referrals provided.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "PAYMENT_PLAN_v1", 
            "version": "v1"
        },
        "updated_at": (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "grp_care",  # Q006 expects grp_care
        "embedding": [0.9] * 1024
    },
    
    # Stop service documents (Q008)
    {
        "_id": "UTILS:CUST:STOP_SERVICE:v3#overview",
        "title": "Stopping Utility Service", 
        "section": "overview",
        "body": "To stop utility service: 1) Call customer service or use online form, 2) Provide forwarding address, 3) Schedule final meter reading, 4) Pay final bill. 48-hour notice required.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "STOP_SERVICE_v3",
            "version": "v3"
        },
        "updated_at": (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "public",  # Q008 expects public
        "embedding": [1.0] * 1024
    },
    {
        "_id": "UTILS:CUST:STOP_SERVICE:v3#final_bill",
        "title": "Final Bill Processing",
        "section": "final_bill",
        "body": "Final bill includes prorated usage, connection fees, and security deposit refund (if applicable). Mailed within 10 business days of service disconnection.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "STOP_SERVICE_v3",
            "version": "v3"
        },
        "updated_at": (datetime.now() - timedelta(days=15)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "public",  # Q008 expects public
        "embedding": [0.95] * 1024
    },
    
    # Time-of-use rate documents (Q011, Q024)
    {
        "_id": "UTILS:RATE:TOU_SWITCH:v3#overview",
        "title": "Time-of-Use Rate Overview",
        "section": "overview",
        "body": "Time-of-use rates vary by time of day. Peak hours: 2-8pm weekdays. Off-peak and super off-peak rates available. Can save money for customers with flexible usage patterns.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "TOU_SWITCH_v3",
            "version": "v3"
        },
        "updated_at": (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d'),
        "content_type": "confluence", 
        "acl_hash": "public",  # Q011, Q024 expect public
        "embedding": [0.85] * 1024
    },
    {
        "_id": "UTILS:RATE:TOU_SWITCH:v3#eligibility",
        "title": "TOU Rate Eligibility",
        "section": "eligibility",
        "body": "Available to residential customers with compatible smart meters. Commercial rates have different schedules. Switch online or call customer service.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "TOU_SWITCH_v3", 
            "version": "v3"
        },
        "updated_at": (datetime.now() - timedelta(days=25)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "public",  # Q011, Q024 expect public
        "embedding": [0.8] * 1024
    },
    
    # Emergency/outage documents (Q021, Q022)
    {
        "_id": "UTILS:EMERGENCY:OUTAGE:v1#reporting", 
        "title": "Power Outage Reporting",
        "section": "reporting",
        "body": "Report power outages online, by phone, or mobile app. Provide account number and description of problem. Updates sent via text/email.",
        "metadata": {
            "space_key": "UTILS",
            "page_id": "OUTAGE_v1",
            "version": "v1"
        },
        "updated_at": (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "public",  # Q021 expects public
        "embedding": [0.75] * 1024
    },
    {
        "_id": "UTILS:EMERGENCY:OUTAGE:v1#restoration",
        "title": "Power Restoration Priority",
        "section": "restoration",
        "body": "Restoration priority: 1) Hospitals and emergency services, 2) Schools and critical infrastructure, 3) High-density residential, 4) Individual customers. Estimated times updated hourly.",
        "metadata": {
            "space_key": "UTILS", 
            "page_id": "OUTAGE_v1",
            "version": "v1"
        },
        "updated_at": (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
        "content_type": "confluence",
        "acl_hash": "grp_ops",  # Q022 expects grp_ops
        "embedding": [0.7] * 1024
    }
]


def delete_existing_documents():
    """Delete existing test documents to start fresh."""
    print("🗑️ Cleaning up existing test documents...")
    
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)
    
    try:
        # Delete all documents in index
        delete_url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_delete_by_query"
        delete_query = {"query": {"match_all": {}}}
        
        response = session.post(delete_url, json=delete_query, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        deleted_count = result.get('deleted', 0)
        print(f"✅ Deleted {deleted_count} existing documents")
        
    except Exception as e:
        print(f"⚠️  Could not delete existing documents: {e}")


def create_aligned_documents():
    """Create test documents aligned with golden set expectations."""
    print(f"🔧 Creating {len(ALIGNED_DOCUMENTS)} ACL-aligned documents...")
    
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)
    
    success_count = 0
    
    for doc in ALIGNED_DOCUMENTS:
        doc_id = doc.pop("_id")  # Remove _id from doc body
        
        try:
            # URL encode the document ID
            encoded_doc_id = quote_plus(doc_id)
            url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_doc/{encoded_doc_id}"
            response = session.put(url, json=doc, timeout=30)
            response.raise_for_status()
            
            success_count += 1
            acl = doc.get('acl_hash', 'none')
            print(f"✅ Created: {doc_id} (ACL: {acl})")
            
        except Exception as e:
            print(f"❌ Failed to create {doc_id}: {e}")
    
    print(f"\n📊 Created {success_count}/{len(ALIGNED_DOCUMENTS)} documents successfully")
    
    # Refresh index to make documents searchable
    try:
        refresh_url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_refresh"
        response = session.post(refresh_url, timeout=10)
        response.raise_for_status()
        print("✅ Index refreshed - documents ready for search")
    except Exception as e:
        print(f"⚠️  Index refresh failed: {e}")


def verify_acl_alignment():
    """Verify documents are correctly aligned with ACL expectations."""
    print(f"\n🔍 Verifying ACL alignment with golden set...")
    
    session = requests.Session()
    session.auth = (USERNAME, PASSWORD)
    
    # Check ACL distribution
    try:
        # Aggregation query to count documents by ACL
        agg_url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_search" 
        agg_query = {
            "size": 0,
            "aggs": {
                "acl_distribution": {
                    "terms": {
                        "field": "acl_hash",
                        "size": 10
                    }
                }
            }
        }
        
        response = session.post(agg_url, json=agg_query, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        print(f"📊 ACL Distribution:")
        for bucket in data["aggregations"]["acl_distribution"]["buckets"]:
            acl = bucket["key"]
            count = bucket["doc_count"]
            print(f"   • {acl}: {count} documents")
        
        # Test a specific query with ACL filter
        test_url = f"{OPENSEARCH_URL}/{INDEX_NAME}/_search"
        test_query = {
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {"query": "start utility service", "fields": ["title", "body"]}}
                    ],
                    "filter": [
                        {"term": {"acl_hash": "public"}}
                    ]
                }
            },
            "size": 5
        }
        
        response = session.post(test_url, json=test_query, timeout=10)
        response.raise_for_status()
        test_data = response.json()
        
        print(f"\n🔍 Test Query Results (ACL: public, query: 'start utility service'):")
        print(f"   Total hits: {test_data['hits']['total']['value']}")
        
        for hit in test_data['hits']['hits']:
            doc_id = hit['_id']
            acl = hit['_source'].get('acl_hash', 'none')
            title = hit['_source'].get('title', 'No title')
            print(f"   • {doc_id} (ACL: {acl})")
            print(f"     Title: {title}")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Setting up ACL-aligned test documents")
    print("=" * 60)
    
    delete_existing_documents()
    create_aligned_documents()
    
    if verify_acl_alignment():
        print("\n🎉 ACL-aligned document setup complete!")
        print("Ready to run A/B evaluation with proper ACL filtering")
        
        print(f"\n📋 Key Alignments:")
        print(f"   • Q001 'start utility service' → public ACL docs")
        print(f"   • Q002 'credit score' → grp_care ACL docs")
        print(f"   • Q003 'corporate deposits' → grp_ops ACL docs")
        print(f"   • Q028 'payment plan eligibility' → public ACL (boundary test)")
        print(f"   • Q029 'start service eligibility' → [] (ACL boundary test)")
    else:
        print("\n❌ Setup verification failed")