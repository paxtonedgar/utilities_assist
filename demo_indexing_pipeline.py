#!/usr/bin/env python3
"""
Demo script for the unified indexing pipeline.

Shows:
- Blue/green reindex with proper versioning
- Embedding dimension validation (1536)
- Configurable embedding providers (mock/JPMC)
- ACL filtering and time-decay consistency
- Document validation and error handling
"""

import asyncio
import os
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.infra.indexing_pipeline import (
    Document, 
    UnifiedIndexingPipeline,
    create_embedding_provider
)
from src.infra.opensearch_client import OpenSearchClient, SearchFilters
from src.infra.config import SearchCfg


def create_sample_documents() -> list[Document]:
    """Create sample documents for indexing demo."""
    now = datetime.now()
    
    return [
        Document(
            doc_id="UTILS:START:v3#overview",
            title="Starting Utility Service - Latest Guide",
            body="Complete guide for starting new utility service with updated 2023 procedures. Includes credit check requirements, deposit calculations, and online application process.",
            section="overview",
            metadata={
                "space_key": "UTILS",
                "page_id": "START_SERVICE_v3",
                "version": "v3",
                "category": "workflow"
            },
            updated_at=now - timedelta(days=5),  # Very recent
            content_type="confluence",
            acl_hash="grp_customer_service",
            canonical_id="START_SERVICE_OVERVIEW",
            space_key="UTILS"
        ),
        Document(
            doc_id="UTILS:START:v2#process",
            title="Service Start Process Steps",
            body="Step by step process for customer service representatives to initiate new utility service. Covers eligibility verification and documentation requirements.",
            section="process",
            metadata={
                "space_key": "UTILS",
                "page_id": "START_SERVICE_v2", 
                "version": "v2",
                "category": "workflow"
            },
            updated_at=now - timedelta(days=45),  # Older but relevant
            content_type="confluence",
            acl_hash="grp_customer_service",
            canonical_id="START_SERVICE_PROCESS",
            space_key="UTILS"
        ),
        Document(
            doc_id="UTILS:OPS:RESTRICTED#procedures",
            title="Internal Operations Procedures",
            body="Restricted internal procedures for operations team. Contains sensitive billing and system access protocols.",
            section="procedures",
            metadata={
                "space_key": "UTILS",
                "page_id": "OPS_PROCEDURES",
                "version": "current",
                "category": "operations",
                "priority": "high"
            },
            updated_at=now - timedelta(days=20),
            content_type="confluence", 
            acl_hash="grp_operations_restricted",  # Different ACL
            canonical_id="OPS_PROCEDURES",
            space_key="UTILS"
        ),
        Document(
            doc_id="UTILS:GEN:PLATFORM#welcome",
            title="Welcome to Platform Overview",
            body="General welcome message and platform introduction. This provides a high-level overview of our platform capabilities.",
            section="introduction",
            metadata={
                "space_key": "UTILS",
                "page_id": "PLATFORM_WELCOME",
                "version": "current"
            },
            updated_at=now - timedelta(days=2),  # Recent but generic
            content_type="confluence",
            acl_hash="public",
            canonical_id="PLATFORM_OVERVIEW",
            space_key="UTILS"
        )
    ]


async def demo_embedding_providers():
    """Demo different embedding providers."""
    print("\n🔌 EMBEDDING PROVIDERS DEMO")
    print("=" * 50)
    
    # Test mock provider (default)
    os.environ["EMBEDDING_PROVIDER"] = "mock"
    mock_provider = create_embedding_provider()
    
    test_texts = ["Customer service guide", "Utility procedures"]
    embeddings = await mock_provider.create_embeddings(test_texts)
    
    print(f"🟢 Mock Provider:")
    print(f"   Created {len(embeddings)} embeddings")
    print(f"   Dimensions: {len(embeddings[0])} (should be 1536)")
    print(f"   Sample values: {embeddings[0][:3]}...")
    
    # Show JPMC provider configuration (without actually calling it)
    print(f"\n🏢 JPMC Provider Configuration:")
    print(f"   Set EMBEDDING_PROVIDER=jpmc")
    print(f"   Set JPMC_EMBEDDING_API_KEY=your-api-key")
    print(f"   Set JPMC_EMBEDDING_BASE_URL=https://api.jpmc.internal")
    print(f"   Provider will validate 1536 dimensions automatically")


async def demo_document_validation():
    """Demo document validation."""
    print("\n📋 DOCUMENT VALIDATION DEMO")
    print("=" * 50)
    
    # Valid document
    valid_doc = Document(
        doc_id="VALID:DOC",
        title="Valid Document",
        body="This document has all required fields.",
        section="test",
        metadata={"version": "v1"},
        updated_at=datetime.now(),
        acl_hash="public"
    )
    
    errors = valid_doc.validate()
    print(f"✅ Valid document: {len(errors)} errors")
    
    # Invalid document
    invalid_doc = Document(
        doc_id="",  # Empty ID
        title="",   # Empty title
        body="",    # Empty body
        section="test",
        metadata="not_dict",  # Wrong type
        updated_at="not_date",  # Wrong type
    )
    
    errors = invalid_doc.validate()
    print(f"❌ Invalid document: {len(errors)} errors")
    for error in errors[:3]:  # Show first 3 errors
        print(f"   • {error}")


async def demo_blue_green_indexing():
    """Demo blue/green indexing with versioning."""
    print("\n🔄 BLUE/GREEN INDEXING DEMO")
    print("=" * 50)
    
    # Mock search configuration
    config = SearchCfg(
        host="http://localhost:9200",
        username="admin", 
        password="admin",
        timeout_s=30
    )
    
    # Create sample documents
    documents = create_sample_documents()
    print(f"📄 Prepared {len(documents)} documents for indexing")
    
    # Show what would happen (don't actually index to avoid OpenSearch dependency)
    print(f"\n🔧 Blue/Green Process:")
    print(f"   1. Current alias: utilities_current")
    print(f"   2. Create new index: utilities_v3 (next version)")
    print(f"   3. Index {len(documents)} documents with 1536-dim embeddings")
    print(f"   4. Switch alias: utilities_current -> utilities_v3")
    print(f"   5. Old index utilities_v2 remains for rollback")
    
    print(f"\n📊 Document ACL Distribution:")
    acl_counts = {}
    for doc in documents:
        acl = doc.acl_hash or "none"
        acl_counts[acl] = acl_counts.get(acl, 0) + 1
    
    for acl, count in acl_counts.items():
        print(f"   • {acl}: {count} documents")


def demo_acl_filtering():
    """Demo ACL filtering and time decay."""
    print("\n🔐 ACL FILTERING & TIME DECAY DEMO")
    print("=" * 50)
    
    documents = create_sample_documents()
    
    print("📅 Document Time Distribution:")
    for doc in documents:
        days_old = (datetime.now() - doc.updated_at).days
        print(f"   • {doc.doc_id}: {days_old} days old (ACL: {doc.acl_hash})")
    
    print(f"\n🔍 Sample Queries with ACL Filtering:")
    
    # Customer service query - should match grp_customer_service
    print(f"   Query: 'How to start utility service'")
    print(f"   ACL Filter: grp_customer_service")
    print(f"   Expected Results:")
    print(f"     1. UTILS:START:v3#overview (5 days old, higher time decay score)")
    print(f"     2. UTILS:START:v2#process (45 days old, lower time decay score)")
    print(f"   Filtered Out: UTILS:OPS:RESTRICTED (different ACL)")
    
    # Operations query - should match grp_operations_restricted  
    print(f"\n   Query: 'internal operations procedures'")
    print(f"   ACL Filter: grp_operations_restricted")
    print(f"   Expected Results:")
    print(f"     1. UTILS:OPS:RESTRICTED#procedures (20 days old)")
    print(f"   Filtered Out: Customer service docs (different ACL)")
    
    print(f"\n⚡ Time Decay Function (Applied to Both BM25 and kNN):")
    print(f"   • Scale: 75d half-life")
    print(f"   • Decay: 0.4 factor")
    print(f"   • Weight: 1.2x multiplier")
    print(f"   • Generic penalty: -1.2 for overview/welcome pages")


async def demo_embedding_dimension_enforcement():
    """Demo embedding dimension validation."""
    print("\n📏 EMBEDDING DIMENSION VALIDATION")
    print("=" * 50)
    
    doc = create_sample_documents()[0]
    
    # Test correct dimensions
    correct_embedding = [0.1] * 1536
    try:
        opensearch_doc = doc.to_opensearch_doc(correct_embedding)
        print(f"✅ 1536 dimensions: SUCCESS")
        print(f"   Document ready for indexing")
    except ValueError as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test wrong dimensions
    wrong_embedding = [0.1] * 512  # Wrong dimensions
    try:
        opensearch_doc = doc.to_opensearch_doc(wrong_embedding)
        print(f"❌ Should have failed with wrong dimensions")
    except ValueError as e:
        print(f"✅ 512 dimensions: REJECTED")
        print(f"   Error: {e}")
    
    print(f"\n🎯 OpenSearch Mapping Enforcement:")
    print(f"   • Index mapping: dimension=1536")
    print(f"   • Validation: Pre-index dimension check")
    print(f"   • Error handling: Clear error messages")
    print(f"   • Consistency: Same dims for BM25 and kNN")


async def main():
    """Run all demos."""
    print("🚀 UNIFIED INDEXING PIPELINE DEMO")
    print("=" * 60)
    print("Features:")
    print("• Blue/green reindex with versioned indices")
    print("• Embedding dimension validation (1536)")  
    print("• Configurable providers (mock/JPMC)")
    print("• Uniform ACL filtering & time-decay")
    print("• Comprehensive error handling")
    
    await demo_embedding_providers()
    await demo_document_validation()
    await demo_blue_green_indexing()
    demo_acl_filtering()
    await demo_embedding_dimension_enforcement()
    
    print("\n" + "=" * 60)
    print("🎉 DEMO COMPLETE!")
    print("\nKey Benefits:")
    print("✅ Zero-downtime deployments with blue/green")
    print("✅ Embedding dimension mismatch prevention") 
    print("✅ Environment-based provider configuration")
    print("✅ Consistent ACL filtering across BM25 and kNN")
    print("✅ Uniform time-decay scoring")
    print("✅ Comprehensive validation and error handling")
    
    print(f"\n📖 Usage:")
    print(f"export EMBEDDING_PROVIDER=mock  # or jpmc")
    print(f"export JPMC_EMBEDDING_API_KEY=your-key  # if using jpmc")
    print(f"python -m src.infra.indexing_pipeline")


if __name__ == "__main__":
    asyncio.run(main())