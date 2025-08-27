#!/usr/bin/env python3
"""
Simple debug script to check what CIU documents are being retrieved.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from infra.resource_manager import get_resources
from agent.tools.search import search_index_tool

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def simple_ciu_search():
    """Simple search to see what CIU documents are retrieved."""
    print("🔍 Simple CIU Search Debug")
    print("=" * 40)
    
    try:
        resources = get_resources()
        if not resources:
            print("❌ No resources available")
            return
        
        query = "Customer Interaction Utility"
        print(f"📝 Query: '{query}'")
        
        results = await search_index_tool(
            index=resources.settings.search_index_alias,
            query=query,
            search_client=resources.search_client,
            embed_client=resources.embed_client,
            embed_model=resources.settings.embed.model,
            top_k=5,  # Just top 5 for analysis
            strategy="enhanced_rrf"
        )
        
        print(f"\n📊 Top {len(results.results)} documents:")
        print("-" * 50)
        
        for i, doc in enumerate(results.results, 1):
            title = doc.meta.get("title", "No title")
            utility = doc.meta.get("utility_name", "Unknown")
            score = round(doc.score, 4)
            
            print(f"\n{i}. DOCUMENT ANALYSIS:")
            print(f"   Score: {score}")
            print(f"   Title: {title}")
            print(f"   Utility Name: {utility}")
            print(f"   Doc ID: {doc.doc_id}")
            print(f"   Content (first 200 chars):")
            print(f"   {doc.text[:200]}...")
            
            # Flag potential issues
            content_lower = doc.text.lower()
            if "product catalog" in content_lower or "pcu" in content_lower:
                print("   🚨 WARNING: This might be PCU content, not CIU!")
            elif "data field" in content_lower and "interaction" in content_lower:
                print("   ⚠️  ISSUE: CIU described as 'data field' - likely wrong!")
            elif "customer interaction utility" in content_lower:
                print("   ✅ Contains CIU keywords")
            else:
                print("   ❓ Unclear if this is relevant to CIU")
        
        print("\n" + "=" * 50)
        print("SUMMARY:")
        print("- Look for documents that incorrectly describe CIU as a 'data field'")
        print("- Check if PCU (Product Catalog Utility) content is contaminating results")
        print("- Verify utility_name metadata is correct")
        
    except Exception as e:
        logger.error(f"Search failed: {e}")

if __name__ == "__main__":
    asyncio.run(simple_ciu_search())