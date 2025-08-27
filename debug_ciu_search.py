#!/usr/bin/env python3
"""
Debug script to investigate CIU search accuracy issues.
Analyze what documents are actually being retrieved for CIU queries.
"""

import sys
import asyncio
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from infra.resource_manager import get_resources
from agent.tools.search import search_index_tool
from services.reranker import CrossEncodeReranker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def debug_ciu_search():
    """Debug what documents are retrieved for CIU query."""
    print("🔍 Debugging CIU Search Accuracy")
    print("=" * 50)
    
    try:
        # Get resources
        resources = get_resources()
        if not resources:
            print("❌ No resources available")
            return
        
        query = "Customer Interaction Utility"
        
        # Perform search
        print(f"📝 Query: '{query}'")
        print(f"🎯 Index: {resources.settings.search_index_alias}")
        
        results = await search_index_tool(
            index=resources.settings.search_index_alias,
            query=query,
            search_client=resources.search_client,
            embed_client=resources.embed_client,
            embed_model=resources.settings.embed.model,
            top_k=8,
            strategy="enhanced_rrf"
        )
        
        print(f"\n📊 Retrieved {len(results.results)} documents:")
        print("-" * 60)
        
        for i, doc in enumerate(results.results, 1):
            title = doc.meta.get("title", "No title")
            utility = doc.meta.get("utility_name", "Unknown utility")
            score = round(doc.score, 4)
            
            print(f"{i}. Score: {score}")
            print(f"   Title: {title}")
            print(f"   Utility: {utility}")
            print(f"   Content preview: {doc.text[:150]}...")
            print(f"   Doc ID: {doc.doc_id}")
            print()
            
            # Check if this looks like wrong content
            content_lower = doc.text.lower()
            if any(wrong in content_lower for wrong in ["product catalog", "pcu", "data field"]):
                print("   ⚠️  POTENTIAL ISSUE: Content may be wrong for CIU")
            elif "customer interaction" in content_lower:
                print("   ✅ Content seems relevant to CIU")
            else:
                print("   ❓ Unclear relevance to CIU")
            print("-" * 60)
        
        return results.results
        
    except Exception as e:
        logger.error(f"Debug search failed: {e}")
        return []

async def analyze_cross_encoder_scores(documents):
    """Analyze how cross-encoder scores the retrieved documents."""
    if not documents:
        return
        
    print("\n🎯 Cross-Encoder Reranking Analysis:")
    print("=" * 50)
    
    try:
        reranker = CrossEncodeReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cpu"
        )
        
        query = "Customer Interaction Utility"
        reranked = await reranker.rerank(query, documents)
        
        print(f"Reranked {len(reranked)} documents:")
        print("-" * 60)
        
        for i, doc in enumerate(reranked[:5], 1):  # Top 5
            title = doc.meta.get("title", "No title")
            score = round(doc.score, 4)
            print(f"{i}. Reranked Score: {score}")
            print(f"   Title: {title}")
            print(f"   Content: {doc.text[:100]}...")
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Reranker analysis failed: {e}")

async def main():
    """Run CIU search debugging."""
    documents = await debug_ciu_search()
    await analyze_cross_encoder_scores(documents)
    
    print("\n📋 SUMMARY:")
    print("- Check if retrieved documents actually describe CIU team/service")
    print("- Look for PCU contamination or generic content")
    print("- Verify document titles and utility_name metadata")
    print("- Cross-encoder scores should favor true CIU content")

if __name__ == "__main__":
    asyncio.run(main())