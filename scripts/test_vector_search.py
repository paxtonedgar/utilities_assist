#!/usr/bin/env python3
"""
Test script to verify FAISS vector search is working correctly.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set environment for local development
os.environ["UTILITIES_CONFIG"] = "config.local.ini"
os.environ["USE_LOCAL_AZURE"] = "true"

from mocks.vector_search import get_vector_search_client, get_status

def test_vector_search():
    """Test the vector search functionality."""
    print("🔍 Testing Mock Vector Search")
    print("=" * 50)
    
    # Get status
    status = get_status()
    print("📊 Vector Search Status:")
    for key, value in status.items():
        status_icon = "✅" if value else "❌"
        print(f"   {status_icon} {key}: {value}")
    
    if not status["ready"]:
        print("\n❌ Vector search not ready!")
        print("Possible solutions:")
        print("1. Run 'make embed-local' to generate embeddings")
        print("2. Check your config.local.ini Azure OpenAI settings")
        print("3. Ensure FAISS and dependencies are installed")
        return False
    
    print(f"\n🎯 Testing with {status['total_vectors']} vectors")
    
    # Test queries
    test_queries = [
        "How do I authenticate with the API?",
        "Kubernetes deployment configuration",
        "Streamlit widgets and components", 
        "Database indexing performance",
        "API rate limiting strategies"
    ]
    
    client = get_vector_search_client()
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔎 Test {i}: '{query}'")
        
        try:
            results = client.search(query, top_k=3)
            
            if results:
                print(f"   Found {len(results)} results:")
                for j, result in enumerate(results, 1):
                    score = result.get('score', 0)
                    title = result.get('title', 'Untitled')
                    doc_id = result.get('doc_id', 'unknown')
                    print(f"   {j}. [{doc_id}] {title} (score: {score:.3f})")
            else:
                print("   No results found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    print("\n" + "=" * 50)
    print("✅ Vector search test completed successfully!")
    return True

def main():
    """Main test function."""
    success = test_vector_search()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())