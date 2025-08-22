#!/usr/bin/env python3
"""
Integration test for search→enhance→combine pipeline.
Tests the complete flow to catch AttributeError and other issues.
"""

import asyncio
import logging
from typing import List

from src.services.models import SearchResult
from src.agent.nodes.intent import intent_node
from src.agent.nodes.combine import combine_node
from src.retrieval.actionability import detect_spans, ViewResult
from src.retrieval.views import run_info_view

# Configure minimal logging for test
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_search_results() -> List[SearchResult]:
    """Create realistic test SearchResult objects."""
    return [
        SearchResult(
            doc_id="doc-1",
            title="Customer Interaction Utility Overview",
            url="https://internal.docs/ciu-overview",
            score=0.85,
            content="The Customer Interaction Utility (CIU) is a comprehensive platform for managing customer communications. It provides APIs for creating tickets, tracking issues, and generating reports.",
            metadata={
                "title": "Customer Interaction Utility Overview",
                "section": "Overview",
                "utility_name": "CIU",
                "api_names": ["create_ticket", "track_issue"]
            }
        ),
        SearchResult(
            doc_id="doc-2", 
            title="CIU API Setup Guide",
            url="https://internal.docs/ciu-setup",
            score=0.75,
            content="To set up CIU API access: 1. Register your application 2. Generate API keys 3. Configure authentication headers 4. Test with sample requests",
            metadata={
                "title": "CIU API Setup Guide",
                "section": "Setup",
                "utility_name": "CIU",
                "section_paths": ["Setup", "Authentication", "Testing"]
            }
        ),
        SearchResult(
            doc_id="doc-3",
            title="CIU Troubleshooting",
            url="https://internal.docs/ciu-troubleshoot", 
            score=0.65,
            content="Common CIU issues: Authentication failures, rate limiting, timeout errors. Check API key validity and retry logic.",
            metadata={
                "title": "CIU Troubleshooting",
                "section": "Troubleshooting",
                "utility_name": "CIU"
            }
        )
    ]


async def test_intent_classification():
    """Test intent classification works without errors."""
    print("🧪 Testing intent classification...")
    
    test_state = {
        'normalized_query': 'what is Customer Interaction Utility',
        'original_query': 'what is Customer Interaction Utility'
    }
    config = {'configurable': {}}
    
    result = await intent_node(test_state, config)
    
    assert 'intent' in result
    intent_obj = result['intent']
    assert hasattr(intent_obj, 'intent')
    assert intent_obj.intent in ['confluence', 'swagger', 'list', 'workflow']
    
    print(f"✅ Intent classified as: {intent_obj.intent} (confidence: {intent_obj.confidence})")
    return result


async def test_search_result_text_property():
    """Test SearchResult.text property works (critical fix)."""
    print("🧪 Testing SearchResult.text property...")
    
    results = create_test_search_results()
    
    for result in results:
        # This was the failing pattern before the fix
        assert hasattr(result, 'text'), f"SearchResult missing .text property"
        assert result.text == result.content, f"Text property doesn't match content"
        
        # Test the specific pattern from actionability.py
        if any(word in result.text.lower() for word in ['customer', 'utility']):
            print(f"✅ Found keywords in: {result.title}")
    
    print("✅ SearchResult.text property working correctly")


async def test_combine_node():
    """Test combine node processes SearchResults without AttributeError."""
    print("🧪 Testing combine node...")
    
    test_state = {
        'normalized_query': 'what is Customer Interaction Utility',
        'search_results': create_test_search_results(),
        'intent': {'intent': 'confluence', 'confidence': 0.7}
    }
    config = {'configurable': {}}
    
    result = await combine_node(test_state, config)
    
    assert 'combined_results' in result
    assert 'final_context' in result
    assert len(result['combined_results']) > 0
    assert len(result['final_context']) > 0
    
    print(f"✅ Combined {len(result['combined_results'])} results")
    print(f"✅ Generated context: {len(result['final_context'])} chars")
    return result


async def test_actionability_detection():
    """Test actionability detection doesn't crash on SearchResult objects."""
    print("🧪 Testing actionability detection...")
    
    # Create test passages (which are actually SearchResult objects)
    passages = create_test_search_results()
    
    try:
        # This should work now with the .text property
        spans = detect_spans(passages)
        print(f"✅ Detected {len(spans)} actionable spans")
        
        # Test text access patterns that were failing
        for passage in passages:
            text_content = passage.text  # This was the AttributeError source
            assert isinstance(text_content, str)
            
        print("✅ Actionability detection completed without errors")
        
    except Exception as e:
        print(f"❌ Actionability detection failed: {e}")
        raise


async def test_full_pipeline():
    """Test the complete search→enhance→combine pipeline."""
    print("🧪 Testing full search→enhance→combine pipeline...")
    
    # Step 1: Intent classification
    intent_result = await test_intent_classification()
    
    # Step 2: SearchResult.text property 
    await test_search_result_text_property()
    
    # Step 3: Combine node processing
    combine_result = await test_combine_node()
    
    # Step 4: Actionability detection
    await test_actionability_detection()
    
    print("✅ Full pipeline test completed successfully!")
    
    # Verify final result structure
    assert 'final_context' in combine_result
    assert 'combined_results' in combine_result
    assert len(combine_result['final_context']) > 100  # Should have meaningful content
    
    return combine_result


async def main():
    """Run all integration tests."""
    print("🚀 Starting search pipeline integration tests...")
    print("=" * 60)
    
    try:
        # Run comprehensive pipeline test
        result = await test_full_pipeline()
        
        print("=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print(f"Pipeline produced {len(result['combined_results'])} results")
        print(f"Final context: {len(result['final_context'])} characters")
        
    except Exception as e:
        print("=" * 60)
        print(f"❌ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())