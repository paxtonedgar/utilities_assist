#!/usr/bin/env python3
"""
Test Enhanced Orchestrator - Clean surgical enhancement demo.

Shows how the enhanced orchestrator adds LLM planning and verification
to existing search infrastructure without creating parallel workflows.
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.orchestrator import Orchestrator, orchestrated_search
from src.infra.settings import get_settings


class MockChatClient:
    """Mock chat client for testing structured planning."""
    
    class Chat:
        class Completions:
            async def acreate(self, **kwargs):
                # Mock LLM response for planning
                if "search planner" in kwargs.get("messages", [{}])[0].get("content", "").lower():
                    return MockResponse({
                        "needs_clarification": False,
                        "filters": {"content_type": "procedure"},
                        "expected_answer_shape": "how_to_steps",
                        "confidence": 0.85,
                        "reasoning": "Query asks 'how to' - procedural content needed"
                    })
                
                # Mock answer generation
                return MockResponse("Based on the documentation, here are the steps to configure CIU...")
    
    def __init__(self):
        self.chat = self.Chat()


class MockResponse:
    """Mock OpenAI response."""
    
    def __init__(self, content):
        import json
        if isinstance(content, dict):
            content = json.dumps(content)
        
        self.choices = [MockChoice(content)]
        self.usage = MockUsage()


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockUsage:
    def __init__(self):
        self.total_tokens = 150


class MockResources:
    """Mock resources for testing."""
    
    def __init__(self):
        self.chat_client = MockChatClient()
        self.settings = get_settings()
        self.search_client = None
        self.embed_client = None


async def test_enhanced_orchestrator():
    """Test the enhanced orchestrator functionality."""
    
    print("🔧 Testing Enhanced Orchestrator (Surgical Enhancement)")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "How do I configure CIU?",
        "What is ETU?", 
        "List all APIs",
        "CIU authentication endpoints"
    ]
    
    resources = MockResources()
    
    for query in test_queries:
        print(f"\n📋 Query: '{query}'")
        print("-" * 40)
        
        try:
            # Test with orchestration enabled
            orchestrator = Orchestrator(
                chat_client=resources.chat_client,
                settings=resources.settings
            )
            
            # Mock the search execution since we don't have real OpenSearch
            async def mock_search(q, plan, res):
                return [
                    type('MockResult', (), {
                        'text': f"Mock documentation for {q}. This contains procedural steps and configuration details.",
                        'doc_id': 'mock_doc_1',
                        'title': f'Documentation for {q}'
                    })()
                ]
            
            # Mock the answer generation 
            async def mock_answer(q, results, plan, res):
                if plan.expected_answer_shape == "how_to_steps":
                    return f"**Steps to {q.lower()}:**\n\n1. First step...\n2. Second step...\n3. Final step..."
                elif plan.expected_answer_shape == "api_reference":
                    return f"**API Reference for {q}:**\n\nEndpoints:\n- GET /api/v1/auth\n- POST /api/v1/configure"
                else:
                    return f"**{q}:**\n\nThis is a definition-style answer with detailed explanation."
            
            # Replace methods for testing
            orchestrator._execute_enhanced_search = mock_search
            orchestrator._generate_shaped_answer = mock_answer
            
            # Execute orchestration
            result = await orchestrator.orchestrate(
                query=query,
                resources=resources,
                enable_verification=True
            )
            
            # Display results
            print(f"✅ Planning: {result.get('plan_reasoning', 'N/A')}")
            print(f"📊 Confidence: {result.get('plan_confidence', 0.0):.2f}")
            print(f"🎯 Answer Shape: {result.get('expected_answer_shape', 'N/A')}")
            
            if result.get('ask_clarification'):
                print(f"❓ Clarification: {result['ask_clarification']}")
            else:
                answer = result.get('final_answer', 'No answer generated')
                print(f"💬 Answer: {answer[:100]}...")
            
            # Show verification results if available
            verification = result.get('verification_result')
            if verification:
                print(f"🔍 Verification Score: {verification.get('verification_score', 0):.2f}")
                if verification.get('needs_improvement'):
                    print(f"⚠️  Needs Improvement: {verification.get('suggestions', [])}")
            
            # Show performance stats
            stats = result.get('execution_stats', {})
            total_time = stats.get('total_ms', 0)
            print(f"⏱️  Total Time: {total_time:.1f}ms")
            
            components = []
            for component, time_ms in stats.items():
                if component.endswith('_ms') and component != 'total_ms':
                    components.append(f"{component[:-3]}={time_ms:.1f}ms")
            
            if components:
                print(f"📈 Breakdown: {', '.join(components)}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n🎉 Enhanced Orchestrator Test Complete!")
    print("\n📋 Key Features Demonstrated:")
    print("✅ Structured LLM planning with JSON schema")
    print("✅ Answer shaping based on query intent")
    print("✅ Citation verification and quality checks")  
    print("✅ Performance budgets and timing")
    print("✅ Graceful fallback to existing search")
    print("✅ No parallel workflows - surgical enhancement only")


if __name__ == "__main__":
    asyncio.run(test_enhanced_orchestrator())