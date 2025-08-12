#!/usr/bin/env python3
"""
Test script to demonstrate telemetry with forced kNN timeout.

This simulates the scenario requested: force a timeout in kNN, 
verify error is present in retrieve_knn telemetry, and confirm 
the system still answers via BM25 fallback.
"""

import asyncio
import logging
import time
from unittest.mock import AsyncMock, Mock
from unittest.mock import patch

# Set up logging to see telemetry output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import sys
sys.path.insert(0, '.')

from src.controllers.turn_controller import handle_turn
from src.infra.config import get_settings
from src.infra.telemetry import get_telemetry_collector, format_event_for_display
from src.services.models import IntentResult, RetrievalResult, SearchResult


class TimeoutMockSearchClient:
    """Mock search client that forces kNN timeout but allows BM25 to succeed."""
    
    def bm25_search(self, query, filters=None, index="confluence_current", k=50, time_decay_half_life_days=120):
        """BM25 search succeeds normally."""
        from src.infra.opensearch_client import SearchResponse, SearchResult
        
        results = [
            SearchResult(
                doc_id="doc1",
                score=0.85,
                title="Test Document 1", 
                body="This is a test document about utilities",
                metadata={"space_key": "TEST"}
            ),
            SearchResult(
                doc_id="doc2",
                score=0.72,
                title="Test Document 2",
                body="Another test document with API information",
                metadata={"space_key": "TEST"}
            )
        ]
        
        return SearchResponse(
            results=results,
            total_hits=2,
            took_ms=45,
            method="bm25"
        )
    
    def knn_search(self, query_vector, filters=None, index="confluence_current", k=50, ef_search=256):
        """kNN search always times out."""
        import time
        time.sleep(0.1)  # Small delay to simulate some processing
        raise TimeoutError("kNN search timed out after 30 seconds")
    
    def rrf_fuse(self, bm25_response, knn_response, k=8, rrf_k=60):
        """RRF fusion - shouldn't be called in timeout scenario."""
        raise Exception("RRF fusion should not be called when kNN fails")


class MockEmbedClient:
    """Mock embedding client that works normally."""
    
    def __init__(self):
        self.embeddings = AsyncMock()
        self.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 1536)]
        )


async def test_knn_timeout_scenario():
    """Test kNN timeout with telemetry tracking."""
    print("🚀 Testing kNN Timeout Scenario with Telemetry")
    print("=" * 60)
    
    # Clear any existing telemetry
    collector = get_telemetry_collector()
    for req_id in collector.get_all_request_ids():
        collector.clear_events(req_id)
    
    settings = get_settings()
    
    # Mock the search client creation to return our timeout client
    with patch('src.controllers.turn_controller.create_search_client') as mock_search_factory:
        mock_search_factory.return_value = TimeoutMockSearchClient()
        
        # Mock the embedding client to work normally  
        with patch('src.controllers.turn_controller.make_embed_client') as mock_embed_factory:
            mock_embed_factory.return_value = MockEmbedClient()
            
            # Mock the chat client to return a simple response
            with patch('src.controllers.turn_controller.make_chat_client') as mock_chat_factory:
                mock_chat_client = AsyncMock()
                mock_chat_factory.return_value = mock_chat_client
                
                # Mock intent determination
                with patch('src.controllers.turn_controller.determine_intent') as mock_intent:
                    mock_intent.return_value = IntentResult(intent="general", confidence=0.8)  # High confidence to trigger hybrid search
                    
                    # Mock response generation
                    with patch('src.controllers.turn_controller.generate_response') as mock_response:
                        async def mock_generate():
                            yield "This is a test response. "
                            yield "The system successfully fell back to BM25 search "
                            yield "when kNN search timed out."
                        mock_response.return_value = mock_generate()
                        
                        # Mock other functions
                        with patch('src.controllers.turn_controller.build_context') as mock_context:
                            mock_context.return_value = "Test context"
                            
                            with patch('src.controllers.turn_controller.verify_answer') as mock_verify:
                                mock_verify.return_value = {
                                    'verdict': 'good', 
                                    'confidence_score': 0.85,
                                    'unmatched_claims_count': 0
                                }
                                
                                with patch('src.controllers.turn_controller.extract_source_chips') as mock_sources:
                                    mock_sources.return_value = []
                                    
                                    # Now run the test
                                    print("📝 Starting turn with high-confidence query (should trigger hybrid search)...")
                                    
                                    req_id = None
                                    final_answer = None
                                    error_occurred = False
                                    
                                    try:
                                        async for update in handle_turn(
                                            user_input="Tell me about the Customer Summary Utility",
                                            settings=settings,
                                            chat_history=[]
                                        ):
                                            if update.get("req_id"):
                                                req_id = update["req_id"]
                                                print(f"📍 Request ID: {req_id}")
                                            
                                            if update["type"] == "response_chunk":
                                                if final_answer is None:
                                                    final_answer = ""
                                                final_answer += update["content"]
                                            
                                            elif update["type"] == "complete":
                                                result = update["result"]
                                                final_answer = result["answer"]
                                                print(f"✅ Turn completed successfully!")
                                                print(f"📝 Answer: {final_answer}")
                                                break
                                                
                                            elif update["type"] == "error":
                                                error_occurred = True
                                                result = update["result"]
                                                print(f"❌ Turn failed: {result['error']}")
                                                break
                                        
                                    except Exception as e:
                                        error_occurred = True
                                        print(f"❌ Exception during turn: {e}")
    
    # Analyze telemetry results
    print("\n" + "="*60)
    print("📊 TELEMETRY ANALYSIS")
    print("="*60)
    
    if req_id:
        events = collector.get_events(req_id)
        
        if events:
            print(f"📈 Captured {len(events)} telemetry events:")
            
            # Look for specific events
            bm25_events = [e for e in events if e.stage == "retrieve_bm25"]
            knn_events = [e for e in events if e.stage == "retrieve_knn"]
            fuse_events = [e for e in events if e.stage == "fuse"]
            overall_events = [e for e in events if e.stage == "overall"]
            
            print(f"\n🔍 BM25 Events: {len(bm25_events)}")
            for event in bm25_events:
                status = "❌ ERROR" if event.error else "✅ OK"
                print(f"   - {status} | {event.ms:.1f}ms | {getattr(event, 'result_count', 'N/A')} results")
                if event.error:
                    print(f"     Error: {event.error[:100]}...")
            
            print(f"\n⚡ kNN Events: {len(knn_events)}")
            for event in knn_events:
                status = "❌ ERROR" if event.error else "✅ OK"
                print(f"   - {status} | {event.ms:.1f}ms | {getattr(event, 'result_count', 'N/A')} results")
                if event.error:
                    print(f"     Error: {event.error[:100]}...")
            
            print(f"\n🔗 Fusion Events: {len(fuse_events)}")
            for event in fuse_events:
                status = "❌ ERROR" if event.error else "✅ OK"
                print(f"   - {status} | {event.ms:.1f}ms | method: {event.method}")
                if event.error:
                    print(f"     Error: {event.error[:100]}...")
            
            print(f"\n🏁 Overall Events: {len(overall_events)}")
            for event in overall_events:
                status = "❌ FAILED" if event.error else "✅ SUCCESS"
                print(f"   - {status} | {event.latency_ms:.1f}ms total")
                if hasattr(event, 'method'):
                    print(f"     Final method: {event.method}")
            
            # Verification checks
            print(f"\n🧪 VERIFICATION:")
            
            # Check 1: kNN should have failed with timeout
            knn_timeout_found = any(
                e.error and "timeout" in e.error.lower() 
                for e in knn_events
            )
            print(f"   ✓ kNN timeout error captured: {'✅ YES' if knn_timeout_found else '❌ NO'}")
            
            # Check 2: BM25 should have succeeded  
            bm25_success = any(
                not e.error and getattr(e, 'result_count', 0) > 0
                for e in bm25_events
            )
            print(f"   ✓ BM25 fallback succeeded: {'✅ YES' if bm25_success else '❌ NO'}")
            
            # Check 3: Overall should be successful (despite kNN failure)
            overall_success = any(
                not e.error 
                for e in overall_events
            )
            print(f"   ✓ Overall turn successful: {'✅ YES' if overall_success else '❌ NO'}")
            
            # Check 4: Final answer should exist
            print(f"   ✓ Answer generated: {'✅ YES' if final_answer and len(final_answer) > 0 else '❌ NO'}")
            
            print(f"\n🎯 SUMMARY:")
            if knn_timeout_found and bm25_success and overall_success and final_answer:
                print("   ✅ TEST PASSED: kNN timeout was handled gracefully")
                print("   ✅ System fell back to BM25 and provided an answer") 
                print("   ✅ All telemetry events were captured correctly")
            else:
                print("   ❌ TEST ISSUES: Some expectations were not met")
            
            print(f"\n📋 DEBUG DRAWER FORMAT:")
            print("   (This is how events would appear in the Streamlit debug drawer)")
            for event in events:
                formatted = format_event_for_display(event)
                print(f"   {formatted['Stage']:15} | {formatted['Duration']:8} | {formatted['Status']:8} | {formatted['Details']}")
                
        else:
            print("❌ No telemetry events captured!")
    else:
        print("❌ No request ID captured!")


async def main():
    """Run the timeout test."""
    await test_knn_timeout_scenario()


if __name__ == "__main__":
    asyncio.run(main())