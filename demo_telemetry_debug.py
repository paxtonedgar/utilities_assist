#!/usr/bin/env python3
"""
Demo script to show telemetry and debug drawer functionality.

Creates sample telemetry events including a kNN timeout scenario
and demonstrates how they would appear in the debug drawer.
"""

import time
from datetime import datetime

import sys
sys.path.insert(0, '.')

from src.infra.telemetry import (
    get_telemetry_collector, 
    format_event_for_display,
    TelemetryEvent,
    log_event,
    generate_request_id
)


def demo_telemetry_events():
    """Create sample telemetry events for a turn with kNN timeout."""
    print("🚀 Telemetry & Debug Drawer Demo")
    print("="*60)
    
    # Clear any existing events
    collector = get_telemetry_collector()
    for req_id in collector.get_all_request_ids():
        collector.clear_events(req_id)
    
    # Generate a request ID for this demo
    req_id = generate_request_id()
    print(f"📍 Demo Request ID: {req_id}")
    
    # Simulate a conversation turn with various stages
    base_time = time.time()
    
    # Stage 1: Query normalization
    log_event(
        stage="normalize",
        req_id=req_id,
        ms=0.5,
        original_length=45,
        normalized_length=52
    )
    
    # Stage 2: Intent classification
    log_event(
        stage="intent", 
        req_id=req_id,
        ms=25.3,
        intent="general",
        confidence=0.82
    )
    
    # Stage 3: Embedding creation
    log_event(
        stage="embedding",
        req_id=req_id,
        ms=120.7,
        text_count=1,
        batch_count=1,
        expected_dims=1536
    )
    
    # Stage 4: BM25 search (succeeds)
    log_event(
        stage="retrieve_bm25",
        req_id=req_id,
        ms=45.2,
        k=10,
        result_count=5,
        top_ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
    )
    
    # Stage 5: kNN search (fails with timeout)
    log_event(
        stage="retrieve_knn", 
        req_id=req_id,
        ms=30000.0,  # 30 seconds
        k=10,
        result_count=0,
        ef_search=256,
        error="TimeoutError: kNN search timed out after 30 seconds"
    )
    
    # Stage 6: RRF fusion (skipped due to kNN failure, fallback to BM25)
    log_event(
        stage="fuse",
        req_id=req_id,
        ms=0.1,
        method="fallback_bm25",
        k_final=5,
        bm25_count=5,
        knn_count=0,
        error="Skipped RRF due to kNN timeout, using BM25 results directly"
    )
    
    # Stage 7: LLM generation
    log_event(
        stage="llm",
        req_id=req_id,
        ms=2340.5,
        model="gpt-4o-mini",
        tokens_in=245,
        tokens_out=127
    )
    
    # Stage 8: Answer verification
    log_event(
        stage="verify",
        req_id=req_id,
        ms=15.2,
        verdict="good",
        unmatched_claims_count=1,
        confidence_score=0.78
    )
    
    # Stage 9: Overall completion (successful despite kNN timeout)
    log_event(
        stage="overall",
        req_id=req_id,
        latency_ms=32547.8,  # ~32.5 seconds total
        success=True,
        result_count=5,
        method="bm25_fallback"
    )
    
    print("✅ Created 9 sample telemetry events")
    
    # Demonstrate telemetry analysis
    print(f"\n📊 TELEMETRY ANALYSIS")
    print("-"*60)
    
    events = collector.get_events(req_id)
    print(f"📈 Captured {len(events)} events for request {req_id[:8]}...")
    
    # Show events in debug drawer format
    print(f"\n🖥️  DEBUG DRAWER VIEW:")
    print("-"*60)
    
    for i, event in enumerate(events):
        formatted = format_event_for_display(event)
        print(f"{i+1:2d}. {formatted['Stage']:15} | {formatted['Duration']:10} | {formatted['Status']:10}")
        print(f"    Details: {formatted['Details']}")
        if 'Error' in formatted:
            print(f"    Error: {formatted['Error']}")
        print(f"    Time: {datetime.fromtimestamp(event.ts).strftime('%H:%M:%S.%f')[:-3]}")
        print()
    
    # Analyze for kNN timeout scenario
    print(f"🧪 SCENARIO VERIFICATION:")
    print("-"*60)
    
    knn_events = [e for e in events if e.stage == "retrieve_knn"]
    knn_timeout = any(
        hasattr(e, 'error') and e.error and "timeout" in e.error.lower()
        for e in knn_events
    )
    print(f"✓ kNN timeout captured: {'✅ YES' if knn_timeout else '❌ NO'}")
    
    bm25_events = [e for e in events if e.stage == "retrieve_bm25"] 
    bm25_success = any(
        not hasattr(e, 'error') or not e.error and hasattr(e, 'result_count') and e.result_count > 0
        for e in bm25_events
    )
    print(f"✓ BM25 fallback worked: {'✅ YES' if bm25_success else '❌ NO'}")
    
    overall_events = [e for e in events if e.stage == "overall"]
    overall_success = any(
        hasattr(e, 'success') and e.success
        for e in overall_events
    )
    print(f"✓ Overall turn success: {'✅ YES' if overall_success else '❌ NO'}")
    
    fuse_events = [e for e in events if e.stage == "fuse"]
    fallback_used = any(
        hasattr(e, 'method') and e.method == "fallback_bm25"
        for e in fuse_events
    )
    print(f"✓ Fallback strategy used: {'✅ YES' if fallback_used else '❌ NO'}")
    
    # Show JSON output (what gets logged to stdout)
    print(f"\n📄 JSON TELEMETRY OUTPUT:")
    print("-"*60)
    print("(This is what appears in stdout for log aggregation)")
    print()
    
    for event in events[:3]:  # Show first 3 events
        print(f"JSON: {event.to_dict()}")
    print("...")
    print(f"(showing first 3 of {len(events)} events)")
    
    # Demonstrate Streamlit integration format
    print(f"\n🎨 STREAMLIT DEBUG DRAWER FORMAT:")
    print("-"*60)
    print("Stage           | Duration   | Status    | Details")
    print("-" * 60)
    
    for event in events:
        formatted = format_event_for_display(event)
        print(f"{formatted['Stage']:15} | {formatted['Duration']:10} | {formatted['Status']:9} | {formatted['Details']}")
    
    print(f"\n🎯 KEY INSIGHTS:")
    print("-"*60)
    print("• kNN search timed out after 30 seconds (visible in telemetry)")
    print("• System gracefully fell back to BM25-only results")
    print("• Overall turn still succeeded despite component failure")
    print("• User received answer within ~32.5 seconds total")
    print("• Debug drawer provides full visibility into all stages")
    print("• Errors are clearly marked and detailed")
    print()
    print("This demonstrates production-ready observability:")
    print("✓ Component failures are captured and visible")
    print("✓ Fallback strategies are logged and traceable")
    print("✓ Performance bottlenecks are easily identified")
    print("✓ End-to-end request tracing is available")


if __name__ == "__main__":
    demo_telemetry_events()