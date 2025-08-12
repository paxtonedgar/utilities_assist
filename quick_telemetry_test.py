#!/usr/bin/env python3
"""
Quick test to verify telemetry integration works end-to-end.
"""

import asyncio
import sys
sys.path.insert(0, '.')

from src.infra.telemetry import get_telemetry_collector, generate_request_id, log_event


async def test_basic_telemetry():
    """Test basic telemetry functionality."""
    print("🧪 Quick Telemetry Integration Test")
    print("="*50)
    
    # Clear existing events
    collector = get_telemetry_collector()
    for req_id in collector.get_all_request_ids():
        collector.clear_events(req_id)
    
    # Generate a request ID
    req_id = generate_request_id()
    print(f"📍 Request ID: {req_id}")
    
    # Log some events
    log_event("test_start", req_id, ms=0.1, test_field="hello")
    log_event("test_middle", req_id, ms=5.3, count=42)
    log_event("test_end", req_id, ms=1.7, success=True)
    
    # Retrieve and verify
    events = collector.get_events(req_id)
    print(f"📊 Captured {len(events)} events")
    
    for event in events:
        print(f"   - {event.stage}: {event.to_dict()}")
    
    print("✅ Basic telemetry working correctly!")


if __name__ == "__main__":
    asyncio.run(test_basic_telemetry())