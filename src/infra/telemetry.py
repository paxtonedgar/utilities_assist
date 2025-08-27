"""
Observability and telemetry for chat turns.

Provides structured JSON logging for each stage with timing and error tracking.
Integrates with Streamlit debug drawer for real-time diagnostics.
"""

import orjson
import sys
import time
import uuid
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import threading
from collections import defaultdict


class TelemetryEvent:
    """Flexible telemetry event that accepts any fields."""

    def __init__(self, stage: str, ts: float, req_id: str, **kwargs):
        self.stage = stage
        self.ts = ts
        self.req_id = req_id

        # Store all additional fields as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, filtering None values."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class TelemetryCollector:
    """Thread-safe telemetry collector for debug drawer."""

    def __init__(self):
        self._events = defaultdict(list)  # req_id -> List[TelemetryEvent]
        self._lock = threading.Lock()

    def add_event(self, event: TelemetryEvent):
        """Add event to collection."""
        with self._lock:
            # Keep only last 10 events per request to avoid memory growth
            self._events[event.req_id].append(event)
            if len(self._events[event.req_id]) > 10:
                self._events[event.req_id] = self._events[event.req_id][-10:]

    def get_events(self, req_id: str) -> List[TelemetryEvent]:
        """Get events for a specific request."""
        with self._lock:
            return self._events.get(req_id, []).copy()



# Global collector instance for Streamlit debug drawer
_global_collector = TelemetryCollector()


def get_telemetry_collector() -> TelemetryCollector:
    """Get the global telemetry collector."""
    return _global_collector


def log_event(stage: str, req_id: str, **fields) -> None:
    """
    Log a telemetry event as JSON to stdout and collect for debug drawer.

    Args:
        stage: Stage name (normalize, intent, retrieve_bm25, etc.)
        req_id: Request ID for tracking across stages
        **fields: Additional fields (ms, error, intent, etc.)
    """
    event = TelemetryEvent(stage=stage, ts=time.time(), req_id=req_id, **fields)

    # Log to stdout as JSON
    json_line = orjson.dumps(event.to_dict()).decode()
    sys.stdout.write(json_line + "\n")
    sys.stdout.flush()

    # Add to collector for debug drawer
    _global_collector.add_event(event)




def log_normalize_stage(req_id: str, original: str, normalized: str, ms: float):
    """Log query normalization stage."""
    log_event(
        stage="normalize",
        req_id=req_id,
        ms=round(ms, 2),
        original_length=len(original),
        normalized_length=len(normalized),
    )


















# Utility functions for common patterns




