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
from dataclasses import dataclass, asdict
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
    
    def clear_events(self, req_id: str):
        """Clear events for a specific request."""
        with self._lock:
            if req_id in self._events:
                del self._events[req_id]
    
    def get_all_request_ids(self) -> List[str]:
        """Get all request IDs with events."""
        with self._lock:
            return list(self._events.keys())


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
    event = TelemetryEvent(
        stage=stage,
        ts=time.time(),
        req_id=req_id,
        **fields
    )
    
    # Log to stdout as JSON
    json_line = orjson.dumps(event.to_dict()).decode()
    sys.stdout.write(json_line + "\n")
    sys.stdout.flush()
    
    # Add to collector for debug drawer
    _global_collector.add_event(event)


@contextmanager
def time_stage(stage: str, req_id: str, **fields):
    """
    Context manager to time a stage and log results.
    
    Args:
        stage: Stage name
        req_id: Request ID
        **fields: Additional fields to include in log
        
    Usage:
        with time_stage("retrieve_bm25", req_id, k=10):
            results = perform_search()
    """
    t0 = time.perf_counter()
    err = None
    
    try:
        yield
    except Exception as e:
        err = repr(e)
        raise
    finally:
        dt = (time.perf_counter() - t0) * 1000
        log_event(
            stage=stage,
            req_id=req_id,
            ms=round(dt, 2),
            error=err,
            **fields
        )


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def log_normalize_stage(req_id: str, original: str, normalized: str, ms: float):
    """Log query normalization stage."""
    log_event(
        stage="normalize",
        req_id=req_id,
        ms=round(ms, 2),
        original_length=len(original),
        normalized_length=len(normalized)
    )


def log_intent_stage(req_id: str, intent: str, confidence: float, ms: float, error: Optional[str] = None):
    """Log intent classification stage."""
    log_event(
        stage="intent",
        req_id=req_id,
        ms=round(ms, 2),
        intent=intent,
        confidence=round(confidence, 3),
        error=error
    )


def log_retrieve_bm25_stage(req_id: str, k: int, result_count: int, ms: float, 
                           top_ids: Optional[List[str]] = None, error: Optional[str] = None):
    """Log BM25 retrieval stage."""
    log_event(
        stage="retrieve_bm25",
        req_id=req_id,
        ms=round(ms, 2),
        k=k,
        result_count=result_count,
        top_ids=top_ids[:5] if top_ids else None,
        error=error
    )


def log_retrieve_knn_stage(req_id: str, k: int, result_count: int, ms: float,
                          ef_search: Optional[int] = None, top_ids: Optional[List[str]] = None, 
                          error: Optional[str] = None):
    """Log kNN retrieval stage."""
    log_event(
        stage="retrieve_knn",
        req_id=req_id,
        ms=round(ms, 2),
        k=k,
        result_count=result_count,
        ef_search=ef_search,
        top_ids=top_ids[:5] if top_ids else None,
        error=error
    )


def log_fuse_stage(req_id: str, method: str, k_final: int, bm25_count: int, knn_count: int, 
                  ms: float, error: Optional[str] = None):
    """Log result fusion stage."""
    log_event(
        stage="fuse",
        req_id=req_id,
        ms=round(ms, 2),
        method=method,
        k_final=k_final,
        bm25_count=bm25_count,
        knn_count=knn_count,
        error=error
    )


def log_llm_stage(req_id: str, model: str, tokens_in: int, tokens_out: int, ms: float,
                 error: Optional[str] = None, context: Optional[str] = None):
    """Log LLM response generation stage."""
    log_event(
        stage="llm",
        req_id=req_id,
        ms=round(ms, 2),
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        error=error,
        context=context
    )


def log_verify_stage(req_id: str, verdict: str, unmatched_claims_count: int, ms: float,
                    confidence_score: Optional[float] = None, error: Optional[str] = None):
    """Log answer verification stage."""
    log_event(
        stage="verify",
        req_id=req_id,
        ms=round(ms, 2),
        verdict=verdict,
        unmatched_claims_count=unmatched_claims_count,
        confidence_score=round(confidence_score, 3) if confidence_score else None,
        error=error
    )


def log_overall_stage(req_id: str, latency_ms: float, success: bool, error: Optional[str] = None,
                     result_count: Optional[int] = None, method: Optional[str] = None):
    """Log overall turn completion."""
    log_event(
        stage="overall",
        req_id=req_id,
        latency_ms=round(latency_ms, 2),
        success=success,
        result_count=result_count,
        method=method,
        error=error
    )


def log_embedding_stage(req_id: str, text_count: int, batch_count: int, expected_dims: int, 
                       ms: float, error: Optional[str] = None):
    """Log embedding creation stage."""
    log_event(
        stage="embedding",
        req_id=req_id,
        ms=round(ms, 2),
        text_count=text_count,
        batch_count=batch_count,
        expected_dims=expected_dims,
        error=error
    )


def format_event_for_display(event: TelemetryEvent) -> Dict[str, Any]:
    """Format telemetry event for display in debug drawer."""
    error = getattr(event, 'error', None)
    ms = getattr(event, 'ms', None)
    
    display = {
        "Stage": event.stage,
        "Duration": f"{ms:.1f}ms" if ms else "—",
        "Status": "❌ Error" if error else "✅ OK"
    }
    
    # Add stage-specific details
    intent = getattr(event, 'intent', None)
    confidence = getattr(event, 'confidence', None)
    result_count = getattr(event, 'result_count', None)
    k = getattr(event, 'k', None)
    top_ids = getattr(event, 'top_ids', None)
    method = getattr(event, 'method', None)
    k_final = getattr(event, 'k_final', None)
    tokens_in = getattr(event, 'tokens_in', None)
    tokens_out = getattr(event, 'tokens_out', None)
    model = getattr(event, 'model', None)
    verdict = getattr(event, 'verdict', None)
    unmatched_claims_count = getattr(event, 'unmatched_claims_count', None)
    latency_ms = getattr(event, 'latency_ms', None)
    
    if event.stage == "intent":
        display["Details"] = f"{intent} ({confidence:.2f})" if intent else "—"
    elif event.stage in ["retrieve_bm25", "retrieve_knn"]:
        details = []
        if result_count is not None and k is not None:
            details.append(f"{result_count}/{k} results")
        if top_ids:
            details.append(f"ids: {', '.join(top_ids[:3])}...")
        display["Details"] = "; ".join(details) if details else "—"
    elif event.stage == "fuse":
        display["Details"] = f"{method} → {k_final} results" if method else "—"
    elif event.stage == "llm":
        if tokens_in and tokens_out:
            display["Details"] = f"{tokens_in}→{tokens_out} tokens, {model}"
        else:
            display["Details"] = model or "—"
    elif event.stage == "verify":
        details = []
        if verdict:
            details.append(f"verdict: {verdict}")
        if unmatched_claims_count is not None:
            details.append(f"unmatched: {unmatched_claims_count}")
        display["Details"] = "; ".join(details) if details else "—"
    elif event.stage == "overall":
        display["Details"] = f"Total: {latency_ms:.1f}ms" if latency_ms else "—"
    else:
        display["Details"] = "—"
    
    # Show error details if present
    if error:
        # Truncate long error messages
        error_msg = error[:100] + "..." if len(error) > 100 else error
        display["Error"] = error_msg
    
    return display


# Utility functions for common patterns

@contextmanager 
def trace_retrieval(stage: str, req_id: str, k: int, **kwargs):
    """Trace retrieval operations with automatic result logging."""
    start_time = time.perf_counter()
    result = None
    error = None
    
    try:
        yield lambda r: globals().update(result=r)  # Allow setting result from context
    except Exception as e:
        error = repr(e)
        raise
    finally:
        ms = (time.perf_counter() - start_time) * 1000
        
        # Extract result info if available
        result_count = len(result.results) if result and hasattr(result, 'results') else 0
        top_ids = [r.doc_id for r in result.results[:5]] if result and hasattr(result, 'results') else None
        
        if stage == "retrieve_bm25":
            log_retrieve_bm25_stage(req_id, k, result_count, ms, top_ids, error)
        elif stage == "retrieve_knn":
            log_retrieve_knn_stage(req_id, k, result_count, ms, 
                                 kwargs.get('ef_search'), top_ids, error)


@contextmanager
def trace_llm_call(req_id: str, model: str):
    """Trace LLM calls with token counting."""
    start_time = time.perf_counter()
    tokens_in = 0
    tokens_out = 0
    error = None
    
    # Context object to collect metrics
    class LLMTrace:
        def set_tokens(self, input_tokens: int, output_tokens: int):
            nonlocal tokens_in, tokens_out
            tokens_in = input_tokens
            tokens_out = output_tokens
    
    trace = LLMTrace()
    
    try:
        yield trace
    except Exception as e:
        error = repr(e)
        raise
    finally:
        ms = (time.perf_counter() - start_time) * 1000
        log_llm_stage(req_id, model, tokens_in, tokens_out, ms, error)