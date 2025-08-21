# src/telemetry/logger.py
"""
Comprehensive structured logging with stage tracking for utilities assistant.

Features:
- JSON structured logging with contextual stage information
- Request ID generation and propagation
- Stage timing decorators with automatic instrumentation
- LangGraph callback handlers for node/tool tracking
- OpenSearch client logging integration
- Streamlit UI log surfacing
"""

import logging
import json
import time
import functools
import asyncio
import uuid
import traceback
from typing import Dict, Any, Optional, Union, List
from contextvars import ContextVar
from pathlib import Path
import sys
import os

from src.infra.settings import get_settings

# Context variables for tracking
_current_stage: ContextVar[Optional[str]] = ContextVar('current_stage', default=None)
_current_req_id: ContextVar[Optional[str]] = ContextVar('current_req_id', default=None)
_user_id: ContextVar[Optional[str]] = ContextVar('user_id', default=None)  
_thread_id: ContextVar[Optional[str]] = ContextVar('thread_id', default=None)

# In-memory log buffer for Streamlit UI
_log_buffer: List[Dict[str, Any]] = []
_max_buffer_size = 1000

def get_context_var(name: str, default=None):
    """Get context variable value."""
    if name == "current_stage":
        return _current_stage.get(default)
    elif name == "current_req_id":
        return _current_req_id.get(default)
    elif name == "user_id":
        return _user_id.get(default)
    elif name == "thread_id":
        return _thread_id.get(default)
    return default

def set_context_var(name: str, value):
    """Set context variable value."""
    if name == "current_stage":
        _current_stage.set(value)
    elif name == "current_req_id":
        _current_req_id.set(value)
    elif name == "user_id":
        _user_id.set(value)
    elif name == "thread_id":
        _thread_id.set(value)

def generate_req_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())[:8]

def get_or_create_req_id() -> str:
    """Get current request ID or create a new one."""
    req_id = get_context_var("current_req_id")
    if not req_id:
        req_id = generate_req_id()
        set_context_var("current_req_id", req_id)
    return req_id


# Constants for hotspot detection
HOTSPOT_THRESHOLD_MS = 2000
WARNING_THRESHOLD_MS = 500

# Performance thresholds for stage name normalization
STAGE_NAME_FIXES = {
    'knn': 'knn',
    'k_nn': 'knn', 
    'kknn': 'knn',
    'bm_25': 'bm25',
    'bm25': 'bm25'
}

# ID field mappings for consistency
ID_FIELD_MAPPINGS = {
    'reg_id': 'req_id',
    'request_id': 'req_id'
}


class TimelineTracker:
    """Thread-safe timeline tracking for hotspot identification."""
    
    def __init__(self):
        self._timeline_buffer: Dict[str, List[Dict[str, Any]]] = {}
    
    def log_event(self, req_id: str, stage: str, duration_ms: float, **metadata):
        """Log timeline event for hotspot identification."""
        if req_id not in self._timeline_buffer:
            self._timeline_buffer[req_id] = []
        
        self._timeline_buffer[req_id].append({
            'stage': self._normalize_stage_name(stage),
            'ms': round(duration_ms, 1),
            **metadata
        })
    
    def get_summary(self, req_id: str) -> str:
        """Get one-line timeline summary for hotspot identification."""
        if req_id not in self._timeline_buffer:
            return ""
        
        events = self._timeline_buffer[req_id]
        if not events:
            return ""
        
        # Create concise timeline: stage→ms format
        timeline_parts = []
        total_ms = 0
        
        for event in events:
            stage = event['stage']
            ms = event['ms']
            total_ms += ms
            timeline_parts.append(self._format_timeline_part(stage, ms))
        
        # Format as single line
        timeline = " | ".join(timeline_parts)
        return f"Timeline[{total_ms:.0f}ms]: {timeline}"
    
    def log_turn_complete(self, req_id: str):
        """Log complete turn timeline and clear buffer."""
        summary = self.get_summary(req_id)
        if summary:
            logger = logging.getLogger("timeline")
            logger.info(f"req_id={req_id} {summary}")
            # Clear buffer for this request
            self._timeline_buffer.pop(req_id, None)
    
    def _normalize_stage_name(self, stage: str) -> str:
        """Normalize stage name for consistency."""
        normalized = stage.lower().replace('-', '_').replace(' ', '_')
        return STAGE_NAME_FIXES.get(normalized, normalized)
    
    def _format_timeline_part(self, stage: str, ms: float) -> str:
        """Format timeline part with hotspot indicators."""
        if ms > HOTSPOT_THRESHOLD_MS:
            return f"{stage}→🔥{ms:.0f}ms"
        elif ms > WARNING_THRESHOLD_MS:
            return f"{stage}→⚠️{ms:.0f}ms"
        else:
            return f"{stage}→{ms:.0f}ms"


# Global instance (singleton pattern for performance)
_timeline_tracker = TimelineTracker()


def log_timeline_event(req_id: str, stage: str, duration_ms: float, **metadata):
    """Log timeline event for hotspot identification."""
    _timeline_tracker.log_event(req_id, stage, duration_ms, **metadata)


def get_timeline_summary(req_id: str) -> str:
    """Get one-line timeline summary for hotspot identification."""
    return _timeline_tracker.get_summary(req_id)


def log_turn_timeline(req_id: str):
    """Log complete turn timeline and clear buffer."""
    _timeline_tracker.log_turn_complete(req_id)


def _normalize_log_keys(**kwargs) -> Dict[str, Any]:
    """Normalize log keys to fix observability hygiene issues."""
    normalized = {}
    
    for key, value in kwargs.items():
        # Clean up key names first
        clean_key = _clean_key_name(key)
        
        # Fix req_id vs reg_id inconsistencies
        if clean_key in ID_FIELD_MAPPINGS:
            clean_key = ID_FIELD_MAPPINGS[clean_key]
        
        # Normalize stage names to snake_case
        if clean_key == 'stage':
            normalized_stage = str(value).lower().replace('-', '_').replace(' ', '_')
            value = STAGE_NAME_FIXES.get(normalized_stage, normalized_stage)
        
        normalized[clean_key] = value
    
    return normalized


def _clean_key_name(key: str) -> str:
    """Clean and normalize log key names."""
    # Remove various quote types
    clean_key = key.replace('"', '').replace("'", '').replace('"', '').replace("'", '')
    # Normalize to snake_case
    return clean_key.lower().replace('-', '_').replace(' ', '_')


def log_event(stage: str, req_id: Optional[str] = None, **kwargs):
    """
    Log a structured event with stage, request ID, and arbitrary key-value pairs.
    
    Args:
        stage: Stage name (e.g., "bm25", "knn", "normalize")
        req_id: Optional request ID (will generate if not provided)
        **kwargs: Additional fields to include in the log event
    """
    if not req_id:
        req_id = get_or_create_req_id()
    
    # Normalize stage name and log keys for observability hygiene
    stage = stage.lower().replace('-', '_').replace(' ', '_')
    kwargs = _normalize_log_keys(**kwargs)
    
    # Check if we should reduce verbosity
    try:
        settings = get_settings()
        reduce_verbosity = settings.reduce_log_verbosity
    except:
        reduce_verbosity = False
    
    # Filter out verbose events in production
    if reduce_verbosity:
        event = kwargs.get("event", "")
        # Only log errors, high-level successes, and important events
        if event not in ["error", "success", "start"] and stage not in ["overall", "bm25", "knn"]:
            # Still add to UI buffer, but don't log to files
            log_data = {
                "stage": stage,
                "req_id": req_id,
                "timestamp": time.time(),
                **kwargs
            }
            _add_to_buffer(log_data)
            return
    
    # Build log data
    log_data = {
        "stage": stage,
        "req_id": req_id,
        "timestamp": time.time(),
        **kwargs
    }
    
    # Add context variables
    user_id = get_context_var("user_id")
    thread_id = get_context_var("thread_id")
    if user_id:
        log_data["user_id"] = user_id
    if thread_id:
        log_data["thread_id"] = thread_id
    
    # Add to buffer for Streamlit UI
    _add_to_buffer(log_data)
    
    # Log via standard logger
    logger = logging.getLogger(f"stage.{stage}")
    logger.info(json.dumps(log_data, default=str))

def _add_to_buffer(log_data: Dict[str, Any]):
    """Add log event to in-memory buffer for UI display."""
    global _log_buffer
    _log_buffer.append(log_data)
    
    # Keep buffer size manageable
    if len(_log_buffer) > _max_buffer_size:
        _log_buffer = _log_buffer[-_max_buffer_size//2:]

def get_stage_logs(req_id: Optional[str] = None, last_n: int = 50) -> List[Dict[str, Any]]:
    """Get recent stage logs, optionally filtered by request ID."""
    if req_id:
        return [log for log in _log_buffer if log.get("req_id") == req_id][-last_n:]
    else:
        return _log_buffer[-last_n:]

def json_logger():
    """Get a JSON logger instance configured for structured logging."""
    return logging.getLogger("json_events")


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging with stage context."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add context variables
        stage = get_context_var("current_stage")
        req_id = get_context_var("current_req_id")
        user_id = get_context_var("user_id")
        thread_id = get_context_var("thread_id")
        
        if stage:
            log_data["stage"] = stage
        if req_id:
            log_data["req_id"] = req_id
        if user_id:
            log_data["user_id"] = user_id
        if thread_id:
            log_data["thread_id"] = thread_id
        
        # Add extra fields from record
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
            
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


def setup_logging():
    """Setup application-wide structured logging."""
    settings = get_settings()
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if settings.enable_structured_logging:
        # JSON structured logging for production
        formatter = StructuredFormatter()
        
        # Console handler with structured output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler for persistent logs
        file_handler = logging.FileHandler(log_dir / "app.log")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
    else:
        # Simple logging for development
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Configure specific loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - structured: {settings.enable_structured_logging}, level: {settings.log_level}")


def stage(name: str):
    """
    Decorator to time functions and emit structured logs.
    
    Emits: {"stage": name, "ms": elapsed_time, "status": "success"|"error", **kwargs}
    
    Args:
        name: Stage name for logging
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            req_id = get_or_create_req_id()
            
            # Extract config info from kwargs for logging
            config_info = _extract_config_info(kwargs)
            
            # Log stage start
            log_event(
                stage=name,
                req_id=req_id,
                event="start",
                inputs_count=len(args),
                **config_info
            )
            
            # Set stage context
            previous_stage = get_context_var("current_stage")
            set_context_var("current_stage", name)
            
            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Extract result info
                result_info = _extract_result_info(result)
                
                # Log stage success
                log_event(
                    stage=name,
                    req_id=req_id,
                    event="end",
                    ms=elapsed_ms,
                    status="success",
                    **result_info
                )
                
                # Add to timeline for hotspot identification
                log_timeline_event(req_id, name, elapsed_ms, **result_info)
                
                return result
                
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Log stage error
                log_event(
                    stage=name,
                    req_id=req_id,
                    event="error",
                    ms=elapsed_ms,
                    status="error",
                    err=True,
                    error_type=type(e).__name__,
                    error_message=str(e)[:200]  # Truncate long error messages
                )
                raise
                
            finally:
                # Restore previous stage context
                set_context_var("current_stage", previous_stage)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            req_id = get_or_create_req_id()
            
            # Extract config info from kwargs for logging
            config_info = _extract_config_info(kwargs)
            
            # Log stage start
            log_event(
                stage=name,
                req_id=req_id,
                event="start",
                inputs_count=len(args),
                **config_info
            )
            
            # Set stage context
            previous_stage = get_context_var("current_stage")
            set_context_var("current_stage", name)
            
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Extract result info
                result_info = _extract_result_info(result)
                
                # Log stage success
                log_event(
                    stage=name,
                    req_id=req_id,
                    event="end",
                    ms=elapsed_ms,
                    status="success",
                    **result_info
                )
                
                # Add to timeline for hotspot identification
                log_timeline_event(req_id, name, elapsed_ms, **result_info)
                
                return result
                
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                
                # Log stage error
                log_event(
                    stage=name,
                    req_id=req_id,
                    event="error",
                    ms=elapsed_ms,
                    status="error",
                    err=True,
                    error_type=type(e).__name__,
                    error_message=str(e)[:200]  # Truncate long error messages
                )
                raise
                
            finally:
                # Restore previous stage context
                set_context_var("current_stage", previous_stage)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

def _extract_config_info(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Extract configuration info from function kwargs for logging."""
    config_info = {}
    
    # Common config fields to log
    if "index" in kwargs:
        config_info["index"] = kwargs["index"]
    if "k" in kwargs:
        config_info["k"] = kwargs["k"]
    if "filters" in kwargs:
        config_info["filters_enabled"] = kwargs["filters"] is not None
    if "search_index" in kwargs:
        config_info["index"] = kwargs["search_index"]
        
    return config_info

def _extract_result_info(result: Any) -> Dict[str, Any]:
    """Extract result information for logging."""
    result_info = {}
    
    if hasattr(result, '__len__'):
        try:
            result_info["result_count"] = len(result)
        except:
            pass
    
    # Handle common result types
    if isinstance(result, dict):
        if "results" in result:
            result_info["result_count"] = len(result["results"])
        if "hits" in result:
            result_info["hits"] = result["hits"]
        if "tokens" in result:
            result_info["tokens"] = result["tokens"]
    
    return result_info

# Keep the old name as an alias for backward compatibility
stage_tracker = stage


class LangGraphCallbackHandler:
    """Enhanced callback handler for LangGraph to track node and tool execution."""
    
    def __init__(self):
        self.node_starts = {}
        self.tool_starts = {}
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Called when a LangGraph node starts."""
        chain_id = kwargs.get('run_id', str(uuid.uuid4())[:8])
        node_name = serialized.get('name', 'unknown_node')
        
        self.node_starts[chain_id] = time.time()
        
        # Log node start
        log_event(
            stage="graph_node",
            node=node_name,
            chain_id=chain_id,
            event="start",
            input_size=len(str(inputs)),
            ts=time.time()
        )
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs):
        """Called when a LangGraph node completes."""
        chain_id = kwargs.get('run_id', 'unknown')
        
        elapsed_ms = 0
        if chain_id in self.node_starts:
            elapsed_ms = (time.time() - self.node_starts[chain_id]) * 1000
            del self.node_starts[chain_id]
        
        # Log node completion
        log_event(
            stage="graph_node",
            chain_id=chain_id,
            event="end",
            ms=elapsed_ms,
            status="success",
            output_size=len(str(outputs)),
            ts=time.time()
        )
    
    def on_chain_error(self, error: Exception, **kwargs):
        """Called when a LangGraph node fails."""
        chain_id = kwargs.get('run_id', 'unknown')
        
        elapsed_ms = 0
        if chain_id in self.node_starts:
            elapsed_ms = (time.time() - self.node_starts[chain_id]) * 1000
            del self.node_starts[chain_id]
        
        # Log node error
        log_event(
            stage="graph_node",
            chain_id=chain_id,
            event="error",
            ms=elapsed_ms,
            status="error",
            err=True,
            error_type=type(error).__name__,
            error_message=str(error)[:200],
            ts=time.time()
        )
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Called when a tool starts executing."""
        run_id = kwargs.get('run_id', str(uuid.uuid4())[:8])
        tool_name = serialized.get('name', 'unknown_tool')
        
        self.tool_starts[run_id] = time.time()
        
        # Log tool start
        log_event(
            stage="tool",
            tool=tool_name,
            run_id=run_id,
            event="start",
            input_size=len(input_str),
            ts=time.time()
        )
    
    def on_tool_end(self, output: str, **kwargs):
        """Called when a tool completes."""
        run_id = kwargs.get('run_id', 'unknown')
        
        elapsed_ms = 0
        if run_id in self.tool_starts:
            elapsed_ms = (time.time() - self.tool_starts[run_id]) * 1000
            del self.tool_starts[run_id]
        
        # Log tool completion
        log_event(
            stage="tool",
            run_id=run_id,
            event="end",
            ms=elapsed_ms,
            status="success",
            output_size=len(output),
            ts=time.time()
        )
    
    def on_tool_error(self, error: Exception, **kwargs):
        """Called when a tool fails."""
        run_id = kwargs.get('run_id', 'unknown')
        
        elapsed_ms = 0
        if run_id in self.tool_starts:
            elapsed_ms = (time.time() - self.tool_starts[run_id]) * 1000
            del self.tool_starts[run_id]
        
        # Log tool error
        log_event(
            stage="tool",
            run_id=run_id,
            event="error",
            ms=elapsed_ms,
            status="error",
            err=True,
            error_type=type(error).__name__,
            error_message=str(error)[:200],
            ts=time.time()
        )


def set_user_context(user_id: str, thread_id: Optional[str] = None):
    """Set user context for structured logging."""
    set_context_var("user_id", user_id)
    if thread_id:
        set_context_var("thread_id", thread_id)


def clear_user_context():
    """Clear user context."""
    set_context_var("user_id", None)
    set_context_var("thread_id", None)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(name)


# Stage definitions for consistency
class Stages:
    """Predefined stage names for consistent tracking."""
    
    # Request processing stages
    REQUEST_START = "request_start"
    AUTH_CHECK = "auth_check" 
    INPUT_VALIDATION = "input_validation"
    
    # LangGraph stages  
    GRAPH_INVOKE = "graph_invoke"
    INTENT_ANALYSIS = "intent_analysis"
    QUERY_REWRITE = "query_rewrite"
    SEARCH_EXECUTION = "search_execution"
    RESULT_FUSION = "result_fusion"
    SUMMARIZATION = "summarization"
    
    # Infrastructure stages
    OPENSEARCH_QUERY = "opensearch_query"
    LLM_CALL = "llm_call"
    EMBEDDING_GENERATION = "embedding_generation"
    
    # Response stages
    RESPONSE_FORMATTING = "response_formatting"
    REQUEST_COMPLETE = "request_complete"


# Auto-setup logging on import if not in test environment
if "pytest" not in sys.modules and not os.getenv("PYTEST_CURRENT_TEST"):
    setup_logging()