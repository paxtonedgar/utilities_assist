# src/telemetry/__init__.py
"""
Telemetry and observability module for utilities assistant.

Provides structured logging, performance monitoring, and LangGraph integration
for comprehensive observability across all application stages.
"""

from .logger import (
    setup_logging,
    stage,
    stage_tracker,
    log_event,
    json_logger,
    get_stage_logs,
    generate_req_id,
    get_or_create_req_id,
    set_context_var,
    get_context_var,
    LangGraphCallbackHandler,
    set_user_context,
    clear_user_context,
    get_logger,
    Stages
)

__all__ = [
    'setup_logging',
    'stage',
    'stage_tracker',
    'log_event',
    'json_logger',
    'get_stage_logs',
    'generate_req_id', 
    'get_or_create_req_id',
    'set_context_var',
    'get_context_var',
    'LangGraphCallbackHandler',
    'set_user_context',
    'clear_user_context',
    'get_logger',
    'Stages'
]