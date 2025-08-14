# src/agent/constants.py
"""
State key constants for LangGraph workflow.

Centralized constants to prevent KeyError issues and maintain consistency.
"""

# State key constants - use these across all nodes to prevent KeyError issues
ORIGINAL_QUERY = "original_query"
NORMALIZED_QUERY = "normalized_query" 
INTENT = "intent"