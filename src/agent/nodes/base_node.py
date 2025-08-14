# src/agent/nodes/base_node.py
"""
Base node handler for LangGraph nodes with DRY principles.

Eliminates wrapper function repetition by providing common error handling,
logging, and state management patterns.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from src.telemetry.logger import log_event, stage

logger = logging.getLogger(__name__)


class BaseNodeHandler(ABC):
    """Base class for all LangGraph node handlers with common wrapper logic."""
    
    def __init__(self, node_name: str):
        self.node_name = node_name
        self.logger = logging.getLogger(f"{__name__}.{node_name}")
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any], config: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute the node-specific logic. Must be implemented by subclasses."""
        pass
    
    async def __call__(self, state: Dict[str, Any], *, config: Optional[Dict] = None, store=None) -> Dict[str, Any]:
        """
        Common wrapper logic for all nodes - eliminates repetition.
        
        Handles:
        - Error logging and propagation
        - Execution timing
        - State validation
        - Workflow path tracking
        """
        start_time = time.time()
        
        # Add node to workflow path
        workflow_path = state.get("workflow_path", [])
        state = {**state, "workflow_path": workflow_path + [self.node_name]}
        
        try:
            # Log node entry
            log_event(
                stage=self.node_name,
                event="start", 
                query=state.get("normalized_query", ""),
                thread_id=config.get("configurable", {}).get("thread_id") if config else None
            )
            
            # Execute node-specific logic - MUST pass config through
            result = await self.execute(state, config)
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                raise ValueError(f"Node {self.node_name} must return a dictionary, got {type(result)}")
            
            # Log success
            execution_time = (time.time() - start_time) * 1000
            log_event(
                stage=self.node_name,
                event="success",
                ms=execution_time,
                result_count=len(result.get("search_results", [])) if "search_results" in result else 0
            )
            
            # Merge result with existing state to preserve all fields
            return {**state, **result}
            
        except Exception as e:
            # Log error
            execution_time = (time.time() - start_time) * 1000
            log_event(
                stage=self.node_name,
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                ms=execution_time
            )
            
            # Add to error messages for debugging
            error_messages = state.get("error_messages", [])
            error_message = f"Error in {self.node_name}: {str(e)}"
            
            self.logger.error(error_message, exc_info=True)
            
            return {
                **state,
                "error_messages": error_messages + [error_message]
            }


class SearchNodeHandler(BaseNodeHandler):
    """Base class for search-related nodes with search-specific logging."""
    
    async def __call__(self, state: Dict[str, Any], *, config: Optional[Dict] = None, store=None) -> Dict[str, Any]:
        """Enhanced wrapper for search nodes with search-specific metrics."""
        start_time = time.time()
        
        # Add node to workflow path
        workflow_path = state.get("workflow_path", [])
        state = {**state, "workflow_path": workflow_path + [self.node_name]}
        
        try:
            # Log search node entry with query details
            log_event(
                stage=self.node_name,
                event="start",
                query=state.get("normalized_query", ""),
                intent=state.get("intent", {}).get("intent") if state.get("intent") else None,
                thread_id=config.get("configurable", {}).get("thread_id") if config else None
            )
            
            # Execute search logic - MUST pass config through
            result = await self.execute(state, config)
            
            # Enhanced search logging
            execution_time = (time.time() - start_time) * 1000
            search_results = result.get("search_results", [])
            
            log_event(
                stage=self.node_name,
                event="success",
                ms=execution_time,
                result_count=len(search_results),
                avg_score=sum(r.score for r in search_results) / len(search_results) if search_results else 0,
                top_score=max(r.score for r in search_results) if search_results else 0
            )
            
            # Merge result with existing state to preserve all fields
            return {**state, **result}
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            log_event(
                stage=self.node_name,
                event="error", 
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                ms=execution_time
            )
            
            error_messages = state.get("error_messages", [])
            error_message = f"Search error in {self.node_name}: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            
            return {
                **state,
                "error_messages": error_messages + [error_message],
                "search_results": []  # Provide empty results for search failures
            }