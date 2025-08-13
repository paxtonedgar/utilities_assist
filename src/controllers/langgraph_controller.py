"""
LangGraph-based controller for sophisticated query processing.

This controller provides the same interface as turn_controller but uses
the advanced multi-agent LangGraph workflow internally.
"""

import logging
import time
from typing import List, Dict, Any, Optional, AsyncGenerator

from workflows.langgraph_workflow import handle_turn_with_langgraph
from workflows.state import WorkflowConfig
from infra.resource_manager import get_resources, RAGResources
from services.models import TurnResult, IntentResult
from infra.telemetry import generate_request_id, log_overall_stage

logger = logging.getLogger(__name__)


async def handle_turn_langgraph(
    user_input: str,
    resources: Optional[RAGResources] = None,
    chat_history: List[Dict[str, str]] = None,
    use_mock_corpus: bool = False,
    enable_langgraph: bool = True,
    workflow_config: Optional[WorkflowConfig] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Enhanced turn handler that can switch between traditional and LangGraph processing.
    
    This function provides the same interface as the original handle_turn but with
    the ability to enable sophisticated multi-agent processing via LangGraph.
    
    Args:
        user_input: Raw user input
        resources: Pre-configured resource container (auto-fetched if None)
        chat_history: Recent conversation history
        use_mock_corpus: If True, use confluence_mock index instead of confluence_current
        enable_langgraph: If True, use LangGraph workflow; if False, fall back to traditional
        workflow_config: Configuration for LangGraph workflow behavior
        
    Yields:
        Dict with turn progress updates and final result
    """
    start_time = time.time()
    req_id = generate_request_id()
    
    try:
        # Get shared resources (maintaining Phase 1 performance optimizations)
        if resources is None:
            resources = get_resources()
            if resources is None:
                raise RuntimeError("Resources not initialized. Call initialize_resources() at startup.")
        
        logger.info(f"Processing query with LangGraph{'enabled' if enable_langgraph else 'disabled'} (resource age: {resources.get_age_seconds():.1f}s)")
        
        if enable_langgraph:
            # Use advanced LangGraph workflow
            async for update in handle_turn_with_langgraph(
                user_input=user_input,
                resources=resources,
                chat_history=chat_history,
                use_mock_corpus=use_mock_corpus,
                workflow_config=workflow_config
            ):
                yield update
        else:
            # Fall back to traditional controller
            from controllers.turn_controller import handle_turn as handle_turn_traditional
            async for update in handle_turn_traditional(
                user_input=user_input,
                resources=resources,
                chat_history=chat_history,
                use_mock_corpus=use_mock_corpus
            ):
                yield update
                
    except Exception as e:
        logger.error(f"LangGraph turn handling failed: {e}")
        
        # Log overall failure
        total_latency = (time.time() - start_time) * 1000
        log_overall_stage(
            req_id=req_id,
            latency_ms=total_latency,
            success=False,
            error=repr(e)
        )
        
        # Return error result
        error_result = TurnResult(
            answer=f"I encountered an error processing your request: {str(e)}",
            sources=[],
            intent=IntentResult(intent="error", confidence=0.0),
            response_time_ms=total_latency,
            error=str(e)
        )
        
        yield {
            "type": "error",
            "result": error_result.dict(),
            "turn_id": f"lg_{req_id}",
            "req_id": req_id
        }


def create_workflow_config_from_settings(resources: RAGResources) -> WorkflowConfig:
    """
    Create workflow configuration from existing settings.
    
    This allows the LangGraph workflow to inherit configuration from the
    existing settings system while providing workflow-specific customization.
    """
    # Extract relevant parameters from existing configuration
    temperature = float(resources.get_config_param('temperature', 0.2))
    max_tokens = int(resources.get_config_param('max_tokens_2k', 1500))
    
    # Create workflow configuration
    return WorkflowConfig(
        # Query complexity thresholds (can be tuned based on performance)
        simple_query_threshold=0.3,
        complex_query_threshold=0.7,
        
        # Search configuration
        max_parallel_searches=3,
        search_timeout_seconds=30,
        enable_cross_index_search=True,
        
        # Result synthesis  
        max_context_length=max_tokens * 4,  # Rough context window estimate
        diversity_lambda=0.75,  # MMR parameter from existing system
        
        # Response generation
        enable_streaming=True,
        chunk_size=50,
        
        # Performance (maintain Phase 1 optimizations)
        enable_caching=True,
        cache_ttl_seconds=300
    )


class LangGraphControllerManager:
    """
    Manager class for coordinating LangGraph workflow usage.
    
    This class provides centralized configuration and monitoring for
    LangGraph workflow usage across the application.
    """
    
    def __init__(self, default_enabled: bool = True):
        self.default_enabled = default_enabled
        self.workflow_metrics = {}
        self._config_cache = {}
    
    async def handle_turn(self, 
                         user_input: str,
                         resources: Optional[RAGResources] = None,
                         chat_history: List[Dict[str, str]] = None,
                         use_mock_corpus: bool = False,
                         force_langgraph: Optional[bool] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Managed turn handling with intelligent LangGraph usage.
        
        Args:
            user_input: User query
            resources: RAG resources
            chat_history: Conversation history
            use_mock_corpus: Whether to use mock data
            force_langgraph: Force LangGraph on/off, or None to use smart detection
        """
        # Determine whether to use LangGraph
        use_langgraph = self._should_use_langgraph(user_input, force_langgraph)
        
        # Get or create workflow configuration
        if resources and resources.settings not in self._config_cache:
            self._config_cache[resources.settings] = create_workflow_config_from_settings(resources)
        
        workflow_config = self._config_cache.get(resources.settings) if resources else None
        
        # Process the turn
        async for update in handle_turn_langgraph(
            user_input=user_input,
            resources=resources,
            chat_history=chat_history,
            use_mock_corpus=use_mock_corpus,
            enable_langgraph=use_langgraph,
            workflow_config=workflow_config
        ):
            # Track metrics
            if update.get("type") == "complete":
                self._update_workflow_metrics(use_langgraph, update)
            
            yield update
    
    def _should_use_langgraph(self, user_input: str, force_langgraph: Optional[bool]) -> bool:
        """
        Determine whether to use LangGraph based on query characteristics.
        
        This function implements smart routing logic to use LangGraph for
        complex queries while falling back to traditional processing for simple ones.
        """
        if force_langgraph is not None:
            return force_langgraph
        
        if not self.default_enabled:
            return False
        
        # Simple heuristics for detecting complex queries
        query_lower = user_input.lower()
        
        # Comparative indicators
        comparative_keywords = ["compare", "versus", "vs", "difference between", "which is better"]
        if any(keyword in query_lower for keyword in comparative_keywords):
            return True
        
        # Multi-part question indicators
        multipart_indicators = ["and also", "in addition", "furthermore", "what about", "how about"]
        if any(indicator in query_lower for indicator in multipart_indicators):
            return True
        
        # List-based queries that might benefit from structured processing
        list_indicators = ["list all", "show me all", "what are the", "which ones"]
        if any(indicator in query_lower for indicator in list_indicators):
            return True
        
        # Complex procedural queries
        if len(query_lower.split()) > 15:  # Long queries might be complex
            return True
        
        # Questions with multiple question words
        question_words = ["what", "how", "when", "where", "why", "which", "who"]
        question_count = sum(1 for word in question_words if word in query_lower)
        if question_count >= 2:
            return True
        
        # Default to traditional processing for simple queries
        return False
    
    def _update_workflow_metrics(self, used_langgraph: bool, update: Dict[str, Any]):
        """Update internal metrics for workflow usage."""
        result = update.get("result", {})
        response_time = result.get("response_time_ms", 0)
        
        workflow_type = "langgraph" if used_langgraph else "traditional"
        
        if workflow_type not in self.workflow_metrics:
            self.workflow_metrics[workflow_type] = {
                "count": 0,
                "total_time": 0,
                "avg_time": 0
            }
        
        metrics = self.workflow_metrics[workflow_type]
        metrics["count"] += 1
        metrics["total_time"] += response_time
        metrics["avg_time"] = metrics["total_time"] / metrics["count"]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for both workflow types."""
        return dict(self.workflow_metrics)
    
    def enable_langgraph(self):
        """Enable LangGraph by default."""
        self.default_enabled = True
    
    def disable_langgraph(self):
        """Disable LangGraph by default."""
        self.default_enabled = False


# Global manager instance for easy access
langgraph_manager = LangGraphControllerManager(default_enabled=False)  # Start disabled for gradual rollout


# Convenience functions for backward compatibility
async def handle_turn_enhanced(
    user_input: str,
    resources: Optional[RAGResources] = None,
    chat_history: List[Dict[str, str]] = None,
    use_mock_corpus: bool = False
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Enhanced turn handler with automatic LangGraph usage detection.
    
    This function can be used as a drop-in replacement for the original handle_turn
    with intelligent routing between traditional and LangGraph processing.
    """
    async for update in langgraph_manager.handle_turn(
        user_input=user_input,
        resources=resources,
        chat_history=chat_history,
        use_mock_corpus=use_mock_corpus
    ):
        yield update