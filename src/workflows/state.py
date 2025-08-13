"""LangGraph workflow state definitions."""

import operator
from typing import Dict, List, Any, Optional, Annotated, TypedDict, Literal
from dataclasses import dataclass
from services.models import SearchResult, IntentResult

# Core workflow state that gets passed between all agents
class WorkflowState(TypedDict):
    # === Input ===
    original_query: str
    user_context: Optional[Dict[str, Any]]
    request_id: Optional[str]
    
    # === Query Processing ===
    normalized_query: Optional[str]
    intent: Optional[IntentResult]
    query_complexity: Optional[Literal["simple", "complex", "multi_part", "comparative"]]
    sub_queries: Optional[List[str]]
    search_strategies: Optional[List[str]]
    
    # === Search Results (accumulative) ===
    # Using Annotated with operator.add to accumulate results from parallel agents
    search_results: Annotated[List[SearchResult], operator.add]
    result_sources: Optional[Dict[str, List[SearchResult]]]
    
    # === Generation ===
    synthesized_context: Optional[str]
    response_chunks: Annotated[List[str], operator.add]
    final_answer: Optional[str]
    
    # === Metadata ===
    workflow_path: Annotated[List[str], operator.add]  # Track which agents were called
    performance_metrics: Optional[Dict[str, float]]
    error_messages: Annotated[List[str], operator.add]


# Specialized states for different workflow branches

class QueryAnalysisState(TypedDict):
    """State for query analysis and decomposition."""
    original_query: str
    normalized_query: str
    complexity_score: float
    decomposition_needed: bool
    query_type: str  # "factual", "comparative", "multi_step", "list"
    entities_mentioned: List[str]
    

class SearchCoordinationState(TypedDict):
    """State for coordinating multiple search operations."""
    search_plan: List[Dict[str, Any]]  # List of search tasks with parameters
    parallel_searches: List[str]  # Names of search agents to run in parallel
    search_priority: Dict[str, int]  # Priority scores for different searches
    

class ResultSynthesisState(TypedDict):
    """State for synthesizing results from multiple sources."""
    raw_results: List[SearchResult]
    grouped_results: Dict[str, List[SearchResult]]  # Grouped by source/topic
    relevance_scores: Dict[str, float]
    synthesis_strategy: str  # "concatenate", "compare", "summarize", "rank"


@dataclass
class WorkflowConfig:
    """Configuration for workflow behavior."""
    # Query complexity thresholds
    simple_query_threshold: float = 0.3
    complex_query_threshold: float = 0.7
    
    # Search configuration
    max_parallel_searches: int = 3
    search_timeout_seconds: int = 30
    enable_cross_index_search: bool = True
    
    # Result synthesis
    max_context_length: int = 8000
    diversity_lambda: float = 0.75  # MMR parameter
    
    # Response generation
    enable_streaming: bool = True
    chunk_size: int = 50
    
    # Performance
    enable_caching: bool = True
    cache_ttl_seconds: int = 300


# Common utility functions for state management

def initialize_workflow_state(query: str, request_id: str = None, user_context: Dict[str, Any] = None) -> WorkflowState:
    """Initialize a new workflow state."""
    return WorkflowState(
        original_query=query,
        user_context=user_context or {},
        request_id=request_id,
        normalized_query=None,
        intent=None,
        query_complexity=None,
        sub_queries=None,
        search_strategies=None,
        search_results=[],
        result_sources=None,
        synthesized_context=None,
        response_chunks=[],
        final_answer=None,
        workflow_path=[],
        performance_metrics=None,
        error_messages=[]
    )


def add_workflow_step(state: WorkflowState, step_name: str) -> Dict[str, Any]:
    """Add a workflow step to the path tracking."""
    return {"workflow_path": [step_name]}


def log_error(state: WorkflowState, error_msg: str) -> Dict[str, Any]:
    """Log an error in the workflow state."""
    return {"error_messages": [error_msg]}


def update_metrics(state: WorkflowState, metrics: Dict[str, float]) -> Dict[str, Any]:
    """Update performance metrics."""
    current_metrics = state.get("performance_metrics", {})
    current_metrics.update(metrics)
    return {"performance_metrics": current_metrics}