# src/agent/nodes/search_nodes.py
"""
Search-specific node handlers implementing the base node pattern.

Replaces individual wrapper functions with clean, testable classes.
"""

from typing import Dict, Any, List
import logging

from .base_node import SearchNodeHandler
from agent.nodes.summarize import summarize_node
from agent.nodes.intent import intent_node
from agent.tools.search import adaptive_search_tool, multi_index_search_tool
from services.models import SearchResult

logger = logging.getLogger(__name__)


class SummarizeNode(SearchNodeHandler):
    """Handles query summarization and normalization."""
    
    def __init__(self):
        super().__init__("summarize")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query summarization logic."""
        original_query = state.get("original_query", "")
        
        # Use existing summarize_node function
        result = await summarize_node(original_query)
        
        return {
            "normalized_query": result.get("normalized_query", original_query)
        }


class IntentNode(SearchNodeHandler):
    """Handles intent classification."""
    
    def __init__(self):
        super().__init__("intent")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intent classification logic."""
        normalized_query = state.get("normalized_query", "")
        
        # Use existing intent_node function
        intent_result = await intent_node(normalized_query)
        
        return {
            "intent": intent_result
        }


class ConfluenceSearchNode(SearchNodeHandler):
    """Handles Confluence-specific search."""
    
    def __init__(self):
        super().__init__("search_confluence")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Confluence search logic."""
        normalized_query = state.get("normalized_query", "")
        
        # Use adaptive search tool with confluence index
        search_results = await adaptive_search_tool.ainvoke({
            "query": normalized_query,
            "index_name": "confluence_current",
            "search_type": "hybrid"
        })
        
        return {
            "search_results": search_results.get("results", [])
        }


class SwaggerSearchNode(SearchNodeHandler):
    """Handles Swagger/API documentation search."""
    
    def __init__(self):
        super().__init__("search_swagger")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Swagger search logic."""
        normalized_query = state.get("normalized_query", "")
        
        # Use adaptive search tool with swagger index
        search_results = await adaptive_search_tool.ainvoke({
            "query": normalized_query,
            "index_name": "swagger_current", 
            "search_type": "hybrid"
        })
        
        return {
            "search_results": search_results.get("results", [])
        }


class MultiSearchNode(SearchNodeHandler):
    """Handles multi-index search for complex queries."""
    
    def __init__(self):
        super().__init__("search_multi")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-index search logic."""
        normalized_query = state.get("normalized_query", "")
        intent = state.get("intent")
        
        # Use multi-index search tool
        search_results = await multi_index_search_tool.ainvoke({
            "query": normalized_query,
            "indices": ["confluence_current", "swagger_current"],
            "intent": intent.intent if intent else "general",
            "search_type": "hybrid"
        })
        
        return {
            "search_results": search_results.get("results", [])
        }


class RewriteQueryNode(SearchNodeHandler):
    """Handles query rewriting for improved search results."""
    
    def __init__(self):
        super().__init__("rewrite_query")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query rewriting logic."""
        original_query = state.get("original_query", "")
        normalized_query = state.get("normalized_query", "")
        search_results = state.get("search_results", [])
        loop_count = state.get("loop_count", 0)
        
        # Simple rewrite strategy - expand with synonyms or rephrase
        # This is a placeholder - implement your actual rewriting logic
        rewritten_query = await self._rewrite_query(
            original_query, normalized_query, search_results
        )
        
        return {
            "normalized_query": rewritten_query,
            "loop_count": loop_count + 1
        }
    
    async def _rewrite_query(
        self, 
        original_query: str, 
        normalized_query: str, 
        search_results: List[SearchResult]
    ) -> str:
        """
        Rewrite query to improve search results.
        
        This is a placeholder implementation - replace with actual rewriting logic.
        """
        # Simple strategy: add synonyms or alternative phrasings
        if "API" in normalized_query:
            return normalized_query.replace("API", "service endpoint")
        elif "utility" in normalized_query.lower():
            return normalized_query + " service tool"
        else:
            return f"{normalized_query} documentation guide"