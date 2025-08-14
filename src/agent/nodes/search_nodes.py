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

# Import constants to prevent KeyError issues
from agent.constants import ORIGINAL_QUERY, NORMALIZED_QUERY

logger = logging.getLogger(__name__)


class SummarizeNode(SearchNodeHandler):
    """Handles query summarization and normalization."""
    
    def __init__(self):
        super().__init__("summarize")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query summarization logic."""
        # Use existing summarize_node function with proper config parameter
        # Note: The original function expects (state, config, store=None)
        # We provide a minimal config for compatibility
        config = {"configurable": {}}
        result = await summarize_node(state, config)
        return result


class IntentNode(SearchNodeHandler):
    """Handles intent classification."""
    
    def __init__(self):
        super().__init__("intent")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute intent classification logic."""
        # Use existing intent_node function with proper config parameter
        # Note: The original function expects (state, config, store=None)
        # We provide a minimal config for compatibility
        config = {"configurable": {}}
        result = await intent_node(state, config)
        return result


class ConfluenceSearchNode(SearchNodeHandler):
    """Handles Confluence-specific search."""
    
    def __init__(self):
        super().__init__("search_confluence")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Confluence search logic - REAL implementation from original."""
        try:
            # Extract resources from global resource manager
            from infra.resource_manager import get_resources
            resources = get_resources()
            
            query = state.get(NORMALIZED_QUERY, "")
            intent = state.get("intent")
            
            # Handle empty query to prevent infinite loops
            if not query or query.strip() == "":
                logger.error("Empty query provided to Confluence search")
                return {
                    "search_results": [],
                    "workflow_path": state.get("workflow_path", []) + ["search_confluence_error"],
                    "error_messages": state.get("error_messages", []) + ["Empty query provided"]
                }
            
            result = await adaptive_search_tool(
                query=query,
                intent_confidence=intent.confidence if intent else 0.5,
                intent_type="confluence",
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                search_index=resources.settings.search_index_alias,
                top_k=10
            )
            
            # Mark results with search method
            for search_result in result.results:
                search_result.metadata["search_method"] = "confluence"
                search_result.metadata["search_id"] = "confluence"
            
            return {
                "search_results": result.results,
                "workflow_path": state.get("workflow_path", []) + ["search_confluence"]
            }
            
        except Exception as e:
            logger.error(f"Confluence search failed: {e}")
            return {
                "search_results": [],
                "workflow_path": state.get("workflow_path", []) + ["search_confluence_error"],
                "error_messages": state.get("error_messages", []) + [f"Confluence search failed: {e}"]
            }


class SwaggerSearchNode(SearchNodeHandler):
    """Handles Swagger/API documentation search."""
    
    def __init__(self):
        super().__init__("search_swagger")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Swagger search logic - REAL implementation from original."""
        try:
            # Extract resources from global resource manager
            from infra.resource_manager import get_resources
            resources = get_resources()
            
            query = state["normalized_query"]
            intent = state.get("intent")
            
            result = await adaptive_search_tool(
                query=query,
                intent_confidence=intent.confidence if intent else 0.5,
                intent_type="swagger",
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                search_index="khub-opensearch-swagger-index",
                top_k=10
            )
            
            # Mark results with search method
            for search_result in result.results:
                search_result.metadata["search_method"] = "swagger"
                search_result.metadata["search_id"] = "swagger"
            
            return {
                "search_results": result.results,
                "workflow_path": state.get("workflow_path", []) + ["search_swagger"]
            }
            
        except Exception as e:
            logger.error(f"Swagger search failed: {e}")
            return {
                "search_results": [],
                "workflow_path": state.get("workflow_path", []) + ["search_swagger_error"],
                "error_messages": state.get("error_messages", []) + [f"Swagger search failed: {e}"]
            }


class MultiSearchNode(SearchNodeHandler):
    """Handles multi-index search for complex queries."""
    
    def __init__(self):
        super().__init__("search_multi")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute multi-index search logic - REAL implementation from original."""
        try:
            # Extract resources from global resource manager
            from infra.resource_manager import get_resources
            resources = get_resources()
            
            query = state["normalized_query"]
            intent = state.get("intent")
            
            # Define indices to search for compound queries
            indices = [
                resources.settings.search_index_alias,  # Use configured index
                "khub-opensearch-swagger-index"
            ]
            
            # Search all indices
            results_list = await multi_index_search_tool(
                indices=indices,
                query=query,
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                top_k_per_index=8  # Get fewer per index to avoid overwhelming context
            )
            
            # Combine results from all indices
            all_results = []
            for i, result in enumerate(results_list):
                index_name = indices[i] if i < len(indices) else f"index_{i}"
                for search_result in result.results:
                    search_result.metadata["search_method"] = "multi_index"
                    search_result.metadata["search_id"] = f"multi_{index_name}"
                    all_results.append(search_result)
            
            logger.info(f"Multi-search found {len(all_results)} results across {len(indices)} indices")
            
            return {
                "search_results": all_results,
                "workflow_path": state.get("workflow_path", []) + ["search_multi"]
            }
            
        except Exception as e:
            logger.error(f"Multi-index search failed: {e}")
            return {
                "search_results": [],
                "workflow_path": state.get("workflow_path", []) + ["search_multi_error"],
                "error_messages": state.get("error_messages", []) + [f"Multi-search failed: {e}"]
            }


class RewriteQueryNode(SearchNodeHandler):
    """Handles query rewriting for improved search results."""
    
    def __init__(self):
        super().__init__("rewrite_query")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query rewriting logic."""
        original_query = state.get(ORIGINAL_QUERY, "")
        normalized_query = state.get(NORMALIZED_QUERY, "")
        search_results = state.get("search_results", [])
        loop_count = state.get("loop_count", 0)
        
        # Handle empty queries - don't rewrite if there's nothing to work with
        if not normalized_query or normalized_query.strip() == "":
            logger.warning("Cannot rewrite empty query, using original query")
            fallback_query = original_query or "general information"
            return {
                NORMALIZED_QUERY: fallback_query,
                "loop_count": loop_count + 1,
                "rewrite_attempts": state.get("rewrite_attempts", 0) + 1
            }
        
        # Perform query rewriting
        rewritten_query = await self._rewrite_query(
            original_query, normalized_query, search_results
        )
        
        # Ensure we don't return an empty query
        if not rewritten_query or rewritten_query.strip() == "":
            logger.warning("Rewrite produced empty query, using fallback")
            rewritten_query = normalized_query or original_query or "general information"
        
        logger.info(f"Query rewrite: '{normalized_query}' -> '{rewritten_query}'")
        
        return {
            NORMALIZED_QUERY: rewritten_query,
            "loop_count": loop_count + 1,
            "rewrite_attempts": state.get("rewrite_attempts", 0) + 1
        }
    
    async def _rewrite_query(
        self, 
        original_query: str, 
        normalized_query: str, 
        search_results: List[SearchResult]
    ) -> str:
        """Rewrite query using LLM for better search results - REAL implementation."""
        from infra.resource_manager import get_resources
        
        try:
            resources = get_resources()
            if not resources:
                logger.warning("Resources not available, using fallback rewrite strategy")
                return self._fallback_rewrite(normalized_query)
            
            # Analyze current results to understand what's missing
            result_titles = [r.metadata.get("title", "") for r in search_results[:5]]
            result_summary = "; ".join(result_titles[:3])
            
            rewrite_prompt = f"""
            The original query "{normalized_query}" returned {len(search_results)} results, but coverage seems insufficient.
            
            Current results include: {result_summary}
            
            Please rewrite this query to find more comprehensive information. Consider:
            - Using different keywords or synonyms
            - Being more specific about what's needed
            - Expanding the scope if the query was too narrow
            - Focusing on key concepts if the query was too broad
            
            Rewritten query:
            """
            
            # Use the working Azure OpenAI client configuration
            from langchain_openai import AzureChatOpenAI
            from langchain_core.messages import HumanMessage, SystemMessage
            
            # Get authentication parameters from working clients.py approach
            from utils import load_config
            import os
            
            # Load config for API key (matching clients.py pattern)
            auth_config = load_config()
            api_key = auth_config.get('azure_openai', 'api_key', fallback=None)
            
            # Get Bearer token from token provider if available
            headers = {"user_sid": os.getenv("JPMC_USER_SID", "REPLACE")}
            if resources.token_provider:
                try:
                    bearer_token = resources.token_provider()
                    headers["Authorization"] = f"Bearer {bearer_token}"
                except Exception as e:
                    logger.warning(f"Bearer token failed, using API key only: {e}")
            
            # Create LangChain client with same auth pattern as clients.py
            langchain_client = AzureChatOpenAI(
                api_version=resources.settings.chat.api_version,
                azure_deployment=resources.settings.chat.model,
                azure_endpoint=resources.settings.chat.api_base,
                api_key=api_key,
                default_headers=headers,
                temperature=0.1,
                max_tokens=200
            )
            
            response = await langchain_client.ainvoke([
                SystemMessage(content="You are a query optimization expert. Rewrite queries to improve search results. Return ONLY the rewritten query, nothing else."),
                HumanMessage(content=rewrite_prompt)
            ])
            
            # Extract actual query from LLM response (often contains explanations)
            raw_response = response.content.strip()
            rewritten_query = self._extract_query_from_response(raw_response, normalized_query)
            
            logger.info(f"Query rewritten: '{normalized_query}' -> '{rewritten_query}'")
            
            return rewritten_query
            
        except Exception as e:
            logger.error(f"LLM query rewriting failed: {e}")
            return self._fallback_rewrite(normalized_query)
    
    def _extract_query_from_response(self, raw_response: str, original_query: str) -> str:
        """Extract actual query from LLM response that may contain explanations."""
        # Handle empty/whitespace responses
        if not raw_response or raw_response.strip() == "":
            logger.warning("Empty response from LLM, using fallback")
            return self._fallback_rewrite(original_query)
        
        # Look for patterns that indicate actual queries
        import re
        
        # Pattern 1: Look for "Rewritten query:" followed by actual query
        rewritten_pattern = r'(?i)rewritten\s+query:?\s*(.+?)(?:\n|$)'
        match = re.search(rewritten_pattern, raw_response)
        if match:
            extracted = match.group(1).strip().strip('"\'')
            if len(extracted) > 3 and len(extracted) < 200:  # Reasonable query length
                return extracted
        
        # Pattern 2: Look for queries in quotes  
        quote_pattern = r'["\']([^"\'\n]{10,100})["\']'
        matches = re.findall(quote_pattern, raw_response)
        if matches:
            # Pick the longest reasonable match
            valid_matches = [m for m in matches if len(m.strip()) > 5 and not m.startswith("Example:")]
            if valid_matches:
                return max(valid_matches, key=len).strip()
        
        # Pattern 3: Last resort - take first sentence that's reasonable length
        sentences = raw_response.split('\n')
        for sentence in sentences:
            sentence = sentence.strip()
            if (10 <= len(sentence) <= 150 and 
                not sentence.startswith(('To ', 'Please ', 'Consider ', 'Here ', 'Original ', 'Since ')) and
                not sentence.endswith((':',))):
                return sentence
        
        # If all else fails, use fallback
        logger.warning("Could not extract query from LLM response, using fallback")
        return self._fallback_rewrite(original_query)
    
    def _fallback_rewrite(self, query: str) -> str:
        """Fallback rewrite strategy when LLM is unavailable - REAL implementation."""
        # Enhanced fallback strategy with comprehensive synonym mapping
        import re
        
        # Create a copy to modify
        rewritten = query.lower()
        
        # API terminology expansions
        api_synonyms = {
            r'\bapi\b': 'service endpoint interface',
            r'\bendpoint\b': 'API service method',
            r'\brest\b': 'RESTful web service',
            r'\bhttp\b': 'web protocol request'
        }
        
        # Utility and service expansions  
        utility_synonyms = {
            r'\butility\b': 'service tool component',
            r'\bservice\b': 'utility component system',
            r'\btool\b': 'utility service function'
        }
        
        # Technical concept expansions
        technical_synonyms = {
            r'\bauthentication\b': 'auth security credential validation',
            r'\bauthorization\b': 'permission access control security',
            r'\bconfigur\w*\b': 'setup configuration parameter settings',
            r'\berror\b': 'exception failure issue problem',
            r'\bresponse\b': 'result output return data'
        }
        
        # Apply synonym mappings
        all_synonyms = {**api_synonyms, **utility_synonyms, **technical_synonyms}
        
        for pattern, replacement in all_synonyms.items():
            if re.search(pattern, rewritten):
                rewritten = re.sub(pattern, replacement, rewritten)
                break  # Only apply first matching synonym to avoid over-expansion
        
        # Add context hints based on query intent
        if any(term in query.lower() for term in ['how', 'configure', 'setup', 'install']):
            rewritten += " procedure steps guide documentation"
        elif any(term in query.lower() for term in ['what', 'define', 'explain']):
            rewritten += " definition explanation overview"
        elif any(term in query.lower() for term in ['list', 'show', 'all', 'available']):
            rewritten += " available options catalog inventory"
        else:
            rewritten += " documentation guide reference"
        
        return rewritten.title()  # Return with proper capitalization