# src/agent/nodes/search_nodes.py
"""
Search-specific node handlers implementing the base node pattern.

Replaces individual wrapper functions with clean, testable classes.
"""

from typing import Dict, Any, List
import logging

from .base_node import SearchNodeHandler
from src.agent.nodes.summarize import summarize_node
from src.agent.nodes.intent import intent_node
from src.agent.tools.search import adaptive_search_tool, multi_index_search_tool
from src.services.models import SearchResult

# Import constants to prevent KeyError issues
from src.agent.constants import ORIGINAL_QUERY, NORMALIZED_QUERY

# Pre-import resource management and utilities to avoid repeated dynamic imports
from src.infra.resource_manager import get_resources
from agent.nodes.base_node import get_intent_confidence
from src.infra.search_config import OpenSearchConfig

logger = logging.getLogger(__name__)


class BaseSearchNodeMixin:
    """Consolidated utilities to eliminate repeated boilerplate across search nodes."""
    
    def _get_resources(self):
        """Centralized resource access with error handling."""
        return get_resources()
    
    def _preserve_state_with_path(self, state: Dict[str, Any], node_name: str, **updates) -> Dict[str, Any]:
        """Preserve all existing state fields with standardized path tracking."""
        return {
            **state,  # Preserve all existing state
            "workflow_path": state.get("workflow_path", []) + [node_name],
            **updates
        }
    
    def _preserve_state_with_error(self, state: Dict[str, Any], node_name: str, error: str) -> Dict[str, Any]:
        """Preserve state with error tracking."""
        return {
            **state,  # Preserve all existing state
            "search_results": [],
            "workflow_path": state.get("workflow_path", []) + [f"{node_name}_error"],
            "error_messages": state.get("error_messages", []) + [error]
        }
    
    def _handle_empty_query(self, state: Dict[str, Any], node_name: str) -> Dict[str, Any]:
        """Standardized empty query handling."""
        logger.error(f"Empty query provided to {node_name}")
        return self._preserve_state_with_error(state, node_name, "Empty query provided")
    
    def _mark_search_results(self, results: List[SearchResult], search_method: str, search_id: str = None):
        """Mark search results with metadata."""
        search_id = search_id or search_method
        for search_result in results:
            search_result.metadata["search_method"] = search_method
            search_result.metadata["search_id"] = search_id
    
    def _handle_empty_retrieval(self, state: Dict[str, Any], node_name: str, query: str) -> Dict[str, Any]:
        """Standardized empty retrieval guard with suggestions."""
        logger.info(f"EMPTY {node_name.upper()} RETRIEVAL - Short-circuiting to prevent expensive LLM call")
        
        # Generate context-aware suggestions inline (only used here)
        from agent.acronym_map import UTILITY_ACRONYMS
        suggestions = []
        
        # Check if query contains an acronym
        query_upper = query.upper()
        for acronym, expansion in UTILITY_ACRONYMS.items():
            if acronym in query_upper:
                suggestions.extend([
                    f"{expansion} onboarding",
                    f"{expansion} API",
                    f"{expansion} integration guide",
                    f"create {acronym} client ID"
                ])
                break
        
        # Default suggestions if no acronym found
        if not suggestions:
            suggestions = [
                f"{query} onboarding",
                f"{query} API documentation", 
                f"{query} setup guide",
                f"utilities {query}"
            ]
        
        suggestion_text = "Try searching for: " + " | ".join(suggestions[:3])
        
        return self._preserve_state_with_path(state, f"{node_name}_empty_guard",
            search_results=[],
            combined_results=[],
            final_context=f"No documentation found for '{query}'.",
            final_answer=f"I didn't find documentation matching '{query}'. {suggestion_text}"
        )


class SummarizeNode(SearchNodeHandler, BaseSearchNodeMixin):
    """Handles query summarization and normalization."""
    
    def __init__(self):
        super().__init__("summarize")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute query summarization logic."""
        if config is None:
            config = {"configurable": {}}
        result = await summarize_node(state, config)
        return result


class IntentNode(SearchNodeHandler, BaseSearchNodeMixin):
    """Handles intent classification."""
    
    def __init__(self):
        super().__init__("intent")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute intent classification logic."""
        if config is None:
            config = {"configurable": {}}
        result = await intent_node(state, config)
        return result


class ConfluenceSearchNode(SearchNodeHandler, BaseSearchNodeMixin):
    """Handles Confluence-specific search."""
    
    def __init__(self):
        super().__init__("search_confluence")
    
    def _get_intent_based_index(self, intent: Dict[str, Any], default_index: str) -> str:
        """Select optimal index based on intent to reduce latency."""
        from agent.nodes.base_node import get_intent_label
        intent_type = get_intent_label(intent)
        
        index_mapping = {
            "definition": default_index,
            "confluence": default_index,
            "swagger": "swagger-api-index",
            "api": "swagger-api-index",
            "list": default_index,
            "workflow": default_index,
        }
        
        selected_index = index_mapping.get(intent_type, default_index)
        if selected_index != default_index:
            logger.info(f"Intent-based routing: {intent_type} → {selected_index}")
        
        return selected_index

    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute Confluence search logic with intent-based index routing."""
        try:
            resources = self._get_resources()
            query = state.get(NORMALIZED_QUERY, "")
            intent = state.get("intent")
            
            if not query or query.strip() == "":
                return self._handle_empty_query(state, "search_confluence")
            
            optimal_index = self._get_intent_based_index(intent, resources.settings.search_index_alias)
            intent_confidence = get_intent_confidence(intent)
            logger.debug(f"Using intent confidence: {intent_confidence} for confluence search")
            
            result = await adaptive_search_tool(
                query=query,
                intent_confidence=intent_confidence,
                intent_type="confluence",
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                search_index=optimal_index,
                top_k=10
            )
            
            self._mark_search_results(result.results, "confluence")
            
            if len(result.results) == 0:
                return self._handle_empty_retrieval(state, "confluence", query)
            
            return self._preserve_state_with_path(state, "search_confluence",
                search_results=result.results
            )
            
        except Exception as e:
            logger.error(f"Confluence search failed: {e}")
            return self._preserve_state_with_error(state, "search_confluence", f"Confluence search failed: {e}")


class SwaggerSearchNode(SearchNodeHandler, BaseSearchNodeMixin):
    """Handles Swagger/API documentation search."""
    
    def __init__(self):
        super().__init__("search_swagger")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute Swagger search logic."""
        try:
            resources = self._get_resources()
            query = state["normalized_query"]
            intent = state.get("intent")
            
            intent_confidence = get_intent_confidence(intent)
            logger.debug(f"Using intent confidence: {intent_confidence} for swagger search")
            
            optimal_index = OpenSearchConfig.get_swagger_index()
            logger.info(f"Using Swagger-specific index: {optimal_index}")
            
            result = await adaptive_search_tool(
                query=query,
                intent_confidence=intent_confidence,
                intent_type="swagger",
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                search_index=optimal_index,
                top_k=10
            )
            
            self._mark_search_results(result.results, "swagger")
            
            if len(result.results) == 0:
                return self._preserve_state_with_path(state, "search_swagger_empty_guard",
                    search_results=[],
                    combined_results=[],
                    final_context="No swagger/API documents found matching your query. Please try more specific terms or check if the API documentation exists.",
                    final_answer="I couldn't find any relevant API documentation for your query. Please try using different keywords or more specific terms."
                )
            
            return self._preserve_state_with_path(state, "search_swagger",
                search_results=result.results
            )
            
        except Exception as e:
            logger.error(f"Swagger search failed: {e}")
            return self._preserve_state_with_error(state, "search_swagger", f"Swagger search failed: {e}")


class MultiSearchNode(SearchNodeHandler, BaseSearchNodeMixin):
    """Handles multi-index search for complex queries."""
    
    def __init__(self):
        super().__init__("search_multi")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute multi-index search logic."""
        try:
            resources = self._get_resources()
            query = state["normalized_query"]
            intent = state.get("intent")
            
            indices = [
                OpenSearchConfig.get_default_index(),
                OpenSearchConfig.get_swagger_index()
            ]
            
            results_list = await multi_index_search_tool(
                indices=indices,
                query=query,
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                top_k_per_index=8
            )
            
            all_results = []
            for i, result in enumerate(results_list):
                index_name = indices[i] if i < len(indices) else f"index_{i}"
                for search_result in result.results:
                    search_result.metadata["search_method"] = "multi_index"
                    search_result.metadata["search_id"] = f"multi_{index_name}"
                    all_results.append(search_result)
            
            logger.info(f"Multi-search found {len(all_results)} results across {len(indices)} indices")
            
            if len(all_results) == 0:
                return self._preserve_state_with_path(state, "search_multi_empty_guard",
                    search_results=[],
                    combined_results=[],
                    final_context="No documents found matching your query. Please try more specific terms or check if the information exists in the knowledge base.",
                    final_answer="I couldn't find any relevant documents for your query. Please try using different keywords or more specific terms."
                )
            
            return self._preserve_state_with_path(state, "search_multi",
                search_results=all_results
            )
            
        except Exception as e:
            logger.error(f"Multi-index search failed: {e}")
            return self._preserve_state_with_error(state, "search_multi", f"Multi-search failed: {e}")


class RewriteQueryNode(SearchNodeHandler, BaseSearchNodeMixin):
    """Handles query rewriting for improved search results."""
    
    def __init__(self):
        super().__init__("rewrite_query")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute query rewriting logic."""
        original_query = state.get(ORIGINAL_QUERY, "")
        normalized_query = state.get(NORMALIZED_QUERY, "")
        search_results = state.get("search_results", [])
        loop_count = state.get("loop_count", 0)
        
        # Handle empty queries - don't rewrite if there's nothing to work with
        if not normalized_query or normalized_query.strip() == "":
            logger.warning("Cannot rewrite empty query, using original query")
            fallback_query = original_query or "general information"
            return self._preserve_state_with_path(state, "rewrite_query",
                **{NORMALIZED_QUERY: fallback_query},
                loop_count=loop_count + 1,
                rewrite_attempts=state.get("rewrite_attempts", 0) + 1
            )
        
        # Perform query rewriting
        rewritten_query = await self._rewrite_query(
            original_query, normalized_query, search_results
        )
        
        # Ensure we don't return an empty query
        if not rewritten_query or rewritten_query.strip() == "":
            logger.warning("Rewrite produced empty query, using fallback")
            rewritten_query = normalized_query or original_query or "general information"
        
        # Strip quotes and clean query for better BM25 recall
        rewritten_query = self._clean_rewritten_query(rewritten_query)
        
        logger.info(f"Query rewrite: '{normalized_query}' -> '{rewritten_query}'")
        
        return self._preserve_state_with_path(state, "rewrite_query",
            **{NORMALIZED_QUERY: rewritten_query},
            loop_count=loop_count + 1,
            rewrite_attempts=state.get("rewrite_attempts", 0) + 1
        )
    
    async def _rewrite_query(
        self, 
        original_query: str, 
        normalized_query: str, 
        search_results: List[SearchResult]
    ) -> str:
        """Rewrite query using LLM for better search results with domain awareness."""
        # Use centralized resource access
        resources = self._get_resources()
        
        try:
            if not resources:
                logger.warning("Resources not available, using fallback rewrite strategy")
                return self._fallback_rewrite(normalized_query)
            
            # Add domain context to prevent misinterpretation
            domain_context = """
IMPORTANT: This is JPMorgan utilities documentation.
- Focus on financial technology APIs, services, and utilities
- Preserve any utility acronyms in their technical context
- Avoid medical or non-financial domain interpretations
"""
            
            # Analyze current results to understand what's missing
            result_titles = [r.metadata.get("title", "") for r in search_results[:5]]
            result_summary = "; ".join(result_titles[:3])
            
            rewrite_prompt = f"""
            The original query "{normalized_query}" returned {len(search_results)} results, but coverage seems insufficient.
            {domain_context}
            Current results include: {result_summary}
            
            Please rewrite this query to find more comprehensive information in the utilities documentation. Consider:
            - Using different keywords or synonyms related to financial utilities/APIs
            - Being more specific about technical aspects (APIs, onboarding, configuration)
            - Expanding utility-specific terminology 
            - Focus on actionable documentation keywords
            
            CRITICAL: Preserve the original technical/utility context. Do not change domain (e.g., finance to healthcare).
            
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
                SystemMessage(content="""You are a query optimization expert for JPMorgan utility documentation. 
                Rewrite queries to improve search results within the financial technology domain.
                CRITICAL: Never change domain context (e.g., financial utilities to healthcare).
                Preserve utility acronyms and their technical context.
                Return ONLY the rewritten query, nothing else."""),
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
    
    def _clean_rewritten_query(self, query: str) -> str:
        """Clean rewritten query by removing quotes and operators that hurt BM25 recall."""
        if not query:
            return query
            
        cleaned = query.strip()
        
        # Strip surrounding quotes that reduce BM25 recall
        if (cleaned.startswith('"') and cleaned.endswith('"')) or \
           (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        
        # Remove operator-style syntax that may confuse BM25
        import re
        # Remove explicit AND/OR operators (let BM25 handle naturally)
        cleaned = re.sub(r'\s+AND\s+', ' ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\s+OR\s+', ' ', cleaned, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        cleaned = ' '.join(cleaned.split())
        
        return cleaned
    
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