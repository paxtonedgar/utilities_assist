# src/agent/nodes/processing_nodes.py
"""
Processing node handlers for combining results, generating answers, etc.

Clean implementations using base node pattern.
"""

from typing import Dict, Any, List
import logging

from .base_node import BaseNodeHandler
from agent.nodes.combine import combine_node
from services.respond import generate_response, extract_source_chips, verify_answer
from services.models import SearchResult

logger = logging.getLogger(__name__)


class CombineNode(BaseNodeHandler):
    """Handles combining and ranking search results."""
    
    def __init__(self):
        super().__init__("combine")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute result combination logic."""
        search_results = state.get("search_results", [])
        intent = state.get("intent")
        
        # Use existing combine_node function - MUST pass config
        combined_result = await combine_node({
            "search_results": search_results,
            "intent": intent,
            "normalized_query": state.get("normalized_query", ""),
            "workflow_path": state.get("workflow_path", [])
        }, config)
        
        return {
            "combined_results": combined_result.get("combined_results", search_results),
            "final_context": combined_result.get("final_context", "")
        }


class AnswerNode(BaseNodeHandler):
    """Handles final answer generation with streaming support."""
    
    def __init__(self):
        super().__init__("answer")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute answer generation logic."""
        normalized_query = state.get("normalized_query", "")
        final_context = state.get("final_context", "")
        combined_results = state.get("combined_results", [])
        intent = state.get("intent")
        
        # Get chat client from resources
        from infra.resource_manager import get_resources
        resources = get_resources()
        if not resources or not resources.chat_client:
            logger.error("Chat client not available for answer generation")
            return {
                "final_answer": "Unable to generate response - chat service unavailable.",
                "response_chunks": ["Unable to generate response - chat service unavailable."],
                "answer_verification": {"has_content": True, "confidence_score": 0.0},
                "source_chips": []
            }
        
        # Generate streaming response - collect all chunks
        response_chunks = []
        async for chunk in generate_response(
            query=normalized_query,
            context=final_context,
            intent=intent,
            chat_client=resources.chat_client,  # Use actual chat client from resources
            chat_history=[],
            model_name=resources.settings.chat.model,  # Use configured model name
            temperature=0.2
        ):
            response_chunks.append(chunk)
        
        final_answer = "".join(response_chunks)
        
        # Verify answer quality
        answer_verification = verify_answer(final_answer, final_context, normalized_query)
        
        # Extract source chips for UI
        source_chips = extract_source_chips(combined_results, max_chips=5)
        
        return {
            "final_answer": final_answer,
            "response_chunks": response_chunks,
            "answer_verification": answer_verification,
            "source_chips": [chip.dict() for chip in source_chips]
        }


class RestartNode(BaseNodeHandler):
    """Handles restart/reset requests."""
    
    def __init__(self):
        super().__init__("restart")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute restart logic - clear context and provide fresh start message."""
        return {
            "final_answer": "Context cleared. I'm ready to help with your next question!",
            "response_chunks": ["Context cleared. I'm ready to help with your next question!"],
            "search_results": [],
            "combined_results": [],
            "final_context": "",
            "error_messages": []
        }


class ListHandlerNode(BaseNodeHandler):
    """Handles list queries using OpenSearch aggregations."""
    
    def __init__(self):
        super().__init__("list_handler")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute list generation logic."""
        normalized_query = state.get("normalized_query", "")
        
        # Extract list type from query
        list_type = self._extract_list_type(normalized_query)
        
        # Get list from OpenSearch aggregations
        # This is a placeholder - implement actual aggregation logic
        items = await self._get_list_from_opensearch(list_type)
        
        # Format response
        formatted_response = self._format_list_response(items, list_type, normalized_query)
        
        return {
            "final_context": formatted_response,
            "combined_results": [],  # List queries don't need search results
            "workflow_path": state.get("workflow_path", []) + ["list_handler"]
        }
    
    def _extract_list_type(self, query: str) -> str:
        """Extract what type of list is being requested - REAL implementation from original."""
        query_lower = query.lower()
        
        # Real logic from original implementation with proper field mapping
        if "api" in query_lower:
            return "apis"
        elif "apg" in query_lower or "api group" in query_lower:
            return "apgs"
        elif "product" in query_lower:
            return "products"
        elif "utility" in query_lower or "service" in query_lower:
            return "utilities"
        elif "field" in query_lower or "parameter" in query_lower:
            return "fields"
        else:
            # Default to APIs if unclear (matching original logic)
            return "apis"
    
    async def _get_list_from_opensearch(self, list_type: str) -> List[str]:
        """Get unique values using OpenSearch aggregations - REAL implementation."""
        from infra.resource_manager import get_resources
        
        try:
            resources = get_resources()
            if not resources:
                logger.error("Resources not available for list aggregation")
                return []
            
            # Map list types to field names (from original implementation)
            field_mapping = {
                "apis": "api_name",
                "apgs": "apg_name", 
                "products": "product_name",
                "utilities": "utility_name",
                "fields": "field_name"
            }
            
            field_name = field_mapping.get(list_type, "api_name")
            
            # Build aggregation query
            agg_body = {
                "size": 0,  # Don't need document hits, just aggregations
                "aggs": {
                    f"unique_{list_type}": {
                        "terms": {
                            "field": f"{field_name}.keyword",  # Use keyword field for aggregations
                            "size": 1000,  # Get up to 1000 unique values
                            "order": {"_key": "asc"}  # Alphabetical order
                        }
                    }
                }
            }
            
            # Execute aggregation query using OpenSearch client
            search_client = resources.search_client
            index_name = resources.settings.search_index_alias
            
            url = f"{search_client.base_url}/{index_name}/_search"
            
            # Use same auth pattern as regular searches
            from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
            _setup_jpmc_proxy()
            aws_auth = _get_aws_auth()
            
            import requests
            if aws_auth:
                response = requests.post(url, json=agg_body, auth=aws_auth, timeout=30.0)
            else:
                response = search_client.session.post(url, json=agg_body, timeout=30.0)
            
            response.raise_for_status()
            data = response.json()
            
            # Extract unique values from aggregation results
            buckets = data.get("aggregations", {}).get(f"unique_{list_type}", {}).get("buckets", [])
            unique_values = [bucket["key"] for bucket in buckets]
            
            logger.info(f"Found {len(unique_values)} unique {list_type}")
            return unique_values
            
        except Exception as e:
            logger.error(f"OpenSearch aggregation failed: {e}")
            return []
    
    def _format_list_response(self, items: List[str], list_type: str, original_query: str) -> str:
        """Format list response in the old system's style - REAL implementation from original."""
        if not items:
            return f"No {list_type} found in the system."
        
        # Create structured list response matching old system format
        if list_type == "apis":
            response = f"I have knowledge of the following APIs:\n\n"
            for i, api in enumerate(items, 1):
                response += f"{i}. {api}\n"
        elif list_type == "apgs": 
            response = f"I have knowledge of the following APGs:\n\n"
            for i, apg in enumerate(items, 1):
                response += f"- APG: {apg}\n"
        elif list_type == "products":
            response = f"I have knowledge of the following Products:\n\n"
            for i, product in enumerate(items, 1):
                response += f"- Product Name: {product}\n"
        else:
            response = f"Available {list_type}:\n\n"
            for item in items:
                response += f"- {item}\n"
        
        # Add follow-up questions (from old system)
        if list_type == "apgs":
            response += f"\n**Follow-up questions:**\n- Which specific APG do you want to know more about?\n- What APIs are in a particular APG?\n- How do these APGs relate to specific Products?"
        elif list_type == "apis":
            response += f"\n**Follow-up questions:**\n- Which API do you want detailed information about?\n- What parameters does a specific API accept?\n- Which APG does a particular API belong to?"
        elif list_type == "products":
            response += f"\n**Follow-up questions:**\n- Which Product do you want to explore?\n- What APGs are part of a specific Product?\n- What APIs are available in a particular Product?"
        
        return response


class WorkflowSynthesizerNode(BaseNodeHandler):
    """Handles workflow/procedure synthesis queries."""
    
    def __init__(self):
        super().__init__("workflow_synthesizer")
    
    async def execute(self, state: Dict[str, Any], config: Dict = None) -> Dict[str, Any]:
        """Execute workflow synthesis logic."""
        normalized_query = state.get("normalized_query", "")
        
        # Search for workflow-related content
        # This is a placeholder - implement actual workflow search and synthesis
        workflow_steps = await self._synthesize_workflow(normalized_query)
        
        formatted_response = self._format_workflow_response(workflow_steps)
        
        return {
            "final_context": workflow_steps,
            "combined_results": [],
            "workflow_path": state.get("workflow_path", []) + ["workflow_synthesizer"]
        }
    
    async def _synthesize_workflow(self, query: str) -> str:
        """Synthesize multi-document workflows with step sequencing - REAL implementation."""
        from infra.resource_manager import get_resources
        
        try:
            resources = get_resources()
            if not resources:
                logger.error("Resources not available for workflow synthesis")
                return "Unable to synthesize workflow - resources unavailable"
            
            # Phase 1: Multi-document search with workflow focus
            indices = [
                resources.settings.search_index_alias,  # Main confluence
                "khub-opensearch-swagger-index"  # Technical procedures
            ]
            
            from agent.tools.search import multi_index_search_tool
            results_list = await multi_index_search_tool(
                indices=indices,
                query=query,
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                top_k_per_index=8  # Get more results for workflow synthesis
            )
            
            # Phase 2: Collect and analyze all results
            all_results = []
            for i, result in enumerate(results_list):
                index_name = indices[i] if i < len(indices) else f"index_{i}"
                for search_result in result.results:
                    search_result.metadata["search_method"] = "workflow_synthesis"
                    search_result.metadata["source_index"] = index_name
                    all_results.append(search_result)
            
            # Phase 3: Enhanced workflow context building with step analysis
            workflow_context = self._build_workflow_context(all_results, query)
            
            logger.info(f"Workflow synthesis found {len(all_results)} relevant sources")
            return workflow_context
            
        except Exception as e:
            logger.error(f"Workflow synthesis failed: {e}")
            return f"Unable to synthesize workflow: {str(e)}"
    
    def _build_workflow_context(self, results: List[SearchResult], query: str) -> str:
        """Build workflow context from multiple search results with step analysis."""
        if not results:
            return "No workflow information found for your query."
        
        context_parts = [
            f"**Workflow Guide: {query}**",
            "",
            "*Synthesized from multiple documentation sources*",
            ""
        ]
        
        # Group results by source for better organization
        source_groups = {}
        for result in results:
            source = result.metadata.get("title", "Unknown Source")
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(result)
        
        # Process each source group
        for source, source_results in source_groups.items():
            context_parts.append(f"**From {source}:**")
            
            for result in source_results:
                content = result.content.strip()
                
                # Look for step indicators in content
                if self._contains_step_indicators(content):
                    context_parts.append("*[Contains procedural steps]*")
                
                # Truncate with workflow awareness
                if len(content) > 400:
                    content = self._smart_workflow_truncate(content, 400)
                
                context_parts.append(content)
                context_parts.append("")  # Spacing between results
        
        return "\n".join(context_parts)
    
    def _contains_step_indicators(self, content: str) -> bool:
        """Check if content contains step/procedure indicators."""
        import re
        step_patterns = [
            r'\b\d+\.',  # "1.", "2.", etc.
            r'\bStep \d+',  # "Step 1", "Step 2", etc.
            r'\b(First|Second|Third|Next|Then|Finally|Lastly)\b',
            r'^\s*[-*]\s',  # Bullet points at start of lines
            r'\b(Before|After|Once|When)\b.*\b(complete|finish|done)\b'
        ]
        
        for pattern in step_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                return True
        return False
    
    def _smart_workflow_truncate(self, content: str, max_length: int) -> str:
        """Truncate workflow content at logical step boundaries."""
        import re
        
        if len(content) <= max_length:
            return content
        
        # Try to break at step boundaries
        step_matches = list(re.finditer(r'\b\d+\.', content))
        for match in reversed(step_matches):
            if match.start() < max_length * 0.8:  # Keep at least 80% before truncating
                truncate_point = match.start()
                return content[:truncate_point] + f"\n\n*[Additional steps available...]*"
        
        # Fallback to sentence boundary
        sentences = content[:max_length].split('. ')
        if len(sentences) > 1:
            return '. '.join(sentences[:-1]) + '.'
        
        return content[:max_length - 3] + "..."
    
    def _format_workflow_response(self, workflow_context: str) -> str:
        """Format workflow context into a readable response."""
        if not workflow_context:
            return "No workflow information found for your query."
        
        return workflow_context