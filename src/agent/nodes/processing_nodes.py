# src/agent/nodes/processing_nodes.py
"""
Processing node handlers for combining results, generating answers, etc.

Clean implementations using base node pattern.
"""

from typing import Dict, Any, List
import logging

from .base_node import BaseNodeHandler
from src.agent.nodes.combine import combine_node
from src.services.respond import generate_response, extract_source_chips, verify_answer
from src.services.models import SearchResult

# Pre-import resource management and common utilities to avoid repeated dynamic imports
from src.infra.resource_manager import get_resources
from src.infra.search_config import OpenSearchConfig
from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
from src.agent.tools.search import multi_index_search_tool

logger = logging.getLogger(__name__)


class BaseProcessingNodeMixin:
    """Consolidated utilities to eliminate repeated boilerplate across processing nodes."""

    def _get_resources(self):
        """Centralized resource access with error handling."""
        return get_resources()

    def _preserve_workflow_path(
        self, state: Dict[str, Any], node_name: str, **updates
    ) -> Dict[str, Any]:
        """Preserve state with standardized workflow path tracking."""
        return {
            **updates,
            "workflow_path": state.get("workflow_path", []) + [node_name],
        }

    def _handle_resource_unavailable(self, resource_type: str) -> Dict[str, Any]:
        """Standardized resource unavailable response."""
        logger.error(f"{resource_type} not available")
        return {
            "final_answer": f"Unable to process request - {resource_type.lower()} unavailable.",
            "response_chunks": [
                f"Unable to process request - {resource_type.lower()} unavailable."
            ],
            "answer_verification": {"has_content": True, "confidence_score": 0.0},
            "source_chips": [],
        }

    async def _execute_opensearch_aggregation(
        self, agg_body: Dict, list_type: str
    ) -> List[str]:
        """Centralized OpenSearch aggregation execution."""
        try:
            resources = self._get_resources()
            if not resources:
                logger.error("Resources not available for aggregation")
                return []

            search_client = resources.search_client
            index_name = (
                resources.settings.search_index_alias
                or OpenSearchConfig.get_default_index()
            )
            url = f"{search_client.base_url}/{index_name}/_search"

            _setup_jpmc_proxy()
            aws_auth = _get_aws_auth()

            import requests

            if aws_auth:
                response = requests.post(
                    url, json=agg_body, auth=aws_auth, timeout=30.0
                )
            else:
                response = search_client.session.post(url, json=agg_body, timeout=30.0)

            response.raise_for_status()
            data = response.json()

            buckets = (
                data.get("aggregations", {})
                .get(f"unique_{list_type}", {})
                .get("buckets", [])
            )
            unique_values = [bucket["key"] for bucket in buckets]

            logger.info(f"Found {len(unique_values)} unique {list_type}")
            return unique_values

        except Exception as e:
            logger.error(f"OpenSearch aggregation failed: {e}")
            return []


class CombineNode(BaseNodeHandler, BaseProcessingNodeMixin):
    """Handles combining and ranking search results."""

    def __init__(self):
        super().__init__("combine")

    async def execute(
        self, state: Dict[str, Any], config: Dict = None
    ) -> Dict[str, Any]:
        """Execute result combination logic."""
        search_results = state.get("search_results", [])
        intent = state.get("intent")

        # Use existing combine_node function - MUST pass config
        combined_result = await combine_node(
            {
                "search_results": search_results,
                "intent": intent,
                "normalized_query": state.get("normalized_query", ""),
                "workflow_path": state.get("workflow_path", []),
            },
            config,
        )

        return {
            "combined_results": combined_result.get("combined_results", search_results),
            "final_context": combined_result.get("final_context", ""),
        }


class AnswerNode(BaseNodeHandler, BaseProcessingNodeMixin):
    """Handles final answer generation with streaming support."""

    def __init__(self):
        super().__init__("answer")

    async def execute(
        self, state: Dict[str, Any], config: Dict = None
    ) -> Dict[str, Any]:
        """Execute answer generation using rendered content from actionability engine."""
        normalized_query = state.get("normalized_query", "")
        rendered_content = state.get("rendered_content", "")
        final_context = state.get("final_context", "")
        combined_results = state.get("combined_results", [])
        presenter_choice = state.get("presenter_choice", "")
        intent = state.get("intent")

        # HYBRID APPROACH: Use structured rendering when available, fall back to Jinja templates
        if rendered_content and rendered_content.strip():
            logger.info(
                f"Using structured content from {presenter_choice} presenter (skipping LLM)"
            )
            return {
                "final_answer": rendered_content,
                "response_chunks": [rendered_content],
                "answer_verification": {
                    "has_content": True,
                    "confidence_score": 0.9,
                    "presenter_used": presenter_choice,
                    "actionable_score": state.get("actionable_score", 0.0),
                    "suite_counts": state.get("suite_counts", {}),
                },
                "presenter_choice": presenter_choice,
            }

        # Fall back to traditional Jinja template-based LLM generation
        logger.info("Using traditional Jinja template-based LLM answer generation")

        # NO-ANSWER POLICY: Early exit if no documents found
        # This prevents 6-7s LLM calls with empty context
        if not combined_results or not final_context or not final_context.strip():
            logger.info(
                "No-answer policy triggered - no documents or context available"
            )
            no_answer_msg = (
                "I couldn't find relevant information for that query. "
                "Try rephrasing your question or being more specific about the topic you're looking for."
            )
            return {
                "final_answer": no_answer_msg,
                "response_chunks": [no_answer_msg],
                "answer_verification": {
                    "has_content": False,
                    "confidence_score": 0.0,
                    "no_docs_reason": "empty_context",
                },
                "source_chips": [],
            }

        # Get chat client from resources
        resources = self._get_resources()
        if not resources or not resources.chat_client:
            return self._handle_resource_unavailable("Chat client")

        # Generate streaming response - collect all chunks
        response_chunks = []
        async for chunk in generate_response(
            query=normalized_query,
            context=final_context,
            intent=intent,
            chat_client=resources.chat_client,  # Use actual chat client from resources
            chat_history=[],
            model_name=resources.settings.chat.model,  # Use configured model name
            temperature=0.2,
        ):
            response_chunks.append(chunk)

        final_answer = "".join(response_chunks)

        # Verify answer quality
        answer_verification = verify_answer(
            final_answer, final_context, normalized_query
        )

        # Extract source chips for UI
        source_chips = extract_source_chips(combined_results, max_chips=5)

        return {
            "final_answer": final_answer,
            "response_chunks": response_chunks,
            "answer_verification": answer_verification,
            "source_chips": [chip.dict() for chip in source_chips],
        }


class RestartNode(BaseNodeHandler, BaseProcessingNodeMixin):
    """Handles restart/reset requests."""

    def __init__(self):
        super().__init__("restart")

    async def execute(
        self, state: Dict[str, Any], config: Dict = None
    ) -> Dict[str, Any]:
        """Execute restart logic - clear context and provide fresh start message."""
        return {
            "final_answer": "Context cleared. I'm ready to help with your next question!",
            "response_chunks": [
                "Context cleared. I'm ready to help with your next question!"
            ],
            "search_results": [],
            "combined_results": [],
            "final_context": "",
            "error_messages": [],
        }


class ListHandlerNode(BaseNodeHandler, BaseProcessingNodeMixin):
    """Handles list queries using OpenSearch aggregations."""

    def __init__(self):
        super().__init__("list_handler")

    async def execute(
        self, state: Dict[str, Any], config: Dict = None
    ) -> Dict[str, Any]:
        """Execute list generation logic."""
        normalized_query = state.get("normalized_query", "")

        # Extract list type from query
        list_type = self._extract_list_type(normalized_query)

        # Get list from OpenSearch aggregations
        # This is a placeholder - implement actual aggregation logic
        items = await self._get_list_from_opensearch(list_type)

        # Format response
        formatted_response = self._format_list_response(
            items, list_type, normalized_query
        )

        return self._preserve_workflow_path(
            state,
            "list_handler",
            final_context=formatted_response,
            combined_results=[],  # List queries don't need search results
        )

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
        """Get unique values using OpenSearch aggregations."""
        # Map list types to field names
        field_mapping = {
            "apis": "api_name",
            "apgs": "apg_name",
            "products": "product_name",
            "utilities": "utility_name",
            "fields": "field_name",
        }

        field_name = field_mapping.get(list_type, "api_name")

        # Build aggregation query
        agg_body = {
            "size": 0,
            "aggs": {
                f"unique_{list_type}": {
                    "terms": {
                        "field": f"{field_name}.keyword",
                        "size": 1000,
                        "order": {"_key": "asc"},
                    }
                }
            },
        }

        return await self._execute_opensearch_aggregation(agg_body, list_type)

    def _format_list_response(
        self, items: List[str], list_type: str, original_query: str
    ) -> str:
        """Format list response in the old system's style - REAL implementation from original."""
        if not items:
            return f"No {list_type} found in the system."

        # Create structured list response matching old system format
        if list_type == "apis":
            response = "I have knowledge of the following APIs:\n\n"
            for i, api in enumerate(items, 1):
                response += f"{i}. {api}\n"
        elif list_type == "apgs":
            response = "I have knowledge of the following APGs:\n\n"
            for i, apg in enumerate(items, 1):
                response += f"- APG: {apg}\n"
        elif list_type == "products":
            response = "I have knowledge of the following Products:\n\n"
            for i, product in enumerate(items, 1):
                response += f"- Product Name: {product}\n"
        else:
            response = f"Available {list_type}:\n\n"
            for item in items:
                response += f"- {item}\n"

        # Add follow-up questions (from old system)
        if list_type == "apgs":
            response += "\n**Follow-up questions:**\n- Which specific APG do you want to know more about?\n- What APIs are in a particular APG?\n- How do these APGs relate to specific Products?"
        elif list_type == "apis":
            response += "\n**Follow-up questions:**\n- Which API do you want detailed information about?\n- What parameters does a specific API accept?\n- Which APG does a particular API belong to?"
        elif list_type == "products":
            response += "\n**Follow-up questions:**\n- Which Product do you want to explore?\n- What APGs are part of a specific Product?\n- What APIs are available in a particular Product?"

        return response


class WorkflowSynthesizerNode(BaseNodeHandler, BaseProcessingNodeMixin):
    """Handles workflow/procedure synthesis queries."""

    def __init__(self):
        super().__init__("workflow_synthesizer")

    async def execute(
        self, state: Dict[str, Any], config: Dict = None
    ) -> Dict[str, Any]:
        """Execute workflow synthesis logic."""
        normalized_query = state.get("normalized_query", "")

        # Search for workflow-related content
        # This is a placeholder - implement actual workflow search and synthesis
        workflow_steps = await self._synthesize_workflow(normalized_query)

        formatted_response = self._format_workflow_response(workflow_steps)

        return self._preserve_workflow_path(
            state,
            "workflow_synthesizer",
            final_context=workflow_steps,
            combined_results=[],
        )

    async def _synthesize_workflow(self, query: str) -> str:
        """Synthesize multi-document workflows with step sequencing."""
        try:
            resources = self._get_resources()
            if not resources:
                logger.error("Resources not available for workflow synthesis")
                return "Unable to synthesize workflow - resources unavailable"

            # Phase 1: Multi-document search with workflow focus
            indices = [
                resources.settings.search_index_alias,  # Main confluence
                OpenSearchConfig.get_swagger_index(),  # Technical procedures
            ]

            results_list = await multi_index_search_tool(
                indices=indices,
                query=query,
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                top_k_per_index=8,  # Get more results for workflow synthesis
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
            "",
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
            r"\b\d+\.",  # "1.", "2.", etc.
            r"\bStep \d+",  # "Step 1", "Step 2", etc.
            r"\b(First|Second|Third|Next|Then|Finally|Lastly)\b",
            r"^\s*[-*]\s",  # Bullet points at start of lines
            r"\b(Before|After|Once|When)\b.*\b(complete|finish|done)\b",
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
        step_matches = list(re.finditer(r"\b\d+\.", content))
        for match in reversed(step_matches):
            if match.start() < max_length * 0.8:  # Keep at least 80% before truncating
                truncate_point = match.start()
                return (
                    content[:truncate_point] + "\n\n*[Additional steps available...]*"
                )

        # Fallback to sentence boundary
        sentences = content[:max_length].split(". ")
        if len(sentences) > 1:
            return ". ".join(sentences[:-1]) + "."

        return content[: max_length - 3] + "..."

    def _format_workflow_response(self, workflow_context: str) -> str:
        """Format workflow context into a readable response."""
        if not workflow_context:
            return "No workflow information found for your query."

        return workflow_context
