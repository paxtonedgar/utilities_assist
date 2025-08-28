# src/agent/nodes/processing_nodes.py
"""
Processing node handlers for combining results, generating answers, etc.

Clean implementations using base node pattern.
"""

from typing import Dict, Any, List, AsyncGenerator
import logging
from pathlib import Path
import re
from jinja2 import Environment, FileSystemLoader

from .base_node import BaseNodeHandler
from src.agent.nodes.combine import combine_node
from src.services.models import Passage, SourceChip

# Pre-import resource management and common utilities to avoid repeated dynamic imports
from src.infra.resource_manager import get_resources
from src.infra.search_config import OpenSearchConfig
from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
from src.agent.tools.search import multi_index_search_tool

logger = logging.getLogger(__name__)

# Load jinja templates for response generation
template_dir = Path(__file__).parent.parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))


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

        # Search for workflow-related content and get both context and results
        workflow_context, found_results = await self._synthesize_workflow(
            normalized_query
        )

        return self._preserve_workflow_path(
            state,
            "workflow_synthesizer",
            final_context=workflow_context,
            combined_results=found_results,  # Pass the actual results found
        )

    async def _synthesize_workflow(self, query: str) -> tuple[str, list]:
        """Synthesize multi-document workflows with step sequencing."""
        try:
            resources = self._get_resources()
            if not resources:
                logger.error("Resources not available for workflow synthesis")
                return "Unable to synthesize workflow - resources unavailable", []

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
                top_k_per_index=self._get_resources().settings.search_config.search_top_k_per_index_info,
            )

            # Phase 2: Collect and weight results by index quality
            all_results = []
            for i, result in enumerate(results_list):
                index_name = indices[i] if i < len(indices) else f"index_{i}"
                # Weight confluence higher than swagger (confluence has better CIU content)
                weight_multiplier = 1.0 if "swagger" in index_name.lower() else 1.2

                for search_result in result.results:
                    # Apply index weighting to scores
                    if hasattr(search_result, "score"):
                        search_result.score = search_result.score * weight_multiplier
                    search_result.meta["search_method"] = "workflow_synthesis"
                    search_result.meta["source_index"] = index_name
                    all_results.append(search_result)

            # Phase 3: Enhanced workflow context building with step analysis
            workflow_context = self._build_workflow_context(all_results, query)

            logger.info(f"Workflow synthesis found {len(all_results)} relevant sources")
            return workflow_context, all_results  # Return both context AND results

        except Exception as e:
            logger.error(f"Workflow synthesis failed: {e}")
            return f"Unable to synthesize workflow: {str(e)}", []

    def _build_workflow_context(self, results: List[Passage], query: str) -> str:
        """Build workflow context from multiple search results with step analysis."""
        if not results:
            return "No workflow information found for your query."

        context_parts = [
            f"**Workflow Guide: {query}**",
            "",
            "*Synthesized from multiple documentation sources*",
            "",
        ]

        # Analyze result quality for monitoring and debugging
        try:
            from src.analysis.content_quality import analyze_results

            analyze_results(results, query)  # Analysis logged automatically
        except ImportError:
            logger.debug("Content quality analyzer not available")

        # Get smart quality thresholds based on query analysis
        try:
            from src.analysis.query_analyzer import get_smart_thresholds

            smart_thresholds = get_smart_thresholds(query)
        except ImportError:
            logger.debug("Smart threshold selector not available")
            smart_thresholds = {
                "khub-opensearch-index": 0.15,
                "khub-opensearch-swagger-index": 0.05,
            }

        def get_quality_threshold(result) -> float:
            """Get quality threshold based on source index and query analysis"""
            source_index = result.meta.get("source_index", "")
            return smart_thresholds.get(source_index, 0.10)

        quality_results = [
            r for r in results if getattr(r, "score", 0) >= get_quality_threshold(r)
        ]

        # Apply score calibration to balance competition between indices
        def calibrate_score(result) -> float:
            """Calibrate scores to balance confluence vs swagger competition"""
            source_index = result.meta.get("source_index", "")
            original_score = getattr(result, "score", 0)
            if "swagger" in source_index:
                # Boost swagger scores to compete with confluence (2.5x multiplier, max 0.95)
                return min(original_score * 2.5, 0.95)
            else:
                return original_score

        # Apply calibration and sort by calibrated scores
        for result in quality_results:
            result.calibrated_score = calibrate_score(result)
        sorted_results = sorted(
            quality_results,
            key=lambda x: getattr(x, "calibrated_score", x.score),
            reverse=True,
        )

        # Group sorted results by source for better organization
        source_groups = {}
        for result in sorted_results:
            source = result.meta.get("title", "Unknown Source")
            # Add score info to source name for transparency
            source_index = result.meta.get("source_index", "")
            if source_index == "khub-opensearch-swagger-index":
                source_key = f"{source} (API Docs)"
            else:
                source_key = source

            if source_key not in source_groups:
                source_groups[source_key] = []
            source_groups[source_key].append(result)

        # Process each source group (now in score order)
        for source, source_results in source_groups.items():
            context_parts.append(f"**From {source}:**")

            for result in source_results:
                content = result.text.strip()

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


# === MIGRATED FUNCTIONS FROM SERVICES.RESPOND ===

async def generate_response(
    query: str,
    context: str,
    intent,  # Can be either dict or IntentResult
    chat_client,
    chat_history: List[Dict[str, str]] = None,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    max_tokens: int = 2500,
) -> AsyncGenerator[str, None]:
    """Generate streaming response using LLM with Jinja2 template."""
    try:
        # Use answer.jinja template instead of hardcoded prompt
        template = jinja_env.get_template("answer.jinja")

        # Ensure intent is in correct format for template
        if isinstance(intent, dict):
            template_intent = intent
        elif hasattr(intent, "intent") and hasattr(intent, "confidence"):
            template_intent = {"intent": intent.intent, "confidence": intent.confidence}
        else:
            template_intent = {
                "intent": str(intent) if intent else "unknown",
                "confidence": 0.5,
            }

        prompt = template.render(
            query=query,
            context=context,
            intent=template_intent,
            chat_history=chat_history or [],
        )

        messages = [{"role": "user", "content": prompt}]

        logger.info(
            f"Azure OpenAI request - model: {model_name}, temperature: {temperature}, max_tokens: {max_tokens}"
        )
        logger.info(f"Context length: {len(context)} chars, Query: '{query[:100]}...'")
        logger.debug(f"Full prompt preview: {prompt[:500]}...")

        try:
            response_stream = chat_client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Azure OpenAI request failed: {type(e).__name__}: {str(e)}")
            raise

        response_chunks = []
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_chunks.append(content)
                yield content

        full_response = "".join(response_chunks)
        logger.info(f"LLM response length: {len(full_response)} chars")
        logger.info(f"Response preview: {full_response[:200]}...")
        if len(full_response) > 2000:
            logger.warning(
                f"Long response ({len(full_response)} chars) - check for truncation"
            )

    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        yield f"I encountered an error while generating a response: {str(e)}"


def verify_answer(answer: str, context: str, query: str) -> Dict[str, Any]:
    """Verify answer quality and relevance."""
    metrics = {
        "has_content": len(answer.strip()) > 10,
        "not_error": not _contains_error_phrases(answer),
        "contextual": _answer_uses_context(answer, context),
        "relevant": _answer_addresses_query(answer, query),
        "complete": not _answer_seems_truncated(answer),
        "confidence_score": 0.0,
    }

    # Calculate overall confidence
    passed_checks = sum(metrics[k] for k in metrics if k != "confidence_score")
    metrics["confidence_score"] = passed_checks / 5.0

    return metrics


def extract_source_chips(
    retrieval_results: List[Passage], max_chips: int = 5
) -> List[SourceChip]:
    """Extract source citation chips from retrieval results."""
    chips = []

    for result in retrieval_results[:max_chips]:
        # Extract title from metadata or content
        title = result.meta.get("title", "")
        if not title:
            # Fallback: use first sentence of content
            sentences = result.text.split(". ")
            title = (
                sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]
            )

        # Create excerpt
        excerpt = _create_excerpt(result.text, max_length=150)

        chip = SourceChip(
            title=title,
            doc_id=result.doc_id,
            url=result.meta.get("url"),
            excerpt=excerpt,
        )
        chips.append(chip)

    return chips


# Helper functions for answer verification
def _contains_error_phrases(answer: str) -> bool:
    """Check if answer contains common error phrases."""
    error_phrases = [
        "i don't have information",
        "i cannot find",
        "no information available",
        "unable to provide",
        "i don't know",
        "insufficient information",
        "cannot determine",
        "not enough context",
        "error occurred",
        "something went wrong",
    ]
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in error_phrases)


def _answer_uses_context(answer: str, context: str) -> bool:
    """Check if answer appears to use provided context."""
    if len(context) < 50:
        return False
    
    # Extract meaningful words from context and answer
    context_words = set(re.findall(r"\w{4,}", context.lower()))
    answer_words = set(re.findall(r"\w{4,}", answer.lower()))
    
    # Check overlap (excluding common stopwords)
    overlap = context_words.intersection(answer_words)
    return len(overlap) >= max(3, len(context_words) * 0.1)


def _answer_addresses_query(answer: str, query: str) -> bool:
    """Check if answer addresses the original query."""
    query_words = set(re.findall(r"\w{3,}", query.lower()))
    answer_words = set(re.findall(r"\w{3,}", answer.lower()))
    
    overlap = query_words.intersection(answer_words)
    return len(overlap) >= max(1, len(query_words) * 0.3)


def _answer_seems_truncated(answer: str) -> bool:
    """Check if answer seems truncated or incomplete."""
    answer = answer.strip()
    if len(answer) < 20:
        return True
    
    # Check for abrupt endings
    truncation_indicators = [
        "...",
        "[truncated]",
        "[cut off]",
        "more info",
        "additional details",
    ]
    
    return any(indicator in answer.lower()[-50:] for indicator in truncation_indicators)


def _create_excerpt(text: str, max_length: int = 150) -> str:
    """Create a clean excerpt from text."""
    if len(text) <= max_length:
        return text.strip()
    
    # Try to break at sentence boundary
    sentences = text[:max_length].split(". ")
    if len(sentences) > 1:
        return ". ".join(sentences[:-1]) + "."
    
    # Break at word boundary
    words = text[:max_length].split()
    if len(words) > 1:
        return " ".join(words[:-1]) + "..."
    
    return text[:max_length - 3] + "..."

