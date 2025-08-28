"""
Enhanced Streamlined LLM Orchestrator for Conversational RAG.

Surgically enhanced version that adds structured planning and verification
to the existing orchestrator without creating parallel workflows.
"""

import logging
import json
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PlanningMode(str, Enum):
    """Planning modes for different query types."""

    SIMPLE = "simple"  # Regex-based planning
    STRUCTURED = "structured"  # LLM-based JSON planning


@dataclass
class EnhancedPlan:
    """Enhanced execution plan with structured output support."""

    steps: List[Dict[str, Any]]
    confidence: float = 0.8
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None
    expected_answer_shape: str = "definition"
    reasoning: str = ""
    planning_mode: PlanningMode = PlanningMode.SIMPLE


class Orchestrator:
    """Enhanced orchestrator with structured planning and verification."""

    def __init__(self, chat_client=None, settings=None):
        self.chat_client = chat_client
        self.settings = settings
        self.use_llm = chat_client is not None

        # Performance budgets (configurable from settings or defaults)
        if settings and hasattr(settings, "orchestrator"):
            self.max_total_time_ms = settings.orchestrator.max_total_time_ms
            self.max_planning_time_ms = self.max_total_time_ms // 5  # 20% for planning
            self.max_verification_time_ms = (
                self.max_total_time_ms // 7
            )  # ~14% for verification
        else:
            self.max_total_time_ms = 4000  # p95 ≤ 4s
            self.max_planning_time_ms = 800  # Planning budget
            self.max_verification_time_ms = 600  # Verification budget

    async def orchestrate(
        self,
        query: str,
        resources: Any,
        context: Optional[List[Dict]] = None,
        enable_verification: bool = True,
    ) -> Dict[str, Any]:
        """
        Enhanced orchestration: plan → execute → verify → respond.

        Adds structured planning and optional verification while maintaining
        compatibility with existing search infrastructure.
        """
        start_time = time.time()
        execution_stats = {}

        try:
            # 1. Enhanced Planning (with hard timeout)
            plan_start = time.time()

            try:
                if self.use_llm:
                    plan = await asyncio.wait_for(
                        self._structured_llm_plan(query, context),
                        timeout=self.max_planning_time_ms / 1000,
                    )
                else:
                    plan = self._enhanced_simple_plan(query)
            except asyncio.TimeoutError:
                logger.warning(
                    f"Planning timed out after {self.max_planning_time_ms}ms, using fallback"
                )
                plan = self._enhanced_simple_plan(query)

            execution_stats["planning_ms"] = (time.time() - plan_start) * 1000

            # Handle clarifying questions
            if plan.needs_clarification:
                return {
                    "ask_clarification": plan.clarifying_question,
                    "plan_confidence": plan.confidence,
                    "execution_stats": execution_stats,
                }

            # 2. Execute search (enhanced with plan context and timeout)
            search_start = time.time()

            try:
                remaining_time = self.max_total_time_ms / 1000 - (
                    time.time() - start_time
                )
                search_timeout = min(
                    remaining_time * 0.6, 3.0
                )  # 60% of remaining time, max 3s

                search_results = await asyncio.wait_for(
                    self._execute_enhanced_search(query, plan, resources),
                    timeout=search_timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    f"Search timed out after {search_timeout:.1f}s, using fallback"
                )
                search_results = await self._fallback_search(query, resources)

            execution_stats["search_ms"] = (time.time() - search_start) * 1000

            # 3. Generate draft answer
            answer_start = time.time()

            draft_answer = await self._generate_shaped_answer(
                query, search_results, plan, resources
            )

            execution_stats["answer_ms"] = (time.time() - answer_start) * 1000

            # 4. Optional verification (with plan context)
            final_answer = draft_answer
            verification_result = None

            if enable_verification and self.use_llm and len(search_results) > 0:
                verify_start = time.time()

                # Store plan context for verification
                self._current_plan = plan

                try:
                    remaining_time = self.max_total_time_ms / 1000 - (
                        time.time() - start_time
                    )
                    verify_timeout = min(
                        remaining_time * 0.8, self.max_verification_time_ms / 1000
                    )

                    verification_result = await asyncio.wait_for(
                        self._verify_answer_quality(
                            draft_answer, search_results, query, resources
                        ),
                        timeout=verify_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Verification timed out after {verify_timeout:.1f}s, skipping"
                    )
                    verification_result = {
                        "verification_score": 0.7,
                        "needs_improvement": False,
                        "timeout": True,
                    }

                execution_stats["verification_ms"] = (time.time() - verify_start) * 1000

                # Apply verification improvements
                if verification_result.get("needs_improvement"):
                    final_answer = await self._improve_answer_with_verification(
                        draft_answer,
                        verification_result,
                        query,
                        search_results,
                        resources,
                    )

                # Clean up plan context
                if hasattr(self, "_current_plan"):
                    delattr(self, "_current_plan")

            execution_stats["total_ms"] = (time.time() - start_time) * 1000

            # Prepare plan hints for downstream graph propagation
            plan_hints = {
                "expected_answer_shape": plan.expected_answer_shape,
                "recognized_utility": getattr(plan, "recognized_utility", None),
                "search_policy": getattr(plan, "search_policy", "general"),
                "orchestrator_trace_id": getattr(
                    plan, "orchestrator_trace_id", f"orch_{hash(query) % 10000}"
                ),
                "planning_mode": plan.planning_mode.value,
            }

            return {
                "final_answer": final_answer,
                "search_results": search_results,
                "plan_confidence": plan.confidence,
                "plan_reasoning": plan.reasoning,
                "expected_answer_shape": plan.expected_answer_shape,
                "verification_result": verification_result,
                "execution_stats": execution_stats,
                "plan_hints": plan_hints,  # For downstream graph nodes
                "orchestrator_used": True,
            }

        except Exception as e:
            logger.error(f"Enhanced orchestration failed: {e}")
            # Fallback to simple orchestration
            return await self._fallback_orchestration(query, resources)

    async def _structured_llm_plan(
        self, query: str, context: Optional[List[Dict]]
    ) -> EnhancedPlan:
        """Create structured plan using LLM with JSON schema."""
        try:
            # Build planning prompt with schema
            system_prompt = """You are a search planner. Return JSON only with this exact schema:
{
  "needs_clarification": boolean,
  "clarifying_question": "string (if needs_clarification=true)",
  "filters": {"key": "value"} or null,
  "expected_answer_shape": "definition|how_to_steps|api_reference|troubleshooting",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation"
}

Guidelines:
- Ask clarifying questions if the query is ambiguous about which utility (CIU vs ETU)
- "What is X?" → definition
- "How to do X?" → how_to_steps  
- "API/endpoint" → api_reference
- "error/issue" → troubleshooting"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ]

            # Add context if available
            if context:
                context_text = self._format_context(context)
                messages.append(
                    {"role": "system", "content": f"Context: {context_text}"}
                )

            # Call LLM with timeout
            response = await self.chat_client.chat.completions.acreate(
                model=self.settings.chat.model if self.settings else "gpt-4",
                messages=messages,
                max_tokens=250,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            plan_data = json.loads(response.choices[0].message.content.strip())

            return EnhancedPlan(
                steps=[{"action": "search", "filters": plan_data.get("filters")}],
                confidence=plan_data.get("confidence", 0.8),
                needs_clarification=plan_data.get("needs_clarification", False),
                clarifying_question=plan_data.get("clarifying_question"),
                expected_answer_shape=plan_data.get(
                    "expected_answer_shape", "definition"
                ),
                reasoning=plan_data.get("reasoning", "LLM structured plan"),
                planning_mode=PlanningMode.STRUCTURED,
            )

        except Exception as e:
            logger.warning(
                f"Structured LLM planning failed: {e}, falling back to simple"
            )
            return self._enhanced_simple_plan(query)

    def _enhanced_simple_plan(self, query: str) -> EnhancedPlan:
        """Regex-based planning using existing slotter."""
        from src.agent.intent.slotter import slot

        slot_result = slot(query)

        # Simple mapping to search filters (TODO: Implement filtering)
        # filters = None
        if slot_result.doish:
            # filters = {"content_type": "procedure"}  # TODO: Implement filtering
            pass
        elif "api" in query.lower():
            # filters = {"index": "swagger"}  # TODO: Implement filtering
            pass

    def _enhanced_simple_plan(self, query: str) -> EnhancedPlan:
        """Enhanced regex-based planning using existing slotter."""
        try:
            from src.agent.intent.slotter import slot

            slot_result = slot(query)

            # Enhanced intent → shape mapping
            filters = None
            answer_shape = "definition"

            if slot_result.doish:
                answer_shape = "how_to_steps"
                filters = {"content_type": "procedure"}
            elif "api" in query.lower() or "endpoint" in query.lower():
                answer_shape = "api_reference"
                filters = {"index": "swagger"}
            elif any(
                word in query.lower() for word in ["error", "issue", "trouble", "fix"]
            ):
                answer_shape = "troubleshooting"

            return EnhancedPlan(
                steps=[{"action": "search", "filters": filters}],
                confidence=slot_result.confidence,
                needs_clarification=False,
                expected_answer_shape=answer_shape,
                reasoning=f"Regex-based plan: {slot_result.intent if hasattr(slot_result, 'intent') else 'simple'}",
                planning_mode=PlanningMode.SIMPLE,
            )

        except Exception as e:
            logger.error(f"Enhanced simple planning failed: {e}")
            return EnhancedPlan(
                steps=[{"action": "search"}],
                confidence=0.5,
                reasoning="Emergency fallback plan",
            )

    async def _execute_enhanced_search(
        self, query: str, plan: EnhancedPlan, resources
    ) -> List[Any]:
        """
        Execute utility-anchored retrieval: Pass 1 (high-precision) + Pass 2 (recall).

        This directly addresses CIU runbook surfacing by anchoring to recognized utilities.
        """
        try:
            # Prepare search strategy and imports
            recognized_utility, search_policy, all_results = self._prepare_search_strategy(query)
            
            # PASS 1: Utility-anchored high-precision search
            if recognized_utility:
                utility_results = await self._execute_utility_anchored_search(
                    query, recognized_utility, plan, resources
                )
                all_results.extend(utility_results)
            
            # PASS 2: General recall search (always run for coverage)
            general_results = await self._execute_general_search(
                query, plan, resources, recognized_utility
            )
            all_results.extend(general_results)
            
            # Process and return final results
            return self._process_search_results(
                all_results, recognized_utility, search_policy, plan, query
            )
            
        except Exception as e:
            logger.error(f"Enhanced search failed completely: {e}")
            return []

    def _extract_utility_from_query(self, query: str) -> Optional[str]:
        """Extract recognized utility name from query for anchored retrieval."""
        query_lower = query.lower()

        # Comprehensive utility mapping
        utility_patterns = {
            "Customer Interaction Utility": [
                "ciu",
                "customer interaction utility",
                "customer interaction",
                "customer utility",
                "interaction utility",
            ],
            "Enhanced Transaction Utility": [
                "etu",
                "enhanced transaction utility",
                "transaction utility",
                "enhanced transaction",
                "transaction processing",
            ],
            "Account Utility": [
                "au",
                "account utility",
                "accounts utility",
                "account management",
            ],
            "Customer Summary Utility": [
                "csu",
                "customer summary utility",
                "customer summary",
                "summary utility",
            ],
            "Payment Card Utility": [
                "pcu",
                "payment card utility",
                "payment utility",
                "card utility",
            ],
        }

        # Check for exact matches first (higher precision)
        for utility_name, patterns in utility_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    # Prefer longer, more specific matches
                    if len(pattern) >= 3:  # Skip very short patterns like "au"
                        logger.info(
                            f"Recognized utility: '{utility_name}' from pattern '{pattern}'"
                        )
                        return utility_name

        # Fallback: check short acronyms only if they're whole words
        import re

        for utility_name, patterns in utility_patterns.items():
            for pattern in patterns:
                if len(pattern) <= 3:  # Short acronyms
                    if re.search(rf"\b{re.escape(pattern)}\b", query_lower):
                        logger.info(
                            f"Recognized utility: '{utility_name}' from acronym '{pattern}'"
                        )
                        return utility_name

        return None

    def _weighted_fusion_utility_first(
        self,
        all_results: List[Any],
        recognized_utility: Optional[str],
        plan: EnhancedPlan,
    ) -> List[Any]:
        """Fuse results with utility-anchored results preferred."""
        if not all_results:
            return []

        # Separate utility-anchored from general results
        utility_results = [
            r for r in all_results if getattr(r, "utility_anchored", False)
        ]
        general_results = [
            r for r in all_results if not getattr(r, "utility_anchored", False)
        ]

        logger.info(
            f"Fusion: {len(utility_results)} utility-anchored, {len(general_results)} general"
        )

        # Apply weighted scoring
        for result in utility_results:
            original_score = getattr(result, "score", 0.5)
            result.weighted_score = min(
                original_score * 1.5, 0.95
            )  # 50% boost, cap at 0.95

        for result in general_results:
            result.weighted_score = getattr(result, "score", 0.5)

        # Sort by weighted score, utility-anchored first in ties
        combined_results = utility_results + general_results
        combined_results.sort(
            key=lambda x: (
                getattr(x, "weighted_score", 0),
                getattr(x, "utility_anchored", False),
            ),
            reverse=True,
        )

        # Limit to reasonable size
        max_results = (
            self.settings.search_config.rrf_unique_limit if self.settings else 20
        )
        final_results = combined_results[:max_results]

        return final_results

    def _analyze_citations(self, citations: List[str], sentences: List[str], results: List[Any], resources) -> Dict[str, Any]:
        """Analyze citation coverage and presence."""
        # Citation coverage by sentences
        sentences_with_citations = sum(
            1 for sentence in sentences 
            if any(citation in sentence for citation in citations)
        )
        
        sentence_citation_coverage = sentences_with_citations / max(len(sentences), 1)
        citation_threshold = (
            resources.settings.search_config.citation_coverage_threshold
            if resources and resources.settings else 0.6
        )
        
        return {
            "has_citations": len(citations) > 0,
            "result_citation_coverage": len(citations) / max(len(results), 1) if results else 0,
            "sentence_citation_coverage": sentence_citation_coverage,
            "citation_threshold": citation_threshold,
            "citation_coverage_ok": sentence_citation_coverage >= citation_threshold,
            "sentences_with_citations": sentences_with_citations,
        }
    
    def _check_anchor_requirements(self, citations: List[str], results: List[Any]) -> Dict[str, Any]:
        """Check anchor presence requirements for how-to queries."""
        anchor_check_passed = True
        anchor_details = {"checked": False, "found": False, "relevant_anchors": []}
        
        if (
            hasattr(self, "_current_plan")
            and getattr(self._current_plan, "expected_answer_shape", "") == "how_to_steps"
        ):
            anchor_check_passed, anchor_details = self._check_anchor_presence(citations, results)
        
        return {
            "anchor_check_passed": anchor_check_passed,
            "anchor_details": anchor_details,
        }
    
    def _calculate_grounding_score(self, answer: str, results: List[Any]) -> float:
        """Calculate answer grounding score based on word overlap."""
        answer_words = set(answer.lower().split())
        result_words = set()
        
        # Use full body text from results for better grounding
        for result in results[:5]:  # Check top 5 results
            text = self._get_full_body_text(result)
            result_words.update(text.lower().split()[:100])  # More words for better coverage
        
        return len(answer_words & result_words) / max(len(answer_words), 1)
    
    def _calculate_utility_relevance(self, answer: str) -> float:
        """Calculate utility-specific content relevance score."""
        if not (hasattr(self, "_current_plan") and getattr(self._current_plan, "recognized_utility", None)):
            return 1.0
        
        return self._check_utility_content_relevance(
            answer, self._current_plan.recognized_utility
        )
    
    def _calculate_verification_score(
        self, citation_analysis: Dict, anchor_analysis: Dict, grounding_score: float, utility_score: float
    ) -> float:
        """Calculate composite verification score."""
        return (
            0.25 * (1.0 if citation_analysis["has_citations"] else 0.0)  # Basic citations
            + 0.30 * citation_analysis["sentence_citation_coverage"]  # Citation density
            + 0.20 * (1.0 if anchor_analysis["anchor_check_passed"] else 0.5)  # Anchor relevance
            + 0.15 * min(grounding_score * 2, 1.0)  # Grounding
            + 0.10 * utility_score  # Utility relevance
        )
    
    def _requires_improvement(
        self, verification_score: float, citation_analysis: Dict, anchor_analysis: Dict
    ) -> bool:
        """Determine if answer quality improvement is needed."""
        return (
            verification_score < 0.65
            or not citation_analysis["citation_coverage_ok"]
            or not anchor_analysis["anchor_check_passed"]
        )
    
    def _generate_improvement_suggestions(
        self, citation_analysis: Dict, anchor_analysis: Dict, grounding_score: float, utility_score: float
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        # Citation suggestions
        if not citation_analysis["has_citations"]:
            suggestions.append("Add citations to support claims")
        elif citation_analysis["sentence_citation_coverage"] < citation_analysis["citation_threshold"]:
            coverage = citation_analysis["sentence_citation_coverage"]
            threshold = citation_analysis["citation_threshold"]
            suggestions.append(
                f"Improve citation coverage: only {coverage:.1%} of sentences have citations (need ≥{threshold:.1%})"
            )
        
        # Anchor suggestions
        if not anchor_analysis["anchor_check_passed"] and anchor_analysis["anchor_details"].get("checked"):
            suggestions.append("Include procedural anchors (setup/onboarding/configuration sections)")
        
        # Grounding suggestions
        if grounding_score < 0.3:
            suggestions.append("Improve answer grounding to retrieved content")
        
        # Utility relevance suggestions
        if utility_score < 0.7:
            suggestions.append("Ensure answer focuses on the specific utility mentioned")
        
        return suggestions
    
    def _build_verification_result(
        self, verification_score: float, citation_analysis: Dict, anchor_analysis: Dict,
        grounding_score: float, utility_score: float, needs_improvement: bool, 
        suggestions: List[str], sentences: List[str]
    ) -> Dict[str, Any]:
        """Build final verification result dictionary."""
        return {
            "verification_score": verification_score,
            "has_citations": citation_analysis["has_citations"],
            "citation_coverage": citation_analysis["result_citation_coverage"],
            "sentence_citation_coverage": citation_analysis["sentence_citation_coverage"],
            "citation_coverage_ok": citation_analysis["citation_coverage_ok"],
            "anchor_check": anchor_analysis["anchor_details"],
            "anchor_check_passed": anchor_analysis["anchor_check_passed"],
            "word_overlap": grounding_score,
            "utility_content_score": utility_score,
            "needs_improvement": needs_improvement,
            "suggestions": suggestions,
            "sentences_analyzed": len(sentences),
            "sentences_with_citations": citation_analysis["sentences_with_citations"],
        }

    async def _generate_shaped_answer(
        self, query: str, results: List[Any], plan: EnhancedPlan, resources
    ) -> str:
        """Generate answer shaped according to plan using existing AnswerNode."""
        try:
            from src.agent.nodes.processing_nodes import AnswerNode

            # Format results as context
            final_context = self._format_results_as_context(results)

            # Use existing answer node
            answer_node = AnswerNode()
            answer_result = await answer_node.execute(
                {
                    "normalized_query": query,
                    "final_context": final_context,
                    "combined_results": results,
                    "workflow_path": ["orchestrated"],
                }
            )

            generated_answer = answer_result.get(
                "final_answer", "Unable to generate answer."
            )

            # Add answer shaping based on plan
            if plan.expected_answer_shape == "how_to_steps":
                # Ensure steps are clearly formatted
                if "step" not in generated_answer.lower():
                    generated_answer = (
                        f"**Steps to {query.lower()}:**\n\n{generated_answer}"
                    )
            elif plan.expected_answer_shape == "api_reference":
                # Ensure API format
                if (
                    "endpoint" not in generated_answer.lower()
                    and "api" not in generated_answer.lower()
                ):
                    generated_answer = (
                        f"**API Reference for {query}:**\n\n{generated_answer}"
                    )

            return generated_answer

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer due to processing error."

    async def _verify_answer_quality(
        self, answer: str, results: List[Any], query: str, resources
    ) -> Dict[str, Any]:
        """Enhanced verification with citation coverage and anchor presence heuristics."""
        try:
            # Extract citations and sentences for analysis
            citations = self._extract_citations(answer)
            sentences = self._extract_sentences_for_verification(answer)
            # result_ids = {getattr(r, "doc_id", str(i)) for i, r in enumerate(results)}  # Currently unused

            # 1. Citation coverage by sentences (≥60% threshold)
            sentences_with_citations = 0
            for sentence in sentences:
                if any(citation in sentence for citation in citations):
                    sentences_with_citations += 1

            sentence_citation_coverage = sentences_with_citations / max(
                len(sentences), 1
            )
            citation_threshold = (
                resources.settings.search_config.citation_coverage_threshold
                if resources and resources.settings
                else 0.6
            )
            citation_coverage_ok = sentence_citation_coverage >= citation_threshold

            # 2. Basic citation presence
            has_citations = len(citations) > 0
            result_citation_coverage = (
                len(citations) / max(len(results), 1) if results else 0
            )

            # 3. Anchor presence check for how-to queries
            anchor_check_passed = True
            anchor_details = {"checked": False, "found": False, "relevant_anchors": []}

            if (
                hasattr(self, "_current_plan")
                and getattr(self._current_plan, "expected_answer_shape", "")
                == "how_to_steps"
            ):
                anchor_check_passed, anchor_details = self._check_anchor_presence(
                    citations, results
                )

            # 4. Enhanced grounding check with full body text
            answer_words = set(answer.lower().split())
            result_words = set()

            # Use full body text from results for better grounding
            for result in results[:5]:  # Check top 5 results
                text = self._get_full_body_text(result)
                result_words.update(
                    text.lower().split()[:100]
                )  # More words for better coverage

            word_overlap = len(answer_words & result_words) / max(len(answer_words), 1)

            # 5. Utility-specific content check
            utility_content_score = 1.0
            if hasattr(self, "_current_plan") and getattr(
                self._current_plan, "recognized_utility", None
            ):
                utility_content_score = self._check_utility_content_relevance(
                    answer, self._current_plan.recognized_utility
                )

            # Calculate composite verification score
            verification_score = (
                0.25 * (1.0 if has_citations else 0.0)  # Basic citations
                + 0.30 * sentence_citation_coverage  # Citation density
                + 0.20 * (1.0 if anchor_check_passed else 0.5)  # Anchor relevance
                + 0.15 * min(word_overlap * 2, 1.0)  # Grounding
                + 0.10 * utility_content_score  # Utility relevance
            )

            # Determine if improvement needed
            needs_improvement = (
                verification_score < 0.65
                or not citation_coverage_ok
                or not anchor_check_passed
            )

            # Generate specific suggestions
            suggestions = []
            if not has_citations:
                suggestions.append("Add citations to support claims")
            elif sentence_citation_coverage < citation_threshold:
                suggestions.append(
                    f"Improve citation coverage: only {sentence_citation_coverage:.1%} of sentences have citations (need ≥{citation_threshold:.1%})"
                )

            if not anchor_check_passed and anchor_details.get("checked"):
                suggestions.append(
                    "Include procedural anchors (setup/onboarding/configuration sections)"
                )

            if word_overlap < 0.3:
                suggestions.append("Improve answer grounding to retrieved content")

            if utility_content_score < 0.7:
                suggestions.append(
                    "Ensure answer focuses on the specific utility mentioned"
                )

            return {
                "verification_score": verification_score,
                "has_citations": has_citations,
                "citation_coverage": result_citation_coverage,
                "sentence_citation_coverage": sentence_citation_coverage,
                "citation_coverage_ok": citation_coverage_ok,
                "anchor_check": anchor_details,
                "anchor_check_passed": anchor_check_passed,
                "word_overlap": word_overlap,
                "utility_content_score": utility_content_score,
                "needs_improvement": needs_improvement,
                "suggestions": suggestions,
                "sentences_analyzed": len(sentences),
                "sentences_with_citations": sentences_with_citations,
            }

        except Exception as e:
            logger.error(f"Enhanced answer verification failed: {e}")
            return {
                "verification_score": 0.5,
                "needs_improvement": False,
                "error": str(e),
            }

    def _extract_sentences_for_verification(self, text: str) -> List[str]:
        """Extract meaningful sentences for citation analysis."""
        import re

        # Split on sentence endings, filter out short fragments
        sentences = re.split(r"[.!?]+", text)
        meaningful_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            # Skip very short sentences, headers, and formatting
            if (
                len(sentence) > 20
                and not sentence.startswith("**")
                and not sentence.startswith("#")
                and not sentence.startswith("- ")
            ):
                meaningful_sentences.append(sentence)

        return meaningful_sentences

    def _check_anchor_presence(
        self, citations: List[str], results: List[Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if citations include procedural anchors for how-to queries."""
        procedural_terms = [
            "onboard",
            "setup",
            "config",
            "install",
            "enable",
            "start",
            "getting-started",
            "quickstart",
            "tutorial",
            "guide",
            "procedure",
            "steps",
            "how-to",
        ]

        relevant_anchors = []

        # Check citations for anchor fragments
        for citation in citations:
            if "#" in citation:
                anchor_part = citation.split("#", 1)[1].lower()
                for term in procedural_terms:
                    if term in anchor_part:
                        relevant_anchors.append(citation)
                        break

        # Also check result metadata for procedural content
        for result in results[:3]:  # Top 3 results
            result_path = getattr(result, "path", "")
            result_section = getattr(result, "section", "")
            combined_metadata = f"{result_path} {result_section}".lower()

            for term in procedural_terms:
                if term in combined_metadata:
                    doc_id = getattr(result, "doc_id", "unknown")
                    relevant_anchors.append(f"{doc_id}#{term}")

        anchor_check_passed = len(relevant_anchors) > 0

        return anchor_check_passed, {
            "checked": True,
            "found": anchor_check_passed,
            "relevant_anchors": relevant_anchors[:3],  # Limit for readability
            "procedural_terms_found": len(relevant_anchors),
        }

    def _get_full_body_text(self, result: Any) -> str:
        """Extract full body text from result, ensuring complete content."""
        # Try multiple text fields in order of preference
        text_fields = ["body", "full_text", "content", "text", "snippet"]

        for field in text_fields:
            text = getattr(result, field, None)
            if text and isinstance(text, str) and len(text) > 100:  # Prefer longer text
                return text

        # Fallback: concatenate available text fields
        all_text_parts = []
        for field in ["title", "summary", "text", "content"]:
            text = getattr(result, field, "")
            if text:
                all_text_parts.append(str(text))

        combined_text = " ".join(all_text_parts)

        # If still too short, this might indicate a content extraction issue
        if len(combined_text) < 50:
            logger.warning(
                f"Short content extracted for result {getattr(result, 'doc_id', 'unknown')}: {len(combined_text)} chars"
            )

        return combined_text

    def _check_utility_content_relevance(
        self, answer: str, recognized_utility: str
    ) -> float:
        """Check if answer content is relevant to the recognized utility."""
        answer_lower = answer.lower()
        utility_lower = recognized_utility.lower()

        # Direct utility name mentions
        utility_mentions = answer_lower.count(utility_lower)

        # Acronym mentions (extract first letters)
        acronym = "".join(word[0] for word in recognized_utility.split()).lower()
        acronym_mentions = answer_lower.count(acronym)

        # Utility-specific terminology
        utility_terms = {
            "customer interaction utility": [
                "customer",
                "interaction",
                "crm",
                "contact",
            ],
            "enhanced transaction utility": [
                "transaction",
                "payment",
                "processing",
                "financial",
            ],
            "account utility": ["account", "user", "profile", "management"],
            "customer summary utility": ["summary", "report", "overview", "dashboard"],
            "payment card utility": ["card", "payment", "billing", "checkout"],
        }

        relevant_terms = utility_terms.get(utility_lower, [])
        term_mentions = sum(answer_lower.count(term) for term in relevant_terms)

        # Calculate relevance score
        total_mentions = utility_mentions + acronym_mentions + term_mentions
        answer_length_words = len(answer.split())

        # Normalize by answer length, with bonuses for direct utility mentions
        base_score = min(
            total_mentions / max(answer_length_words / 20, 1), 1.0
        )  # 1 mention per 20 words = score 1.0
        direct_mention_bonus = 0.3 if utility_mentions > 0 else 0.0

        final_score = min(base_score + direct_mention_bonus, 1.0)

        return final_score

    async def _fallback_search(self, query: str, resources) -> List[Any]:
        """Fallback search when main search times out."""
        try:
            from src.agent.tools.search import search_index_tool
            from src.infra.search_config import OpenSearchConfig

            # Quick, simple search
            result = await search_index_tool(
                index=OpenSearchConfig.get_default_index(),
                query=query,
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                top_k=10,  # Smaller for speed
                strategy="bm25",  # Faster than enhanced_rrf
            )

            return result.results

        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []

    async def _improve_answer_with_verification(
        self,
        draft_answer: str,
        verification: Dict[str, Any],
        query: str,
        results: List[Any],
        resources,
    ) -> str:
        """Improve answer based on verification results."""
        # For now, just add citations if missing
        if not verification.get("has_citations") and results:
            # Add simple citations to the end
            citations_text = "\n\n**Sources:**\n"
            for i, result in enumerate(results[:3]):
                doc_id = getattr(result, "doc_id", f"doc_{i}")
                title = getattr(result, "title", "Untitled")
                citations_text += f"- [{doc_id}] {title}\n"

            return draft_answer + citations_text

        return draft_answer

    async def _fallback_orchestration(self, query: str, resources) -> Dict[str, Any]:
        """Fallback to simple search when orchestration fails."""
        try:
            from src.agent.tools.search import search_index_tool
            from src.agent.nodes.processing_nodes import AnswerNode
            from src.infra.search_config import OpenSearchConfig

            # Simple search
            result = await search_index_tool(
                index=OpenSearchConfig.get_default_index(),
                query=query,
                search_client=resources.search_client,
                embed_client=resources.embed_client,
                embed_model=resources.settings.embed.model,
                strategy="enhanced_rrf",
            )

            # Simple answer
            answer_node = AnswerNode()
            answer_result = await answer_node.execute(
                {"normalized_query": query, "combined_results": result.results}
            )

            return {
                "final_answer": answer_result.get(
                    "final_answer", "Unable to process request."
                ),
                "search_results": result.results,
                "plan_confidence": 0.5,
                "fallback_used": True,
            }

        except Exception as e:
            logger.error(f"Fallback orchestration failed: {e}")
            return {
                "final_answer": "I'm unable to process your request at this time.",
                "search_results": [],
                "plan_confidence": 0.0,
                "fallback_used": True,
            }

    def _format_results_as_context(self, results: List[Any]) -> str:
        """Format search results as context string."""
        if not results:
            return "No relevant information found."

        context_parts = []
        for i, result in enumerate(results[:6]):  # Limit for context size
            text = getattr(result, "text", str(result))
            doc_id = getattr(result, "doc_id", f"doc_{i}")

            if len(text) > 400:
                text = text[:400] + "..."

            context_parts.append(f"[{doc_id}]: {text}")

        return "\n\n".join(context_parts)

    def _extract_citations(self, text: str) -> List[str]:
        """Extract citations from text."""
        import re

        return re.findall(r"\[([^\]]+)\]", text)

    def _format_context(self, context: List[Dict]) -> str:
        """Format conversation context."""
        context_parts = []
        for turn in context[-2:]:  # Last 2 turns
            if "query" in turn:
                context_parts.append(f"Q: {turn['query']}")
            if "answer" in turn:
                answer_preview = (
                    turn["answer"][:100] + "..."
                    if len(turn["answer"]) > 100
                    else turn["answer"]
                )
                context_parts.append(f"A: {answer_preview}")
        return "\n".join(context_parts)
    
    def _prepare_search_strategy(self, query: str) -> tuple:
        """Prepare search strategy and detect recognized utility."""
        recognized_utility = self._extract_utility_from_query(query)
        search_policy = "utility_first" if recognized_utility else "general"
        all_results = []
        return recognized_utility, search_policy, all_results
    
    async def _execute_utility_anchored_search(
        self, query: str, recognized_utility: str, plan: EnhancedPlan, resources
    ) -> List[Any]:
        """Execute Pass 1: Utility-anchored high-precision search."""
        from src.agent.tools.search import search_index_tool
        from src.infra.search_config import OpenSearchConfig
        import asyncio
        
        logger.info(f"Pass 1: Utility-anchored search for '{recognized_utility}'")
        
        # Prepare utility filters
        utility_filters = {
            "utility_name": recognized_utility,
            "boost_utility": resources.settings.search_config.utility_boost_factor,
            "full_body_text": True,
        }
        
        # Merge with plan filters and choose index
        base_filters = plan.steps[0].get("filters") if plan.steps else {}
        utility_filters.update(base_filters)
        
        index = (
            OpenSearchConfig.get_swagger_index() 
            if base_filters.get("index") == "swagger"
            else OpenSearchConfig.get_default_index()
        )
        
        try:
            utility_result = await asyncio.wait_for(
                search_index_tool(
                    index=index,
                    query=f"{query} {recognized_utility}",
                    filters=utility_filters,
                    search_client=resources.search_client,
                    embed_client=resources.embed_client,
                    embed_model=resources.settings.embed.model,
                    top_k=resources.settings.search_config.search_top_k_per_index_info,
                    strategy="enhanced_rrf",
                ),
                timeout=self.max_verification_time_ms / 1000,
            )
            
            # Mark utility-anchored results
            for result in utility_result.results:
                result.utility_anchored = True
            
            logger.info(f"Pass 1 yielded {len(utility_result.results)} utility-anchored results")
            return utility_result.results
            
        except asyncio.TimeoutError:
            logger.warning("Pass 1 utility search timed out, proceeding to Pass 2")
            return []
        except Exception as e:
            logger.warning(f"Pass 1 utility search failed: {e}, proceeding to Pass 2")
            return []
    
    async def _execute_general_search(
        self, query: str, plan: EnhancedPlan, resources, recognized_utility: Optional[str]
    ) -> List[Any]:
        """Execute Pass 2: General recall search."""
        from src.agent.tools.search import search_index_tool, multi_index_search_tool
        from src.infra.search_config import OpenSearchConfig
        import asyncio
        
        logger.info("Pass 2: General hybrid search for recall")
        
        general_filters = plan.steps[0].get("filters") if plan.steps else {}
        general_filters["full_body_text"] = True
        
        # Choose between multi-index or single-index search
        if self._should_use_multi_index_search(general_filters, recognized_utility):
            return await self._execute_multi_index_search(query, general_filters, resources)
        else:
            return await self._execute_single_index_search(query, general_filters, resources)
    
    def _should_use_multi_index_search(self, general_filters: dict, recognized_utility: Optional[str]) -> bool:
        """Determine if multi-index search should be used."""
        return (
            general_filters.get("index") != "swagger" 
            and recognized_utility 
            and not general_filters.get("index")
        )
    
    async def _execute_multi_index_search(self, query: str, general_filters: dict, resources) -> List[Any]:
        """Execute multi-index search strategy."""
        from src.agent.tools.search import multi_index_search_tool
        from src.infra.search_config import OpenSearchConfig
        import asyncio
        
        indices = [
            OpenSearchConfig.get_default_index(),
            OpenSearchConfig.get_swagger_index(),
        ]
        
        try:
            multi_results = await asyncio.wait_for(
                multi_index_search_tool(
                    indices=indices,
                    query=query,
                    filters=general_filters,
                    search_client=resources.search_client,
                    embed_client=resources.embed_client,
                    embed_model=resources.settings.embed.model,
                    top_k_per_index=resources.settings.search_config.search_top_k_per_index_info // 2,
                ),
                timeout=self.max_verification_time_ms / 1000,
            )
            
            # Flatten and mark results
            all_results = []
            for result_set in multi_results:
                for result in result_set.results:
                    result.utility_anchored = False
                all_results.extend(result_set.results)
            
            logger.info(f"Pass 2 multi-index yielded {len(all_results)} general results")
            return all_results
            
        except asyncio.TimeoutError:
            logger.warning("Pass 2 multi-index search timed out")
            return []
        except Exception as e:
            logger.warning(f"Pass 2 multi-index search failed: {e}")
            return []
    
    async def _execute_single_index_search(self, query: str, general_filters: dict, resources) -> List[Any]:
        """Execute single-index search strategy."""
        from src.agent.tools.search import search_index_tool
        from src.infra.search_config import OpenSearchConfig
        import asyncio
        
        index = (
            general_filters.get("index_override")
            or OpenSearchConfig.get_swagger_index() if general_filters.get("index") == "swagger"
            else OpenSearchConfig.get_default_index()
        )
        
        try:
            general_result = await asyncio.wait_for(
                search_index_tool(
                    index=index,
                    query=query,
                    filters=general_filters,
                    search_client=resources.search_client,
                    embed_client=resources.embed_client,
                    embed_model=resources.settings.embed.model,
                    top_k=resources.settings.search_config.search_top_k_info,
                    strategy="enhanced_rrf",
                ),
                timeout=self.max_verification_time_ms / 1000,
            )
            
            # Mark general results
            for result in general_result.results:
                result.utility_anchored = False
            
            logger.info(f"Pass 2 single-index yielded {len(general_result.results)} general results")
            return general_result.results
            
        except asyncio.TimeoutError:
            logger.warning("Pass 2 general search timed out")
            return []
        except Exception as e:
            logger.error(f"Pass 2 general search failed: {e}")
            return []
    
    def _process_search_results(
        self, all_results: List[Any], recognized_utility: Optional[str], 
        search_policy: str, plan: EnhancedPlan, query: str
    ) -> List[Any]:
        """Process and combine search results with metadata."""
        # WEIGHTED FUSION: Utility-anchored results preferred
        final_results = self._weighted_fusion_utility_first(
            all_results, recognized_utility, plan
        )
        
        # Store metadata for downstream nodes
        plan.recognized_utility = recognized_utility
        plan.search_policy = search_policy
        plan.orchestrator_trace_id = f"orch_{hash(query) % 10000}"
        
        logger.info(
            f"Enhanced search complete: {len(final_results)} results, policy={search_policy}"
        )
        return final_results

    def _analyze_citations(self, citations: List[str], sentences: List[str], results: List[Any], resources) -> Dict[str, Any]:
        """Analyze citation coverage and presence."""
        # Citation coverage by sentences
        sentences_with_citations = sum(
            1 for sentence in sentences 
            if any(citation in sentence for citation in citations)
        )
        
        sentence_citation_coverage = sentences_with_citations / max(len(sentences), 1)
        citation_threshold = (
            resources.settings.search_config.citation_coverage_threshold
            if resources and resources.settings else 0.6
        )
        
        return {
            "has_citations": len(citations) > 0,
            "result_citation_coverage": len(citations) / max(len(results), 1) if results else 0,
            "sentence_citation_coverage": sentence_citation_coverage,
            "citation_threshold": citation_threshold,
            "citation_coverage_ok": sentence_citation_coverage >= citation_threshold,
            "sentences_with_citations": sentences_with_citations,
        }
    
    def _check_anchor_requirements(self, citations: List[str], results: List[Any]) -> Dict[str, Any]:
        """Check anchor presence requirements for how-to queries."""
        anchor_check_passed = True
        anchor_details = {"checked": False, "found": False, "relevant_anchors": []}
        
        if (
            hasattr(self, "_current_plan")
            and getattr(self._current_plan, "expected_answer_shape", "") == "how_to_steps"
        ):
            anchor_check_passed, anchor_details = self._check_anchor_presence(citations, results)
        
        return {
            "anchor_check_passed": anchor_check_passed,
            "anchor_details": anchor_details,
        }
    
    def _calculate_grounding_score(self, answer: str, results: List[Any]) -> float:
        """Calculate answer grounding score based on word overlap."""
        answer_words = set(answer.lower().split())
        result_words = set()
        
        # Use full body text from results for better grounding
        for result in results[:5]:  # Check top 5 results
            text = self._get_full_body_text(result)
            result_words.update(text.lower().split()[:100])  # More words for better coverage
        
        return len(answer_words & result_words) / max(len(answer_words), 1)
    
    def _calculate_utility_relevance(self, answer: str) -> float:
        """Calculate utility-specific content relevance score."""
        if not (hasattr(self, "_current_plan") and getattr(self._current_plan, "recognized_utility", None)):
            return 1.0
        
        return self._check_utility_content_relevance(
            answer, self._current_plan.recognized_utility
        )
    
    def _calculate_verification_score(
        self, citation_analysis: Dict, anchor_analysis: Dict, grounding_score: float, utility_score: float
    ) -> float:
        """Calculate composite verification score."""
        return (
            0.25 * (1.0 if citation_analysis["has_citations"] else 0.0)  # Basic citations
            + 0.30 * citation_analysis["sentence_citation_coverage"]  # Citation density
            + 0.20 * (1.0 if anchor_analysis["anchor_check_passed"] else 0.5)  # Anchor relevance
            + 0.15 * min(grounding_score * 2, 1.0)  # Grounding
            + 0.10 * utility_score  # Utility relevance
        )
    
    def _requires_improvement(
        self, verification_score: float, citation_analysis: Dict, anchor_analysis: Dict
    ) -> bool:
        """Determine if answer quality improvement is needed."""
        return (
            verification_score < 0.65
            or not citation_analysis["citation_coverage_ok"]
            or not anchor_analysis["anchor_check_passed"]
        )
    
    def _generate_improvement_suggestions(
        self, citation_analysis: Dict, anchor_analysis: Dict, grounding_score: float, utility_score: float
    ) -> List[str]:
        """Generate specific improvement suggestions."""
        suggestions = []
        
        # Citation suggestions
        if not citation_analysis["has_citations"]:
            suggestions.append("Add citations to support claims")
        elif citation_analysis["sentence_citation_coverage"] < citation_analysis["citation_threshold"]:
            coverage = citation_analysis["sentence_citation_coverage"]
            threshold = citation_analysis["citation_threshold"]
            suggestions.append(
                f"Improve citation coverage: only {coverage:.1%} of sentences have citations (need ≥{threshold:.1%})"
            )
        
        # Anchor suggestions
        if not anchor_analysis["anchor_check_passed"] and anchor_analysis["anchor_details"].get("checked"):
            suggestions.append("Include procedural anchors (setup/onboarding/configuration sections)")
        
        # Grounding suggestions
        if grounding_score < 0.3:
            suggestions.append("Improve answer grounding to retrieved content")
        
        # Utility relevance suggestions
        if utility_score < 0.7:
            suggestions.append("Ensure answer focuses on the specific utility mentioned")
        
        return suggestions
    
    def _build_verification_result(
        self, verification_score: float, citation_analysis: Dict, anchor_analysis: Dict,
        grounding_score: float, utility_score: float, needs_improvement: bool, 
        suggestions: List[str], sentences: List[str]
    ) -> Dict[str, Any]:
        """Build final verification result dictionary."""
        return {
            "verification_score": verification_score,
            "has_citations": citation_analysis["has_citations"],
            "citation_coverage": citation_analysis["result_citation_coverage"],
            "sentence_citation_coverage": citation_analysis["sentence_citation_coverage"],
            "citation_coverage_ok": citation_analysis["citation_coverage_ok"],
            "anchor_check": anchor_analysis["anchor_details"],
            "anchor_check_passed": anchor_analysis["anchor_check_passed"],
            "word_overlap": grounding_score,
            "utility_content_score": utility_score,
            "needs_improvement": needs_improvement,
            "suggestions": suggestions,
            "sentences_analyzed": len(sentences),
            "sentences_with_citations": citation_analysis["sentences_with_citations"],
        }


