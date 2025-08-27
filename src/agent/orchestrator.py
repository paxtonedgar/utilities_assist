"""
Minimal LLM orchestrator for conversational RAG.

This module implements a lightweight orchestrator that sits between the user and
the existing RAG pipeline, enabling multi-turn conversations with tool use.

Architecture:
- Planner: Decides what tools to use based on the query
- Tool Executor: Runs the selected tools (search, filters, etc.)
- Verifier: Checks if results are sufficient
- Answer Generator: Produces the final response

This integrates cleanly with the existing pipeline without disrupting it.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """Available tools for the orchestrator."""
    SEARCH = "search"
    FILTER = "filter"
    EXPAND = "expand"
    CLARIFY = "clarify"
    ANSWER = "answer"


@dataclass
class ToolCall:
    """Represents a single tool invocation."""
    tool: ToolType
    params: Dict[str, Any]
    reasoning: str


@dataclass
class OrchestratorPlan:
    """Execution plan from the orchestrator."""
    tool_calls: List[ToolCall]
    strategy: str
    confidence: float


class LLMOrchestrator:
    """
    Minimal orchestrator that wraps existing RAG functionality.
    
    Key principles:
    - KISS: Keep it simple, leverage existing code
    - DRY: Don't duplicate existing functionality
    - Clean integration: Works alongside existing pipeline
    """
    
    def __init__(self, llm_client=None):
        """
        Initialize orchestrator with optional LLM client.
        
        Args:
            llm_client: Optional LLM client for planning/verification
                       If None, falls back to regex-based planning
        """
        self.llm_client = llm_client
        self.use_llm_planning = llm_client is not None
        
    async def plan(
        self, 
        query: str, 
        context: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OrchestratorPlan:
        """
        Create an execution plan for the query.
        
        Args:
            query: User's question
            context: Previous conversation turns
            metadata: Additional context (user preferences, etc.)
            
        Returns:
            OrchestratorPlan with tool calls and strategy
        """
        if self.use_llm_planning:
            return await self._llm_plan(query, context, metadata)
        else:
            return self._regex_plan(query, context, metadata)
    
    def _regex_plan(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OrchestratorPlan:
        """
        Fallback regex-based planning when LLM is not available.
        
        This leverages the existing slotter logic for consistency.
        """
        from src.agent.intent.slotter import slot
        
        # Use existing slotter for intent detection
        slot_result = slot(query)
        
        tool_calls = []
        
        # Map slotter intent to tool calls
        if slot_result.intent == "procedure" and slot_result.doish:
            # Procedural query - search for how-to content
            tool_calls.append(ToolCall(
                tool=ToolType.SEARCH,
                params={"query": query, "filters": {"content_type": "procedure"}},
                reasoning=f"User asking how to do something: {slot_result.reasons}"
            ))
        elif "api" in query.lower() or "endpoint" in query.lower():
            # API query - search swagger docs
            tool_calls.append(ToolCall(
                tool=ToolType.SEARCH,
                params={"query": query, "index": "swagger"},
                reasoning="User asking about API/endpoints"
            ))
        else:
            # Default info search
            tool_calls.append(ToolCall(
                tool=ToolType.SEARCH,
                params={"query": query, "filters": None},
                reasoning="General information query"
            ))
        
        # Add answer generation step
        tool_calls.append(ToolCall(
            tool=ToolType.ANSWER,
            params={"format": "conversational"},
            reasoning="Generate final answer from search results"
        ))
        
        return OrchestratorPlan(
            tool_calls=tool_calls,
            strategy="sequential",
            confidence=slot_result.confidence
        )
    
    async def _llm_plan(
        self,
        query: str,
        context: Optional[List[Dict[str, str]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> OrchestratorPlan:
        """
        LLM-based planning for more sophisticated routing.
        
        Uses structured output to ensure reliable tool selection.
        """
        # Build planning prompt
        system_prompt = """You are a RAG orchestrator that plans tool execution.
        
Available tools:
- SEARCH: Query the knowledge base
- FILTER: Apply filters to narrow results
- EXPAND: Expand acronyms or clarify terms
- CLARIFY: Ask user for clarification
- ANSWER: Generate final response

Output a JSON plan with:
{
  "tool_calls": [
    {"tool": "SEARCH", "params": {...}, "reasoning": "..."},
    ...
  ],
  "strategy": "sequential|parallel",
  "confidence": 0.0-1.0
}
"""
        
        # Include context if available
        messages = [{"role": "system", "content": system_prompt}]
        
        if context:
            # Add recent conversation history
            for turn in context[-3:]:  # Last 3 turns
                messages.append({"role": turn["role"], "content": turn["content"]})
        
        messages.append({"role": "user", "content": query})
        
        try:
            # Call LLM with structured output
            response = await self.llm_client.create_completion(
                messages=messages,
                temperature=0.1,  # Low temp for consistent planning
                response_format={"type": "json"}
            )
            
            # Parse JSON response
            plan_json = json.loads(response.content)
            
            # Convert to OrchestratorPlan
            tool_calls = [
                ToolCall(
                    tool=ToolType[tc["tool"]],
                    params=tc["params"],
                    reasoning=tc["reasoning"]
                )
                for tc in plan_json["tool_calls"]
            ]
            
            return OrchestratorPlan(
                tool_calls=tool_calls,
                strategy=plan_json.get("strategy", "sequential"),
                confidence=plan_json.get("confidence", 0.8)
            )
            
        except Exception as e:
            logger.warning(f"LLM planning failed, falling back to regex: {e}")
            return self._regex_plan(query, context, metadata)
    
    async def execute_plan(
        self,
        plan: OrchestratorPlan,
        resources: Any
    ) -> Dict[str, Any]:
        """
        Execute the orchestrator's plan using existing tools.
        
        Args:
            plan: The execution plan
            resources: RAG resources (search client, etc.)
            
        Returns:
            Aggregated results from tool execution
        """
        results = {
            "tool_outputs": [],
            "search_results": [],
            "final_answer": None,
            "metadata": {}
        }
        
        for tool_call in plan.tool_calls:
            try:
                if tool_call.tool == ToolType.SEARCH:
                    # Use existing search infrastructure
                    from src.agent.tools.search import search_tool
                    
                    search_result = await search_tool(
                        query=tool_call.params.get("query"),
                        filters=tool_call.params.get("filters"),
                        resources=resources
                    )
                    
                    results["search_results"].extend(search_result.results)
                    results["tool_outputs"].append({
                        "tool": "search",
                        "status": "success",
                        "result_count": len(search_result.results)
                    })
                    
                elif tool_call.tool == ToolType.ANSWER:
                    # Generate answer using existing infrastructure
                    if results["search_results"]:
                        # Use existing answer generation
                        from src.agent.nodes.processing_nodes import AnswerNode
                        
                        answer_node = AnswerNode()
                        answer_result = await answer_node.execute({
                            "combined_results": results["search_results"],
                            "normalized_query": tool_call.params.get("query", ""),
                        })
                        
                        results["final_answer"] = answer_result.get("final_answer")
                    else:
                        results["final_answer"] = "I couldn't find relevant information for your query."
                        
            except Exception as e:
                logger.error(f"Tool execution failed for {tool_call.tool}: {e}")
                results["tool_outputs"].append({
                    "tool": tool_call.tool.value,
                    "status": "error",
                    "error": str(e)
                })
        
        return results
    
    async def verify_results(
        self,
        results: Dict[str, Any],
        query: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify if results are sufficient to answer the query.
        
        Args:
            results: Execution results
            query: Original user query
            
        Returns:
            Tuple of (is_sufficient, retry_suggestion)
        """
        # Simple heuristic verification
        if not results.get("search_results"):
            return False, "No results found. Try rephrasing the query."
        
        if len(results["search_results"]) < 2:
            return False, "Limited results. Consider broadening the search."
        
        # If we have LLM, do smarter verification
        if self.use_llm_planning and results.get("final_answer"):
            try:
                verification_prompt = f"""
                Query: {query}
                Answer: {results['final_answer'][:500]}
                
                Does this answer adequately address the query? 
                Reply with JSON: {{"sufficient": true/false, "reason": "..."}}
                """
                
                response = await self.llm_client.create_completion(
                    messages=[{"role": "user", "content": verification_prompt}],
                    temperature=0.1,
                    response_format={"type": "json"}
                )
                
                verification = json.loads(response.content)
                return verification["sufficient"], verification.get("reason")
                
            except Exception as e:
                logger.warning(f"Verification failed: {e}")
        
        # Default: accept if we have results and an answer
        return bool(results.get("final_answer")), None


# Integration point with existing pipeline
async def orchestrated_search(
    query: str,
    resources: Any,
    context: Optional[List[Dict[str, str]]] = None,
    use_orchestrator: bool = True
) -> Dict[str, Any]:
    """
    Entry point that can be called from existing pipeline.
    
    This wraps the existing search with orchestration when enabled.
    """
    if not use_orchestrator:
        # Fall back to traditional search
        from src.agent.tools.search import search_tool
        return await search_tool(query=query, resources=resources)
    
    # Use orchestrator
    orchestrator = LLMOrchestrator(llm_client=resources.llm_client if hasattr(resources, 'llm_client') else None)
    
    # Plan
    plan = await orchestrator.plan(query, context)
    logger.info(f"Orchestrator plan: {plan.strategy} with {len(plan.tool_calls)} tools")
    
    # Execute
    results = await orchestrator.execute_plan(plan, resources)
    
    # Verify
    is_sufficient, retry_msg = await orchestrator.verify_results(results, query)
    
    if not is_sufficient and retry_msg:
        logger.info(f"Results insufficient: {retry_msg}")
        # Could implement retry logic here
    
    return results