"""
Streamlined LLM orchestrator for conversational RAG - Version 2.

This is a leaner implementation that removes unnecessary abstraction
while maintaining the core value of multi-step planning and execution.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Plan:
    """Simple execution plan."""
    steps: List[Dict[str, Any]]
    confidence: float = 0.8


class StreamlinedOrchestrator:
    """Minimal orchestrator focused on actual functionality."""
    
    def __init__(self, chat_client=None):
        self.chat_client = chat_client
        self.use_llm = chat_client is not None
    
    async def orchestrate(
        self, 
        query: str,
        resources: Any,
        context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Main orchestration: plan → execute → respond.
        
        Removes unnecessary verification step and complex tool abstractions.
        """
        # 1. Plan (LLM or regex)
        if self.use_llm:
            plan = self._llm_plan(query, context)
        else:
            plan = self._simple_plan(query)
        
        # 2. Execute search
        from src.agent.tools.search import search_tool
        
        search_results = await search_tool(
            query=query,
            resources=resources,
            filters=plan.steps[0].get("filters")
        )
        
        # 3. Generate answer (reuse existing node)
        from src.agent.nodes.processing_nodes import AnswerNode
        
        answer_node = AnswerNode()
        answer_result = await answer_node.execute({
            "combined_results": search_results.results,
            "normalized_query": query,
        })
        
        return {
            "search_results": search_results.results,
            "final_answer": answer_result.get("final_answer"),
            "plan_confidence": plan.confidence
        }
    
    def _simple_plan(self, query: str) -> Plan:
        """Regex-based planning using existing slotter."""
        from src.agent.intent.slotter import slot
        
        slot_result = slot(query)
        
        # Simple mapping to search filters
        filters = None
        if slot_result.doish:
            filters = {"content_type": "procedure"}
        elif "api" in query.lower():
            filters = {"index": "swagger"}
            
        return Plan(
            steps=[{"action": "search", "filters": filters}],
            confidence=slot_result.confidence
        )
    
    def _llm_plan(self, query: str, context: Optional[List[Dict]] = None) -> Plan:
        """LLM-based planning with structured output."""
        messages = [
            {"role": "system", "content": "Plan search strategy. Output JSON: {\"filters\": {...}, \"confidence\": 0.0-1.0}"},
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.chat_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            plan_data = json.loads(response.choices[0].message.content)
            
            return Plan(
                steps=[{"action": "search", "filters": plan_data.get("filters")}],
                confidence=plan_data.get("confidence", 0.8)
            )
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using simple plan")
            return self._simple_plan(query)


# Single entry point - no extra wrappers
async def orchestrated_search_v2(query: str, resources: Any, use_orchestrator: bool = True) -> Dict[str, Any]:
    """Direct orchestration without unnecessary layers."""
    if not use_orchestrator:
        from src.agent.tools.search import search_tool
        return await search_tool(query=query, resources=resources)
    
    orchestrator = StreamlinedOrchestrator(
        chat_client=resources.chat_client if hasattr(resources, 'chat_client') else None
    )
    return await orchestrator.orchestrate(query, resources)