"""Intent node - intent classification using jinja template."""

import logging
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from services.models import IntentResult
from services.intent import determine_intent  # Keep as fallback
from src.telemetry.logger import stage
from .base_node import BaseNodeHandler, to_state_dict, from_state_dict

# Import constants to prevent KeyError issues
from agent.constants import NORMALIZED_QUERY, INTENT

logger = logging.getLogger(__name__)

# Load jinja templates
template_dir = Path(__file__).parent.parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))


class IntentAnalysis(BaseModel):
    """Structured output for intent classification."""
    intent: str = Field(description="Primary intent category")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief explanation of the classification")


@stage("classify_intent")
async def intent_node(state, config, *, store=None):
    """
    Classify query intent using LLM with jinja template.
    
    Follows LangGraph pattern with config parameter and optional store injection.
    
    Args:
        state: Workflow state (GraphState or dict) containing normalized_query
        config: RunnableConfig with user context and configuration
        store: Optional BaseStore for cross-thread user memory
        
    Returns:
        State update with intent classification (same type as input)
    """
    incoming_type = type(state)
    
    try:
        # Convert to dict for processing
        s = to_state_dict(state)
        
        logger.info(
            "NODE_START intent | keys=%s | normalized=%r",
            list(s.keys()),
            s.get(NORMALIZED_QUERY)
        )
        
        # Use consistent state keys with fallback
        normalized_query = s.get(NORMALIZED_QUERY, "").strip()
        if not normalized_query:
            logger.warning("intent_node: normalized_query missing/empty; setting intent=None and routing to search fallback.")
            # CRITICAL: Preserve ALL existing state fields and return proper type
            merged = {
                **s,  # Preserve all existing state
                INTENT: {"intent": "confluence", "confidence": 0.1},  # Store as dict consistently
                "workflow_path": s.get("workflow_path", []) + ["intent_empty"]
            }
            return from_state_dict(incoming_type, merged)
        
        # Utilities list for context
        utilities_list = [
            "Customer Summary Utility",
            "Enhanced Transaction Utility", 
            "Account Utility",
            "Customer Interaction Utility",
            "Digital Events",
            "Product Catalog Utility",
            "Global Customer Platform"
        ]
        
        # Try LLM-based intent classification
        try:
            template = jinja_env.get_template("intent.jinja")
            prompt = template.render(
                normalized_query=normalized_query,
                utilities_list=utilities_list
            )
            
            # Extract resources from config
            from infra.resource_manager import get_resources
            resources = get_resources()
            
            # Use the working Azure OpenAI client configuration for LangChain
            from langchain_openai import AzureChatOpenAI
            
            # Get authentication parameters from working clients.py approach
            from utils import load_config
            import os
            
            # Load auth config for API key (matching clients.py pattern) - renamed to avoid collision
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
            
            # Use structured output for better parsing
            intent_analyzer = langchain_client.with_structured_output(IntentAnalysis)
            
            analysis = await intent_analyzer.ainvoke([
                SystemMessage(content="You are an expert intent classification system for enterprise utilities and APIs."),
                HumanMessage(content=prompt)
            ])
            
            intent_result = IntentResult(
                intent=analysis.intent,
                confidence=analysis.confidence
            )
            
            logger.info(f"Intent classified: {analysis.intent} (confidence: {analysis.confidence:.2f}) - {analysis.reasoning}")
            
        except Exception as e:
            logger.warning(f"LLM intent classification failed, using simple fallback: {e}")
            # Simple fallback without complex dependencies - store as dict
            intent_result = {
                "intent": "confluence",
                "confidence": 0.5
            }
        
        # Normalize intent_result to dict format
        if hasattr(intent_result, 'intent'):
            # Convert IntentResult object to dict
            intent_dict = {
                "intent": intent_result.intent,
                "confidence": intent_result.confidence
            }
        else:
            # Already a dict
            intent_dict = intent_result
        
        logger.info(
            "NODE_END intent | intent=%s | confidence=%.2f",
            intent_dict["intent"],
            intent_dict["confidence"]
        )
        
        logger.debug(f"Storing intent as dict: {intent_dict}")
        
        # CRITICAL: Preserve ALL existing state fields and return proper type
        merged = {
            **s,  # Preserve all existing state
            INTENT: intent_dict,  # Store as dict, not IntentResult object
            "workflow_path": s.get("workflow_path", []) + ["intent"]
        }
        return from_state_dict(incoming_type, merged)
        
    except Exception as e:
        logger.error(f"Intent node failed: {e}")
        # Convert to dict if not already done (in case exception happened early)
        s = to_state_dict(state) if 's' not in locals() else s
        
        # Fallback intent and return proper type - store as dict consistently
        merged = {
            **s,  # Preserve all existing state
            INTENT: {"intent": "error", "confidence": 0.0},  # Store as dict, not IntentResult object
            "workflow_path": s.get("workflow_path", []) + ["intent_error"],
            "error_messages": s.get("error_messages", []) + [f"Intent classification failed: {e}"]
        }
        return from_state_dict(incoming_type, merged)


class IntentNode(BaseNodeHandler):
    """Class-based wrapper for intent functionality."""
    
    def __init__(self):
        super().__init__("intent")
    
    async def execute(self, state: dict, config: dict = None) -> dict:
        """Execute the intent logic using the existing function."""
        if config is None:
            config = {"configurable": {"thread_id": "unknown"}}
        return await intent_node(state, config)