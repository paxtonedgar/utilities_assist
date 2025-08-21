"""Intent node - intent classification using jinja template."""

import logging
from pathlib import Path
from typing import Dict, Optional, List
from jinja2 import Environment, FileSystemLoader
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.services.models import IntentResult
from src.services.intent import determine_intent  # Keep as fallback
from src.telemetry.logger import stage
from .base_node import BaseNodeHandler, to_state_dict, from_state_dict

# Import constants to prevent KeyError issues
from src.agent.constants import NORMALIZED_QUERY, INTENT

logger = logging.getLogger(__name__)

# Load jinja templates
template_dir = Path(__file__).parent.parent / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

# Constants for heuristic classification
CONFIDENCE_SCORES = {
    'swagger_short': 0.85,
    'confluence_short': 0.75,
    'swagger': 0.85,
    'list': 0.80,
    'workflow': 0.75,
    'utility_acronym': 0.90
}

KEYWORD_PATTERNS = {
    'swagger': ['api', 'endpoint', 'swagger', 'openapi', 'rest', 'json', 'http', 'method'],
    'list': ['list', 'show all', 'what are', 'available', 'services', 'utilities'],
    'workflow': ['workflow', 'process', 'step by step', 'how to', 'procedure']
}


class HeuristicIntentClassifier:
    """Fast heuristic intent classifier following SRP."""
    
    def __init__(self):
        self._utilities_acronyms = self._load_utilities_acronyms()
    
    def _load_utilities_acronyms(self) -> Dict[str, str]:
        """Load utility acronyms safely."""
        try:
            from agent.acronym_map import _load_acronym_data
            return _load_acronym_data()
        except ImportError:
            logger.warning("Acronym map not available for heuristic classification")
            return {}
    
    def classify(self, query: str) -> Optional[Dict[str, str]]:
        """Main classification entry point."""
        if not self._is_valid_query(query):
            return None
        
        query_lower = query.lower().strip()
        tokens = query_lower.split()
        
        # Try short query classification first
        result = self._classify_short_query(query_lower, tokens)
        if result:
            return result
            
        # Try keyword-based classification
        result = self._classify_by_keywords(query_lower)
        if result:
            return result
            
        # Try utility acronym detection
        result = self._classify_by_acronym(tokens)
        if result:
            return result
        
        return None
    
    def _is_valid_query(self, query: str) -> bool:
        """Check if query is valid for heuristic classification."""
        return bool(query and len(query.strip()) >= 3)
    
    def _classify_short_query(self, query_lower: str, tokens: List[str]) -> Optional[Dict[str, str]]:
        """Classify short queries (1-2 words)."""
        if len(tokens) > 2:
            return None
        
        # Check for API-related patterns
        if any(word in query_lower for word in KEYWORD_PATTERNS['swagger'][:5]):  # First 5 are most indicative
            return {
                'intent': 'swagger',
                'confidence': CONFIDENCE_SCORES['swagger_short'],
                'reasoning': 'Short API-related query'
            }
        
        # Default to confluence for short queries
        return {
            'intent': 'confluence',
            'confidence': CONFIDENCE_SCORES['confluence_short'],
            'reasoning': 'Short general query'
        }
    
    def _classify_by_keywords(self, query_lower: str) -> Optional[Dict[str, str]]:
        """Classify based on keyword patterns."""
        for intent, keywords in KEYWORD_PATTERNS.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches >= 1:
                # Higher confidence for more matches
                base_confidence = CONFIDENCE_SCORES[intent]
                confidence = min(base_confidence + (matches - 1) * 0.05, 0.95)
                return {
                    'intent': intent,
                    'confidence': confidence,
                    'reasoning': f'Keyword match: {matches} patterns'
                }
        return None
    
    def _classify_by_acronym(self, tokens: List[str]) -> Optional[Dict[str, str]]:
        """Classify based on utility acronyms."""
        if not self._utilities_acronyms:
            return None
            
        for token in tokens:
            if token.upper() in self._utilities_acronyms:
                return {
                    'intent': 'confluence',
                    'confidence': CONFIDENCE_SCORES['utility_acronym'],
                    'reasoning': f'Utility acronym detected: {token.upper()}'
                }
        return None


# Global instance for performance (avoid recreating)
_heuristic_classifier = HeuristicIntentClassifier()


def _heuristic_intent_classifier(query: str) -> Optional[Dict[str, str]]:
    """Fast heuristic intent classifier for simple queries to avoid LLM calls.
    
    Returns classification dict or None if LLM escalation needed.
    Saves ~4s for 80% of simple queries.
    """
    return _heuristic_classifier.classify(query)


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
        
        # Get query for analysis
        normalized_query = s.get(NORMALIZED_QUERY, "")
        
        # PERFORMANCE: Use heuristic classifier for short/simple queries
        heuristic_result = _heuristic_intent_classifier(normalized_query)
        if heuristic_result:
            logger.info(f"Heuristic intent classification: {heuristic_result['intent']} (confidence: {heuristic_result['confidence']:.2f}, saved ~4s LLM call)")
            intent_result = IntentResult(
                intent=heuristic_result['intent'],
                confidence=heuristic_result['confidence'],
                reasoning=heuristic_result['reasoning']
            )
            return from_state_dict({**s, INTENT: intent_result}, incoming_type)
        
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
            from src.infra.resource_manager import get_resources
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