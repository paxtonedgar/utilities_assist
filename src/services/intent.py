"""Intent classification service."""

import logging
import re
from typing import List, Any
from services.models import IntentResult

logger = logging.getLogger(__name__)


async def determine_intent(text: str, chat_client: Any, utilities_list: List[str] = None, model_name: str = "gpt-4o-mini") -> IntentResult:
    """Determine user intent from query text.
    
    Args:
        text: Normalized user query
        chat_client: LLM client for intent classification
        utilities_list: List of known utilities/APIs
        
    Returns:
        IntentResult with classified intent and confidence
    """
    if not text or not text.strip():
        return IntentResult(intent="unknown", confidence=0.0, reasoning="Empty query")
    
    text = text.strip().lower()
    
    # Check for restart intent first
    if text in ["start over", "restart", "reset", "clear"]:
        return IntentResult(
            intent="restart", 
            confidence=1.0, 
            reasoning="Direct restart command"
        )
    
    # Rule-based intent detection for common patterns
    if _is_list_intent(text):
        return IntentResult(
            intent="list",
            confidence=0.9,
            reasoning="Query asks for enumeration/listing"
        )
    
    if _is_swagger_intent(text):
        return IntentResult(
            intent="swagger", 
            confidence=0.8,
            reasoning="Query about API specifications"
        )
    
    # Use LLM for ambiguous cases
    try:
        return await _classify_with_llm(text, chat_client, utilities_list or [], model_name)
    except Exception as e:
        logger.error(f"LLM intent classification failed: {e}")
        # Fallback to confluence for general queries
        return IntentResult(
            intent="confluence",
            confidence=0.5,
            reasoning=f"Fallback after LLM error: {e}"
        )


def _is_list_intent(text: str) -> bool:
    """Check if query is asking for a list of items."""
    list_patterns = [
        r"\blist\b",
        r"\bshow\s+(all|me)\b",
        r"\bwhat\s+(are|all)\s+the\b",
        r"\bhow\s+many\b",
        r"\bwhich\s+(all|ones)\b",
        r"\bgive\s+me\s+(all|the)\b",
        r"\benumerate\b",
        r"\bapis?\s+are\s+there\b",
        r"\butil\w*\s+are\s+(available|there)\b"
    ]
    
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in list_patterns)


def _is_swagger_intent(text: str) -> bool:
    """Check if query is about API specifications.""" 
    swagger_patterns = [
        r"\bendpoint\b",
        r"\brequest\b",
        r"\bresponse\b",
        r"\bparameter\b",
        r"\bfield\b",
        r"\bapi\s+spec\b",
        r"\bswagger\b",
        r"\bopenapi\b",
        r"\bschema\b",
        r"\bjson\b.*\bstructure\b"
    ]
    
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in swagger_patterns)


async def _classify_with_llm(text: str, chat_client: Any, utilities_list: List[str], model_name: str = "gpt-4o-mini") -> IntentResult:
    """Use LLM to classify complex intents."""
    
    utilities_str = ", ".join(utilities_list[:10])  # Limit for prompt size
    
    system_prompt = f"""
You are an intelligent assistant that classifies user queries about enterprise utilities and APIs.

Available utilities: {utilities_str}

Classify the query into ONE of these categories:
- 'confluence': General documentation, explanations, concepts about utilities
- 'swagger': API specifications, endpoints, request/response details, parameters
- 'list': Enumeration of APIs, utilities, or items
- 'info': Detailed information about a specific item

Query: "{text}"

Respond with ONLY the classification in this exact format:
Intent: [intent]
Confidence: [0.0-1.0]
Reasoning: [brief explanation]
"""

    try:
        # Create messages for chat completion
        messages = [{"role": "user", "content": system_prompt}]
        
        # Get LLM response (OpenAI client is sync, not async)
        response = chat_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=100,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        return _parse_intent_response(content)
        
    except Exception as e:
        logger.error(f"LLM intent classification error: {e}")
        raise


def _parse_intent_response(response: str) -> IntentResult:
    """Parse LLM response into IntentResult."""
    intent = "confluence"  # Default fallback
    confidence = 0.5
    reasoning = "Parsed from LLM response"
    
    try:
        # Extract intent
        if "Intent:" in response:
            intent_match = re.search(r"Intent:\s*(\w+)", response)
            if intent_match:
                intent = intent_match.group(1).lower()
        
        # Extract confidence
        if "Confidence:" in response:
            conf_match = re.search(r"Confidence:\s*([\d.]+)", response)
            if conf_match:
                confidence = float(conf_match.group(1))
        
        # Extract reasoning
        if "Reasoning:" in response:
            reason_match = re.search(r"Reasoning:\s*(.+)$", response, re.MULTILINE)
            if reason_match:
                reasoning = reason_match.group(1).strip()
                
    except Exception as e:
        logger.warning(f"Failed to parse LLM response: {e}")
    
    return IntentResult(intent=intent, confidence=confidence, reasoning=reasoning)