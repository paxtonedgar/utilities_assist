"""Response generation and context building services."""

import logging
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
from src.services.models import SearchResult, SourceChip, IntentResult

logger = logging.getLogger(__name__)


# === TEXT PROCESSING UTILITIES ===

def _extract_meaningful_words(text: str, exclude_stopwords: bool = True) -> set:
    """Extract meaningful words from text, optionally excluding stopwords.
    
    Centralizes word extraction logic used throughout the file.
    """
    words = set(re.findall(r'\w+', text.lower()))
    
    if exclude_stopwords:
        # Combined stopwords from both functions + query-specific terms
        stopwords = {
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "what", "how", "when", "where", "why", "which", "who", "is", "are"
        }
        words = words - stopwords
    
    return words


def _calculate_word_overlap(text1: str, text2: str, threshold: float = 0.1) -> bool:
    """Calculate if word overlap between two texts meets threshold.
    
    Consolidates overlap calculation logic used in multiple validation functions.
    """
    if not text1 or not text2:
        return True  # Can't verify without substantial text
    
    words1 = _extract_meaningful_words(text1)
    words2 = _extract_meaningful_words(text2)
    
    if not words1:
        return True
    
    overlap = len(words1 & words2)
    return overlap / len(words1) > threshold


def _text_matches_patterns(text: str, patterns: List[str]) -> bool:
    """Check if text matches any of the provided patterns.
    
    Centralizes pattern matching logic used in validation functions.
    """
    text_lower = text.lower()
    return any(pattern in text_lower for pattern in patterns)


def build_context(
    retrieval_results: List[SearchResult], 
    intent: IntentResult,
    max_context_length: int = 50000
) -> str:
    """Build context string from retrieval results.
    
    Args:
        retrieval_results: Search results to include in context
        intent: Classified intent for context optimization
        max_context_length: Maximum character length for context
        
    Returns:
        Formatted context string for LLM
    """
    if not retrieval_results:
        return "No relevant context found."
    
    context_parts = []
    current_length = 0
    
    # Prioritize results based on intent
    sorted_results = _prioritize_by_intent(retrieval_results, intent)
    
    for i, result in enumerate(sorted_results):
        # Format individual result
        doc_context = _format_result_context(result, i + 1)
        
        # Check if adding this would exceed limit
        if current_length + len(doc_context) > max_context_length:
            break
            
        context_parts.append(doc_context)
        current_length += len(doc_context)
    
    # Build final context
    context = "\n\n".join(context_parts)
    
    if current_length >= max_context_length:
        context += f"\n\n[Note: Context truncated - showing top {len(context_parts)} results]"
    
    return context


async def generate_response(
    query: str,
    context: str,
    intent,  # Can be either dict or IntentResult
    chat_client: Any,
    chat_history: List[Dict[str, str]] = None,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    max_tokens: int = 1500
) -> AsyncGenerator[str, None]:
    """Generate streaming response using LLM.
    
    Args:
        query: User query
        context: Built context from retrieval
        intent: Classified intent
        chat_client: LLM client for generation
        chat_history: Recent conversation history
        
    Yields:
        Streaming response chunks
    """
    try:
        # Build system prompt based on intent
        system_prompt = _build_system_prompt(intent)
        
        # Build message history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent chat history if provided
        if chat_history:
            for msg in chat_history[-6:]:  # Last 6 messages
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current query with context
        user_message = f"""
<context>
{context}
</context>

<question>
{query}
</question>
"""
        messages.append({"role": "user", "content": user_message})
        
        # Debug: log the request details (reduced logging for performance)
        logger.info(f"Azure OpenAI request - model: {model_name}, temperature: {temperature}, max_tokens: {max_tokens}")
        
        # Generate streaming response
        try:
            response_stream = chat_client.chat.completions.create(
                model=model_name,  # Use the deployment name for Azure OpenAI
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"Azure OpenAI request failed: {type(e).__name__}: {str(e)}")
            logger.error(f"Request details: model={model_name}, messages_count={len(messages)}, stream=True, temp={temperature}, max_tokens={max_tokens}")
            raise
        
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        yield f"I encountered an error while generating a response: {str(e)}"


def verify_answer(answer: str, context: str, query: str) -> Dict[str, Any]:
    """Verify answer quality and relevance.
    
    Args:
        answer: Generated answer
        context: Context used for generation
        query: Original user query
        
    Returns:
        Verification results with quality metrics
    """
    metrics = {
        "has_content": len(answer.strip()) > 10,
        "not_error": not _contains_error_phrases(answer),
        "contextual": _answer_uses_context(answer, context),
        "relevant": _answer_addresses_query(answer, query),
        "complete": not _answer_seems_truncated(answer),
        "confidence_score": 0.0
    }
    
    # Calculate overall confidence
    passed_checks = sum(metrics[k] for k in metrics if k != "confidence_score")
    metrics["confidence_score"] = passed_checks / 5.0
    
    return metrics


def extract_source_chips(
    retrieval_results: List[SearchResult],
    max_chips: int = 5
) -> List[SourceChip]:
    """Extract source citation chips from retrieval results.
    
    Args:
        retrieval_results: Search results to create chips from
        max_chips: Maximum number of chips to return
        
    Returns:
        List of SourceChip objects for UI display
    """
    chips = []
    
    for result in retrieval_results[:max_chips]:
        # Extract title from metadata or content
        title = result.metadata.get("title", "")
        if not title:
            # Fallback: use first sentence of content
            sentences = result.content.split('. ')
            title = sentences[0][:50] + "..." if len(sentences[0]) > 50 else sentences[0]
        
        # Create excerpt
        excerpt = _create_excerpt(result.content, max_length=150)
        
        chip = SourceChip(
            title=title,
            doc_id=result.doc_id,
            url=result.metadata.get("url"),
            excerpt=excerpt
        )
        chips.append(chip)
    
    return chips


def _get_intent_label(intent) -> str:
    """Safely extract intent label from either dict or IntentResult object."""
    if intent is None:
        return "unknown"
    if hasattr(intent, "intent"):
        return intent.intent
    if isinstance(intent, dict):
        return intent.get("intent", "unknown")
    return "unknown"


def _prioritize_by_intent(results: List[SearchResult], intent) -> List[SearchResult]:
    """Prioritize search results based on intent - handles both dict and IntentResult."""
    intent_label = _get_intent_label(intent)
    
    if intent_label == "list":
        # For list queries, prioritize results with structured data
        return sorted(results, key=lambda r: (
            -r.score,  # Primary: search relevance
            -len(r.metadata.get("api_names", [])),  # Secondary: number of APIs
            -len(r.content)  # Tertiary: content richness
        ))
    
    elif intent_label == "swagger":
        # For API queries, prioritize technical documentation
        return sorted(results, key=lambda r: (
            -_api_relevance_score(r),  # Primary: API relevance
            -r.score  # Secondary: search relevance
        ))
    
    else:
        # Default: sort by relevance score
        return sorted(results, key=lambda r: -r.score)


def _api_relevance_score(result: SearchResult) -> float:
    """Calculate API relevance score for a result."""
    score = 0.0
    content_lower = result.content.lower()
    
    # API-related keywords boost
    api_keywords = ["endpoint", "request", "response", "parameter", "field", "api", "swagger"]
    for keyword in api_keywords:
        score += content_lower.count(keyword) * 0.1
    
    # Metadata boosts
    if result.metadata.get("type") == "api_spec":
        score += 0.5
    
    return score


def _format_result_context(result: SearchResult, index: int) -> str:
    """Format a single search result for context."""
    title = result.metadata.get("title", f"Document {index}")
    
    context_block = f"[Source {index}: {title}]\n{result.content}"
    
    # Add metadata if relevant
    if result.metadata.get("api_names"):
        context_block += f"\nRelated APIs: {', '.join(result.metadata['api_names'])}"
    
    return context_block


def _build_system_prompt(intent) -> str:
    """Build system prompt based on classified intent - handles both dict and IntentResult."""
    
    intent_label = _get_intent_label(intent)
    
    base_prompt = """You are an enterprise knowledge assistant for product owners and developers. Create clear, executive-ready briefings.

Instructions:
- Write clear, scannable responses using bullets and concise paragraphs
- Start with a brief overview, then provide key details
- Use **bold headers** to organize sections and HTML tags where helpful (<p>, <ul>, <li>, <table>)
- Include specific examples and use cases when available
- For technical details, explain "what it is," "why it matters," and "how it's used"
- For API field/parameter questions, focus solely on the exact field requested
- Generate 3 relevant follow-up questions when appropriate
- If multiple results match criteria and there are more than 3, note "Multiple results match; top results shown. To view all, please rephrase your query to 'List all'."
- Avoid jargon dumps - explain terms clearly for business users
- If information is incomplete, acknowledge limitations clearly
- Use professional tone suitable for briefing executives"""
    
    if intent_label == "list":
        return base_prompt + """

Special Format for List Queries:
- Structure as "I have knowledge of the following [APIs/APGs/Products]:"
- Use numbered lists for APIs, bullet points for APGs/Products  
- Include follow-up questions: "Which specific [item] do you want to know more about?"
- For hierarchical data, show relationships (Product → APG → API)"""
        
    elif intent_label == "swagger":
        return base_prompt + """

Special Format for API Specification Queries:
- Focus on technical details: endpoints, parameters, response formats
- Use <pre><code> for JSON examples and technical specifications
- For field questions: return all associated API names directly linked to that exact field
- Highlight important constraints or requirements
- Reference specific API documentation when available"""
        
    # Note: workflow and restart intents not produced by current intent classification
        
    else:  # confluence or general
        return base_prompt + """

Special Format for General/Documentation Queries:
- Provide comprehensive explanations using the context
- Use **bold headers** and clear formatting with lists
- Include examples where appropriate
- For confluence intent: Include References section at end with page_url links as <h4>References</h4><ul><li>links</li></ul>
- Reference sources when making specific claims"""


def _contains_error_phrases(answer: str) -> bool:
    """Check if answer contains common error phrases."""
    error_phrases = [
        "i don't have information",
        "i don't know",
        "i cannot find", 
        "no information available",
        "unable to provide",
        "error occurred"
    ]
    
    return _text_matches_patterns(answer, error_phrases)


def _answer_uses_context(answer: str, context: str) -> bool:
    """Check if answer appears to use provided context."""
    return _calculate_word_overlap(context, answer, threshold=0.1)


def _answer_addresses_query(answer: str, query: str) -> bool:
    """Check if answer appears to address the query."""
    return _calculate_word_overlap(query, answer, threshold=0.3)


def _answer_seems_truncated(answer: str) -> bool:
    """Check if answer appears to be truncated."""
    # Check for abrupt endings
    truncation_indicators = [
        answer.endswith("..."),
        answer.endswith(" and"),
        answer.endswith(" but"),
        answer.endswith(" the"),
        len(answer) > 100 and not answer.rstrip().endswith(('.', '!', '?', ':'))
    ]
    
    return any(truncation_indicators)


def _create_excerpt(content: str, max_length: int = 150) -> str:
    """Create a short excerpt from content."""
    if len(content) <= max_length:
        return content
    
    # Try to break at sentence boundary
    truncated = content[:max_length]
    last_sentence = truncated.rfind('. ')
    
    if last_sentence > max_length * 0.6:  # If we can keep at least 60% and break cleanly
        return truncated[:last_sentence + 1]
    else:
        return truncated[:max_length - 3] + "..."