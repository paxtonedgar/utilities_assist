"""Response generation and context building services."""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, AsyncGenerator
from jinja2 import Environment, FileSystemLoader
from src.services.models import Passage, SourceChip, IntentResult

logger = logging.getLogger(__name__)

# Load jinja templates
template_dir = Path(__file__).parent.parent / "agent" / "prompts"
jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))


# === TEXT PROCESSING UTILITIES ===


def _extract_meaningful_words(text: str, exclude_stopwords: bool = True) -> set:
    """Extract meaningful words from text, optionally excluding stopwords.

    Centralizes word extraction logic used throughout the file.
    """
    words = set(re.findall(r"\w+", text.lower()))

    if exclude_stopwords:
        # Combined stopwords from both functions + query-specific terms
        stopwords = {
            "the",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "what",
            "how",
            "when",
            "where",
            "why",
            "which",
            "who",
            "is",
            "are",
        }
        words = words - stopwords

    return words


def _calculate_word_overlap(text1: str, text2: str, threshold: float = None) -> bool:
    """Calculate if word overlap between two texts meets threshold.

    Consolidates overlap calculation logic used in multiple validation functions.
    Uses default threshold from settings if not provided.
    """
    if not text1 or not text2:
        return True  # Can't verify without substantial text

    # Get default threshold from settings if not provided
    if threshold is None:
        from src.infra.settings import get_settings

        settings = get_settings()
        threshold = settings.quality_thresholds.context_overlap_threshold

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




async def generate_response(
    query: str,
    context: str,
    intent,  # Can be either dict or IntentResult
    chat_client: Any,
    chat_history: List[Dict[str, str]] = None,
    model_name: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    max_tokens: int = 2500,  # Increased from 1500 to prevent truncation
) -> AsyncGenerator[str, None]:
    """Generate streaming response using LLM with Jinja2 template.

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
        # Use answer.jinja template instead of hardcoded prompt
        template = jinja_env.get_template("answer.jinja")

        # Ensure intent is in correct format for template
        # Template expects intent.intent and intent.confidence
        if isinstance(intent, dict):
            template_intent = intent
        elif hasattr(intent, "intent") and hasattr(intent, "confidence"):
            template_intent = {"intent": intent.intent, "confidence": intent.confidence}
        else:
            # Fallback for unknown intent structure
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

        # Build message with template-rendered prompt
        messages = [{"role": "user", "content": prompt}]

        # Debug: log the request details and context info
        logger.info(
            f"Azure OpenAI request - model: {model_name}, temperature: {temperature}, max_tokens: {max_tokens}"
        )
        logger.info(f"Context length: {len(context)} chars, Query: '{query[:100]}...'")
        logger.debug(f"Full prompt preview: {prompt[:500]}...")

        # Generate streaming response
        try:
            response_stream = chat_client.chat.completions.create(
                model=model_name,  # Use the deployment name for Azure OpenAI
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"Azure OpenAI request failed: {type(e).__name__}: {str(e)}")
            logger.error(
                f"Request details: model={model_name}, messages_count={len(messages)}, stream=True, temp={temperature}, max_tokens={max_tokens}"
            )
            raise

        # Collect response for debug logging
        response_chunks = []
        for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_chunks.append(content)
                yield content

        # Log response summary for debugging
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
        "confidence_score": 0.0,
    }

    # Calculate overall confidence
    passed_checks = sum(metrics[k] for k in metrics if k != "confidence_score")
    metrics["confidence_score"] = passed_checks / 5.0

    return metrics


def extract_source_chips(
    retrieval_results: List[Passage], max_chips: int = 5
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










# _build_system_prompt function removed - now using answer.jinja template instead


def _contains_error_phrases(answer: str) -> bool:
    """Check if answer contains common error phrases."""
    error_phrases = [
        "i don't have information",
        "i don't know",
        "i cannot find",
        "no information available",
        "unable to provide",
        "error occurred",
    ]

    return _text_matches_patterns(answer, error_phrases)


def _answer_uses_context(answer: str, context: str) -> bool:
    """Check if answer appears to use provided context."""
    return _calculate_word_overlap(
        context, answer
    )  # Uses default threshold from settings


def _answer_addresses_query(answer: str, query: str) -> bool:
    """Check if answer appears to address the query."""
    from src.infra.settings import get_settings

    settings = get_settings()
    return _calculate_word_overlap(
        query, answer, threshold=settings.quality_thresholds.query_overlap_threshold
    )


def _answer_seems_truncated(answer: str) -> bool:
    """Check if answer appears to be truncated."""
    # Check for abrupt endings
    truncation_indicators = [
        answer.endswith("..."),
        answer.endswith(" and"),
        answer.endswith(" but"),
        answer.endswith(" the"),
        len(answer) > 100 and not answer.rstrip().endswith((".", "!", "?", ":")),
    ]

    return any(truncation_indicators)


def _create_excerpt(content: str, max_length: int = 150) -> str:
    """Create a short excerpt from content."""
    if len(content) <= max_length:
        return content

    # Try to break at sentence boundary
    truncated = content[:max_length]
    last_sentence = truncated.rfind(". ")

    if (
        last_sentence > max_length * 0.6
    ):  # If we can keep at least 60% and break cleanly
        return truncated[: last_sentence + 1]
    else:
        return truncated[: max_length - 3] + "..."
