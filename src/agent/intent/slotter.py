# src/agent/intent/slotter.py
"""
Regex-first intent classification to replace expensive LLM normalize+intent.
Provides fast (<1ms) intent classification with optional ONNX fallback for ambiguous cases.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Set
from src.util.cache import slot_cache

logger = logging.getLogger(__name__)

Intent = Literal["info", "procedure", "mixed"]
Specificity = Literal["low", "med", "high"]
TimeUrgency = Literal["none", "soft", "hard"]


@dataclass(frozen=True)
class Colors:
    """Coloring attributes that steer retrieval and presentation."""

    actionability_est: float  # [0,3] from verbs/steps/endpoints
    suite_affinity: Dict[str, float]  # {"jira":0.9,"api":0.7,"teams":0.4}
    artifact_types: Set[str]  # {"endpoint","ticket","form","runbook","channel","table"}
    specificity: Specificity  # "low|med|high" anchor presence
    troubleshoot_flag: bool  # error codes, stack traces
    time_urgency: TimeUrgency  # "none|soft|hard" urgency level
    safety_flags: Set[str]  # {"pii","cred","secrets"}


# Stable default for backward compatibility
DEFAULT_COLORS = Colors(
    actionability_est=0.0,
    suite_affinity={},
    artifact_types=set(),
    specificity="low",
    troubleshoot_flag=False,
    time_urgency="none",
    safety_flags=set(),
)


@dataclass(frozen=True)
class SlotResult:
    """Result of intent classification with coloring."""

    intent: Intent
    doish: bool  # "sounds like a task"
    reasons: List[str]  # short flags explaining the route
    features: Dict[str, int]  # debug: binary features used
    confidence: float  # 0.0-1.0 confidence in classification
    colors: Colors  # NEW: coloring attributes for retrieval steering


# Compile regex patterns once for performance
PROCEDURE_PATTERNS = [
    re.compile(r"\b(?:how|guide|tutorial|walkthrough)\b", re.IGNORECASE),
    re.compile(r"\bonboard(?:ing)?|on-?board\b", re.IGNORECASE),
    re.compile(r"\bsteps?\b", re.IGNORECASE),
    re.compile(r"\bjira|servicenow|request|intake|ticket\b", re.IGNORECASE),
    re.compile(r"\bcreate|submit|open\s+(?:a\s+)?(?:ticket|request)\b", re.IGNORECASE),
    re.compile(r"\bapi\s+key\b", re.IGNORECASE),
    re.compile(r"\bendpoint\b", re.IGNORECASE),
    re.compile(r"\b(?:POST|GET|PUT|DELETE|PATCH)\s+/\w+", re.IGNORECASE),
    re.compile(r"\bsetup|configure|install\b", re.IGNORECASE),
    re.compile(r"\btroubleshoot(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\bfix|resolve|solve\b", re.IGNORECASE),
]

INFO_PATTERNS = [
    re.compile(r"\bwhat\s+is\b", re.IGNORECASE),
    re.compile(r"\boverview\b", re.IGNORECASE),
    re.compile(r"\bdefinition\b", re.IGNORECASE),
    re.compile(r"\bexplain\b", re.IGNORECASE),
    re.compile(r"\bdocs?|documentation\b", re.IGNORECASE),
    re.compile(r"\bdescribe\b", re.IGNORECASE),
    re.compile(r"\bmeaning\b", re.IGNORECASE),
    re.compile(r"\bunderstand\b", re.IGNORECASE),
]

# Question words that suggest information seeking
QUESTION_PATTERNS = [
    re.compile(r"^\s*(?:what|who|when|where|which)\b", re.IGNORECASE),
    re.compile(r"\?\s*$"),  # Ends with question mark
]

# Action words that suggest procedures
ACTION_PATTERNS = [
    re.compile(r"^\s*(?:how|can\s+i|help\s+me)\b", re.IGNORECASE),
    re.compile(r"\bneed\s+to\b", re.IGNORECASE),
    re.compile(r"\bwant\s+to\b", re.IGNORECASE),
]


def slot(user_text: str) -> SlotResult:
    """
    Classify user intent using regex-first approach.

    Args:
        user_text: User query text

    Returns:
        SlotResult with intent classification

    Examples:
        >>> result = slot("What is CIU?")
        >>> result.intent
        'info'
        >>> result.doish
        False

        >>> result = slot("How do I onboard to CIU?")
        >>> result.intent
        'procedure'
        >>> result.doish
        True

        >>> result = slot("CIU API documentation")
        >>> result.intent in ['info', 'mixed']
        True

        >>> result = slot("")
        >>> result.confidence
        0.0
    """
    if not user_text or not user_text.strip():
        return SlotResult(
            intent="info",
            doish=False,
            reasons=["empty_query"],
            features={},
            confidence=0.0,
            colors=DEFAULT_COLORS,
        )

    # Check cache first
    result = _get_cached_or_compute(user_text)
    if result:
        return result

    # Extract features and calculate scores
    features = _extract_features(user_text)
    scores = _calculate_pattern_scores(features)
    
    # Classify intent and determine properties
    intent, confidence = _classify_intent_from_scores(scores, user_text)
    doish = _determine_doish_flag(scores)
    reasons = _build_classification_reasons(scores, intent)
    colors = _compute_colors(user_text, features)
    
    # Build and cache result
    result = SlotResult(
        intent=intent,
        doish=doish,
        reasons=reasons,
        features=features,
        confidence=confidence,
        colors=colors,
    )
    
    _cache_result(user_text, result)
    
    logger.debug(
        f"Slotted '{user_text[:50]}...' -> {intent} (doish={doish}, conf={confidence:.2f})"
    )
    
    return result


def _extract_features(text: str) -> Dict[str, int]:
    """Extract binary features from text for classification."""
    features = {}

    # Procedure pattern features
    for i, pattern in enumerate(PROCEDURE_PATTERNS):
        features[f"proc_{i}"] = 1 if pattern.search(text) else 0

    # Info pattern features
    for i, pattern in enumerate(INFO_PATTERNS):
        features[f"info_{i}"] = 1 if pattern.search(text) else 0

    # Question pattern features
    for i, pattern in enumerate(QUESTION_PATTERNS):
        features[f"quest_{i}"] = 1 if pattern.search(text) else 0

    # Action pattern features
    for i, pattern in enumerate(ACTION_PATTERNS):
        features[f"action_{i}"] = 1 if pattern.search(text) else 0

    # Length features
    word_count = len(text.split())
    features["short_query"] = 1 if word_count <= 3 else 0
    features["long_query"] = 1 if word_count >= 10 else 0

    # Punctuation features
    features["has_question_mark"] = 1 if "?" in text else 0
    features["has_imperative"] = (
        1
        if any(text.lower().startswith(imp) for imp in ["help", "show", "tell", "give"])
        else 0
    )

    return features


def _looks_like_procedure_heuristic(text: str) -> bool:
    """Heuristic fallback for procedure detection."""
    text_lower = text.lower()

    # Check for imperative mood indicators
    imperative_starters = ["help", "show", "tell", "give", "provide", "list", "find"]
    if any(text_lower.startswith(starter) for starter in imperative_starters):
        return True

    # Check for utility/API terms that often require procedures
    utility_terms = [
        "api",
        "service",
        "utility",
        "tool",
        "integration",
        "client",
        "auth",
    ]
    action_terms = ["access", "use", "connect", "register", "enable"]

    has_utility = any(term in text_lower for term in utility_terms)
    has_action = any(term in text_lower for term in action_terms)

    return has_utility and has_action


# Utility functions for backward compatibility
def get_intent_confidence(slot_result: SlotResult) -> float:
    """Extract confidence score for backward compatibility."""
    return slot_result.confidence


def get_intent_label(slot_result: SlotResult) -> str:
    """Extract intent label for backward compatibility."""
    return slot_result.intent


def _get_cached_or_compute(user_text: str) -> SlotResult:
    """Check cache for existing result."""
    normalized = " ".join(user_text.lower().strip().split())
    return slot_cache.get(normalized)


def _calculate_pattern_scores(features: Dict[str, int]) -> Dict[str, int]:
    """Calculate scores for all pattern types."""
    return {
        "procedure": sum(features.get(f"proc_{i}", 0) for i in range(len(PROCEDURE_PATTERNS))),
        "info": sum(features.get(f"info_{i}", 0) for i in range(len(INFO_PATTERNS))),
        "question": sum(features.get(f"quest_{i}", 0) for i in range(len(QUESTION_PATTERNS))),
        "action": sum(features.get(f"action_{i}", 0) for i in range(len(ACTION_PATTERNS)))
    }


def _classify_intent_from_scores(scores: Dict[str, int], user_text: str) -> tuple[Intent, float]:
    """Classify intent based on pattern scores with fallback logic."""
    procedure_score = scores["procedure"]
    info_score = scores["info"]
    question_score = scores["question"]
    action_score = scores["action"]
    
    # Primary classification logic
    if procedure_score > 0 and info_score > 0:
        return "mixed", 0.8
    elif procedure_score > 0 or action_score > 0:
        return "procedure", (0.9 if procedure_score > 1 else 0.7)
    elif info_score > 0 or question_score > 0:
        return "info", (0.9 if info_score > 1 else 0.7)
    else:
        # Fallback heuristics
        return _apply_fallback_heuristics(user_text)


def _apply_fallback_heuristics(user_text: str) -> tuple[Intent, float]:
    """Apply heuristic fallback when no patterns match."""
    if _looks_like_procedure_heuristic(user_text):
        return "procedure", 0.5
    else:
        return "info", 0.4


def _determine_doish_flag(scores: Dict[str, int]) -> bool:
    """Determine if query sounds like a task/action."""
    return scores["procedure"] > 0 or scores["action"] > 0


def _build_classification_reasons(scores: Dict[str, int], intent: Intent) -> List[str]:
    """Build list of classification reasons based on scores."""
    reasons = []
    
    # Add procedure/action reasons
    if scores["procedure"] > 0:
        reasons.append(f"procedure_patterns:{scores['procedure']}")
    if scores["action"] > 0:
        reasons.append(f"action_patterns:{scores['action']}")
    
    # Add info/question reasons
    if scores["info"] > 0:
        reasons.append(f"info_patterns:{scores['info']}")
    if scores["question"] > 0:
        reasons.append(f"question_patterns:{scores['question']}")
    
    # Add fallback reasons if no patterns matched
    if not reasons:
        if intent == "procedure":
            reasons.append("heuristic_procedure")
        else:
            reasons.append("default_info")
    
    return reasons


def _cache_result(user_text: str, result: SlotResult) -> None:
    """Cache the classification result."""
    normalized = " ".join(user_text.lower().strip().split())
    slot_cache.put(normalized, result)


def _compute_colors(text: str, features: Dict[str, int]) -> Colors:
    """Compute coloring attributes using ColorsBuilder singleton."""
    try:
        from .coloring.builder import get_colors_builder

        builder = get_colors_builder()
        return builder.compute(text, features)
    except Exception as e:
        logger.error(f"Colors computation failed: {e}")
        return DEFAULT_COLORS
