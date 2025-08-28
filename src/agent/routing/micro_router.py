# src/agent/routing/micro_router.py
"""
Micro-Router: Simple rule-based query routing without ML models.

Replaces complex intent classification with 6-10 rock-solid regex patterns.
Single decision path, no cascading fallbacks, maximum debuggability.
"""

import re
import logging
from typing import Literal, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Route destinations
RouteType = Literal[
    "api",        # API documentation queries  
    "list",       # Enumeration queries
    "compare",    # X vs Y comparisons
    "procedure",  # How-to/setup queries
    "general"     # Default documentation search
]


@dataclass
class RouteResult:
    """Result of micro-routing with explanation."""
    route: RouteType
    confidence: float  # 0.0-1.0 confidence in routing decision
    pattern_matched: str  # Which pattern triggered the route
    query_normalized: str  # Cleaned query for downstream use


# Compiled regex patterns for maximum performance
API_PATTERNS = [
    re.compile(r"\b(?:GET|POST|PUT|DELETE|PATCH)\s+/\S+", re.IGNORECASE),
    re.compile(r"\bendpoint|swagger|openapi\b", re.IGNORECASE),
    re.compile(r"\bapi\s+(?:documentation|docs|spec|reference)\b", re.IGNORECASE),
    re.compile(r"\brest\s+api\b", re.IGNORECASE),
]

LIST_PATTERNS = [
    re.compile(r"^(?:list|show me|what are all|how many)\b", re.IGNORECASE),
    re.compile(r"\ball\s+(?:available|existing)\b", re.IGNORECASE),
    re.compile(r"\blist\s+(?:of\s+)?(?:apis|utilities|services|endpoints)\b", re.IGNORECASE),
]

COMPARE_PATTERNS = [
    re.compile(r"\b(?:vs|versus|compare|difference|which is better)\b", re.IGNORECASE),
    re.compile(r"\b\w+\s+vs\s+\w+\b", re.IGNORECASE),  # "X vs Y" pattern
    re.compile(r"\bdifference between\b", re.IGNORECASE),
]

PROCEDURE_PATTERNS = [
    re.compile(r"\bhow do i|how to|can i\b", re.IGNORECASE),
    re.compile(r"\bsetup|configure|install|onboard\b", re.IGNORECASE),
    re.compile(r"\bsteps|guide|tutorial|walkthrough\b", re.IGNORECASE),
    re.compile(r"\btroubleshoot|fix|resolve|solve\b", re.IGNORECASE),
    re.compile(r"\bcreate|submit|open\s+(?:a\s+)?(?:ticket|request)\b", re.IGNORECASE),
]

# Utility acronyms that should stay in general search (don't over-specialize)
UTILITY_ACRONYMS = {
    "ciu", "customer interaction utility",
    "etu", "enhanced transaction utility", 
    "csu", "customer summary utility",
    "au", "account utility",
    "pcu", "product catalog utility"
}


def micro_route(query: str) -> RouteResult:
    """
    Route query using simple, deterministic rules.
    
    No models, no cascading fallbacks, maximum debuggability.
    Returns single route decision with explanation.
    """
    if not query or not query.strip():
        return RouteResult(
            route="general",
            confidence=0.0,
            pattern_matched="empty_query",
            query_normalized=""
        )
    
    # Normalize query for pattern matching
    normalized = query.strip().lower()
    
    # Check for utility acronyms first (guardrail)
    if _is_utility_acronym_query(normalized):
        logger.info(f"Utility acronym detected: '{query}' -> general search")
        return RouteResult(
            route="general",
            confidence=0.9,
            pattern_matched="utility_acronym",
            query_normalized=normalized
        )
    
    # API documentation patterns
    for pattern in API_PATTERNS:
        if pattern.search(query):
            logger.info(f"API pattern matched: '{query}' -> api search")
            return RouteResult(
                route="api",
                confidence=0.8,
                pattern_matched=f"api_pattern: {pattern.pattern}",
                query_normalized=normalized
            )
    
    # List/enumeration patterns  
    for pattern in LIST_PATTERNS:
        if pattern.search(query):
            logger.info(f"List pattern matched: '{query}' -> list handler")
            return RouteResult(
                route="list",
                confidence=0.9,
                pattern_matched=f"list_pattern: {pattern.pattern}",
                query_normalized=normalized
            )
    
    # Comparison patterns
    for pattern in COMPARE_PATTERNS:
        if pattern.search(query):
            logger.info(f"Compare pattern matched: '{query}' -> comparison")
            return RouteResult(
                route="compare", 
                confidence=0.8,
                pattern_matched=f"compare_pattern: {pattern.pattern}",
                query_normalized=normalized
            )
    
    # Procedure/how-to patterns
    for pattern in PROCEDURE_PATTERNS:
        if pattern.search(query):
            logger.info(f"Procedure pattern matched: '{query}' -> procedure")
            return RouteResult(
                route="procedure",
                confidence=0.7,
                pattern_matched=f"procedure_pattern: {pattern.pattern}",
                query_normalized=normalized
            )
    
    # Default to general documentation search
    logger.info(f"No specific pattern matched: '{query}' -> general search")
    return RouteResult(
        route="general",
        confidence=0.5,
        pattern_matched="default",
        query_normalized=normalized
    )


def _is_utility_acronym_query(query: str) -> bool:
    """Check if query is a short utility acronym that should stay in general search."""
    tokens = query.strip().split()
    
    # Must be short (≤3 tokens) 
    if len(tokens) > 3:
        return False
    
    # Check if any token matches known utility acronyms
    for token in tokens:
        token_clean = token.lower().strip('.,?!')
        if token_clean in UTILITY_ACRONYMS:
            return True
    
    return False


def extract_comparison_entities(query: str) -> Optional[tuple[str, str]]:
    """
    Extract X and Y from comparison queries for multi-search.
    
    Returns (entity1, entity2) if successfully extracted, None otherwise.
    """
    # Try "X vs Y" pattern
    vs_match = re.search(r"\b(\w+(?:\s+\w+)?)\s+(?:vs|versus)\s+(\w+(?:\s+\w+)?)\b", query, re.IGNORECASE)
    if vs_match:
        return vs_match.group(1).strip(), vs_match.group(2).strip()
    
    # Try "difference between X and Y"
    diff_match = re.search(r"\bdifference\s+between\s+(\w+(?:\s+\w+)?)\s+and\s+(\w+(?:\s+\w+)?)\b", query, re.IGNORECASE)
    if diff_match:
        return diff_match.group(1).strip(), diff_match.group(2).strip()
    
    # Try "compare X and Y" 
    comp_match = re.search(r"\bcompare\s+(\w+(?:\s+\w+)?)\s+(?:and|with)\s+(\w+(?:\s+\w+)?)\b", query, re.IGNORECASE)
    if comp_match:
        return comp_match.group(1).strip(), comp_match.group(2).strip()
    
    return None


# Test patterns during development
if __name__ == "__main__":
    test_queries = [
        "how do I setup CIU",
        "GET /api/v1/users endpoint",
        "list all available APIs", 
        "compare CIU vs ETU",
        "what is Customer Interaction Utility",
        "swagger documentation",
        "troubleshoot connection issues",
        "ETU",  # utility acronym
    ]
    
    for query in test_queries:
        result = micro_route(query)
        print(f"'{query}' -> {result.route} ({result.confidence:.1f}) [{result.pattern_matched}]")