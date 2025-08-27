"""
Query Analysis for Smart Threshold Selection

Analyzes query characteristics to determine optimal thresholds
for different types of API documentation retrieval.
"""

import re
import logging
from typing import Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries that need different threshold strategies."""

    GENERIC_API = "generic_api"  # "list all APIs", "what APIs are available"
    SPECIFIC_UTILITY = "specific_utility"  # "CIU APIs", "ETU onboarding"
    PROCEDURE = "procedure"  # "how to onboard", "steps to configure"
    INFO = "info"  # "what is X", "tell me about Y"
    TROUBLESHOOT = "troubleshoot"  # "fix error", "not working"


@dataclass
class QueryCharacteristics:
    """Analysis of query characteristics affecting threshold selection."""

    query_type: QueryType
    utility_specificity: float  # 0.0 = generic, 1.0 = very specific utility
    procedure_indicators: int  # Count of procedural words/patterns
    api_focus: bool  # Query specifically about APIs
    confidence: float  # Confidence in the analysis
    reasoning: str  # Human-readable explanation


class QueryAnalyzer:
    """Analyzes queries to determine optimal search thresholds."""

    # Utility-specific keywords and patterns
    UTILITY_KEYWORDS = {
        "ciu": [
            "ciu",
            "customer interaction utility",
            "customer interaction",
            "customer utility",
        ],
        "etu": ["etu", "enhanced transaction utility", "transaction utility"],
        "au": ["au", "account utility", "accounts utility"],
        "csu": ["csu", "customer summary utility", "customer summary"],
        "pcu": ["pcu", "payment card utility", "payment utility"],
    }

    # Generic API indicators
    GENERIC_API_PATTERNS = [
        r"\bapi\s*(?:s|es)?\s*(?:available|list|all)\b",
        r"\b(?:what|which|show|list)\s+api\s*(?:s|es)?\b",
        r"\b(?:available|existing)\s+api\s*(?:s|es)?\b",
        r"\ball\s+api\s*(?:s|es)?\b",
    ]

    # Procedural indicators
    PROCEDURE_PATTERNS = [
        r"\bhow\s+to\b",
        r"\bsteps?\s+to\b",
        r"\bguide\s+(?:to|for)\b",
        r"\bonboard(?:ing)?\b",
        r"\bsetup?\b",
        r"\bconfigure?\b",
        r"\bintegrat\w+\b",
        r"\bimplement\w*\b",
        r"\binstall\w*\b",
        r"\benable\b",
        r"\bstart\w+\b",
    ]

    # API-focused indicators
    API_INDICATORS = [
        r"\bapi\b",
        r"\bendpoint\s*(?:s)?\b",
        r"\bswagger\b",
        r"\brest\b",
        r"\bhttp\b",
        r"\brequest\s*(?:s)?\b",
    ]

    def analyze_query(self, query: str) -> QueryCharacteristics:
        """
        Analyze query to determine characteristics for threshold selection.

        Args:
            query: The search query to analyze

        Returns:
            QueryCharacteristics with analysis results
        """
        query_lower = query.lower().strip()

        # Analyze utility specificity
        utility_specificity = self._analyze_utility_specificity(query_lower)

        # Check for generic API patterns
        generic_api_score = self._count_generic_api_patterns(query_lower)

        # Count procedural indicators
        procedure_count = self._count_procedure_indicators(query_lower)

        # Check API focus
        api_focus = self._has_api_focus(query_lower)

        # Determine query type
        query_type = self._determine_query_type(
            query_lower,
            utility_specificity,
            generic_api_score,
            procedure_count,
            api_focus,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(
            utility_specificity, generic_api_score, procedure_count
        )

        # Generate reasoning
        reasoning = self._generate_reasoning(
            query_type,
            utility_specificity,
            generic_api_score,
            procedure_count,
            api_focus,
        )

        return QueryCharacteristics(
            query_type=query_type,
            utility_specificity=utility_specificity,
            procedure_indicators=procedure_count,
            api_focus=api_focus,
            confidence=confidence,
            reasoning=reasoning,
        )

    def _analyze_utility_specificity(self, query_lower: str) -> float:
        """Calculate how utility-specific the query is (0.0 = generic, 1.0 = very specific)."""
        specificity_score = 0.0

        for utility, keywords in self.UTILITY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Longer, more specific keywords get higher scores
                    keyword_score = len(keyword.split()) / 3.0  # Normalize by max words
                    specificity_score = max(specificity_score, keyword_score)

        return min(specificity_score, 1.0)

    def _count_generic_api_patterns(self, query_lower: str) -> int:
        """Count generic API query patterns."""
        count = 0
        for pattern in self.GENERIC_API_PATTERNS:
            if re.search(pattern, query_lower):
                count += 1
        return count

    def _count_procedure_indicators(self, query_lower: str) -> int:
        """Count procedural indicators in the query."""
        count = 0
        for pattern in self.PROCEDURE_PATTERNS:
            matches = re.findall(pattern, query_lower)
            count += len(matches)
        return count

    def _has_api_focus(self, query_lower: str) -> bool:
        """Check if query is focused on APIs."""
        for pattern in self.API_INDICATORS:
            if re.search(pattern, query_lower):
                return True
        return False

    def _determine_query_type(
        self,
        query_lower: str,
        utility_specificity: float,
        generic_api_score: int,
        procedure_count: int,
        api_focus: bool,
    ) -> QueryType:
        """Determine the primary query type."""

        # Check for troubleshooting
        if any(
            word in query_lower
            for word in ["error", "fix", "troubleshoot", "not working", "issue"]
        ):
            return QueryType.TROUBLESHOOT

        # Check for procedure queries
        if procedure_count > 0:
            return QueryType.PROCEDURE

        # Check for generic API queries
        if generic_api_score > 0 and utility_specificity < 0.3:
            return QueryType.GENERIC_API

        # Check for specific utility queries
        if utility_specificity > 0.5:
            return QueryType.SPECIFIC_UTILITY

        # Default to info
        return QueryType.INFO

    def _calculate_confidence(
        self, utility_specificity: float, generic_api_score: int, procedure_count: int
    ) -> float:
        """Calculate confidence in the analysis."""
        confidence = 0.5  # Base confidence

        # High utility specificity increases confidence
        if utility_specificity > 0.7:
            confidence += 0.3
        elif utility_specificity > 0.3:
            confidence += 0.1

        # Clear patterns increase confidence
        if generic_api_score > 1:
            confidence += 0.2
        elif generic_api_score > 0:
            confidence += 0.1

        if procedure_count > 1:
            confidence += 0.2
        elif procedure_count > 0:
            confidence += 0.1

        return min(confidence, 1.0)

    def _generate_reasoning(
        self,
        query_type: QueryType,
        utility_specificity: float,
        generic_api_score: int,
        procedure_count: int,
        api_focus: bool,
    ) -> str:
        """Generate human-readable reasoning for the classification."""
        reasons = []

        if utility_specificity > 0.5:
            reasons.append(f"high utility specificity ({utility_specificity:.2f})")
        elif utility_specificity > 0.2:
            reasons.append(f"moderate utility specificity ({utility_specificity:.2f})")

        if generic_api_score > 0:
            reasons.append(f"{generic_api_score} generic API patterns")

        if procedure_count > 0:
            reasons.append(f"{procedure_count} procedural indicators")

        if api_focus:
            reasons.append("API-focused terminology")

        base_reason = f"Classified as {query_type.value}"
        if reasons:
            return f"{base_reason}: {', '.join(reasons)}"
        else:
            return f"{base_reason}: default classification"


class SmartThresholdSelector:
    """Selects optimal quality thresholds based on query analysis."""

    def __init__(self):
        self.analyzer = QueryAnalyzer()

    def get_quality_thresholds(self, query: str) -> Dict[str, float]:
        """
        Get index-specific quality thresholds based on query analysis.

        Args:
            query: The search query

        Returns:
            Dict mapping index names to threshold values
        """
        analysis = self.analyzer.analyze_query(query)

        # Base thresholds
        confluence_threshold = 0.15
        swagger_threshold = 0.05

        # Adjust based on query type
        if analysis.query_type == QueryType.GENERIC_API:
            # Generic API queries: lower swagger threshold to catch more results
            swagger_threshold = 0.02
            confluence_threshold = 0.10  # Also lower confluence threshold

        elif analysis.query_type == QueryType.SPECIFIC_UTILITY:
            # Specific utility queries: higher confidence, can be more selective
            swagger_threshold = 0.08
            confluence_threshold = 0.20

        elif analysis.query_type == QueryType.PROCEDURE:
            # Procedural queries: balance both sources
            swagger_threshold = 0.05
            confluence_threshold = 0.15

        elif analysis.query_type == QueryType.TROUBLESHOOT:
            # Troubleshooting: cast a wider net
            swagger_threshold = 0.03
            confluence_threshold = 0.12

        # Fine-tune based on utility specificity
        if analysis.utility_specificity > 0.7:
            # Very specific: can be more selective
            swagger_threshold *= 1.5
            confluence_threshold *= 1.2
        elif analysis.utility_specificity < 0.2:
            # Very generic: cast wider net
            swagger_threshold *= 0.7
            confluence_threshold *= 0.8

        logger.info(
            f"Smart thresholds for '{query[:50]}...': "
            f"confluence={confluence_threshold:.3f}, swagger={swagger_threshold:.3f} "
            f"({analysis.reasoning})"
        )

        return {
            "khub-opensearch-index": confluence_threshold,
            "khub-opensearch-swagger-index": swagger_threshold,
        }


# Global instance for use across the application
_smart_threshold_selector = SmartThresholdSelector()


def get_smart_thresholds(query: str) -> Dict[str, float]:
    """Convenience function to get smart thresholds with global selector."""
    return _smart_threshold_selector.get_quality_thresholds(query)


