"""Suite affinity calculator - reuses existing YAML registry for DRY compliance."""

import logging
from typing import Dict
import re

logger = logging.getLogger(__name__)


class SuiteAffinityCalculator:
    """
    Calculate domain-specific affinity scores using existing suite registry.
    
    Reuses YAML suite manifests to maintain DRY compliance and leverage
    existing domain knowledge for routing decisions.
    """
    
    def __init__(self, suite_registry: dict, weights: dict):
        """
        Initialize with suite registry and scoring weights.
        
        Args:
            suite_registry: Dictionary mapping suite names to configurations
            weights: Scoring weights for different signal types
        """
        self.suite_registry = suite_registry
        self.weights = weights
    
    def calculate(self, text: str) -> Dict[str, float]:
        """
        Calculate affinity scores for all registered suites.
        
        Args:
            text: User query text
            
        Returns:
            Dictionary mapping suite names to affinity scores [0,1]
            
        Examples:
            >>> calc = SuiteAffinityCalculator({"jira": {"keywords": ["ticket"]}}, weights)
            >>> scores = calc.calculate("create jira ticket")  
            >>> scores.get("jira", 0) > 0
            True
        """
        affinities = {}
        
        for suite_name, suite_config in self.suite_registry.items():
            try:
                score = self._calculate_suite_score(text, suite_config)
                if score > 0:
                    affinities[suite_name] = min(score, 1.0)  # Cap at 1.0
            except Exception as e:
                logger.warning(f"Error calculating affinity for {suite_name}: {e}")
        
        # Sort by score for logging
        sorted_affinities = dict(sorted(affinities.items(), key=lambda x: x[1], reverse=True))
        
        if sorted_affinities:
            logger.debug(f"Suite affinities: {sorted_affinities} (text: '{text[:30]}...')")
        
        return sorted_affinities
    
    def _calculate_suite_score(self, text: str, suite_config: dict) -> float:
        """Calculate score for a single suite against text."""
        score = 0.0
        text_lower = text.lower()
        
        # URL patterns (strongest signal)
        url_patterns = suite_config.get("url_patterns", [])
        for url_pattern in url_patterns:
            try:
                if re.search(url_pattern, text, re.IGNORECASE):
                    score += self.weights["suite_url_patterns"]
            except re.error:
                logger.warning(f"Invalid URL pattern: {url_pattern}")
        
        # Regex hints (medium signal)
        hints_any = suite_config.get("hints", {}).get("any", [])
        for hint_pattern in hints_any:
            try:
                if re.search(hint_pattern, text, re.IGNORECASE):
                    score += self.weights["suite_hints_any"]
            except re.error:
                logger.warning(f"Invalid hint pattern: {hint_pattern}")
        
        # Keywords (weakest signal)
        keywords = suite_config.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in text_lower:
                score += self.weights["suite_keywords"]
        
        return score