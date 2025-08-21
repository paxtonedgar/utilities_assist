"""Suite affinity calculator - reuses existing YAML registry for DRY compliance."""

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


class SuiteAffinityCalculator:
    """
    Single responsibility: compute suite affinity scores using existing YAML registry.
    
    Reuses the same manifest fields that extractors use:
    - hints.any (regex) → +0.5 each (cap 1.0)
    - keywords (case-insensitive) → +0.2 each  
    - url_patterns → +0.8 each
    - Cap per suite at 1.0, sort by descending score
    """
    
    def __init__(self, registry: dict, weights: Dict[str, float]):
        """
        Initialize with suite registry and weights.
        
        Args:
            registry: Suite registry loaded from YAML manifests
            weights: COLORING_WEIGHTS from config
        """
        self.registry = registry
        self.weights = weights
    
    def calculate(self, text: str) -> Dict[str, float]:
        """
        Calculate affinity scores for all known suites.
        
        Args:
            text: User query text
            
        Returns:
            Dict of {suite_name: score} for meaningful matches (score > 0.1)
            
        Examples:
            >>> calc = SuiteAffinityCalculator({\"jira\": {\"keywords\": [\"ticket\"]}}, weights)
            >>> scores = calc.calculate(\"create jira ticket\")  
            >>> scores.get(\"jira\", 0) > 0
            True
        """
        if not self.registry:
            return {}
            
        affinities = {}
        
        for suite_name, suite_config in self.registry.items():
            score = self._calculate_suite_score(text, suite_config)
            
            # Only include meaningful matches
            if score > 0.1:
                affinities[suite_name] = score
        
        # Sort by descending score for consistent telemetry
        sorted_affinities = dict(
            sorted(affinities.items(), key=lambda item: item[1], reverse=True)
        )
        
        logger.debug(f\"Suite affinities: {sorted_affinities} (text: '{text[:30]}...')\")
        
        return sorted_affinities
    
    def _calculate_suite_score(self, text: str, suite_config: dict) -> float:
        \"\"\"Calculate score for a single suite against text.\"\"\"
        score = 0.0
        text_lower = text.lower()
        
        # URL patterns (strong signal) 
        url_patterns = suite_config.get(\"url_patterns\", [])
        for url_pattern in url_patterns:
            try:
                if re.search(url_pattern, text, re.IGNORECASE):
                    score += self.weights[\"suite_url_patterns\"]
            except re.error:
                logger.warning(f\"Invalid URL pattern: {url_pattern}\")
        
        # Regex hints from hints.any field
        hints_any = suite_config.get(\"hints\", {}).get(\"any\", [])
        for hint_pattern in hints_any:
            try:
                if re.search(hint_pattern, text, re.IGNORECASE):
                    score += self.weights[\"suite_hints_any\"]
            except re.error:
                logger.warning(f\"Invalid hint pattern: {hint_pattern}\")
        
        # Simple keyword matching
        keywords = suite_config.get(\"keywords\", [])
        for keyword in keywords:
            if keyword.lower() in text_lower:
                score += self.weights[\"suite_keywords\"]
        
        # Cap per suite at 1.0
        return min(score, 1.0)