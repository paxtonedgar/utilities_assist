"""Actionability scorer - computes actionability estimate [0,3] from affordances."""

import logging
from typing import Dict, Set, Tuple

logger = logging.getLogger(__name__)


class ActionabilityScorer:
    """
    Single responsibility: compute actionability score [0,3] from query affordances.
    
    Uses normalized unit steps to make thresholds portable:
    - endpoint present → +1.0
    - concrete ticket verb → +1.0 
    - explicit step/runbook cues → +1.0
    - secondary artifacts (forms, channels) → +0.5 each (max +1.0)
    - Clamp final score to 3.0
    """
    
    def __init__(self, weights: Dict[str, float]):
        """Initialize with weights from centralized config."""
        self.weights = weights
    
    def score(self, text: str, features: Dict[str, int]) -> Tuple[float, Set[str]]:
        """
        Compute actionability score and detected artifact types.
        
        Args:
            text: User query text
            features: Binary features from main slotter
            
        Returns:
            (actionability_score, artifact_types_set)
            
        Examples:
            >>> scorer = ActionabilityScorer({"api_endpoint": 1.0, "jira_ticket": 1.0})
            >>> score, artifacts = scorer.score("POST /api/users", {})
            >>> score >= 1.0 and "endpoint" in artifacts
            True
        """
        score = 0.0
        artifacts = set()
        
        # HTTP endpoints (strong procedure signal)
        if self._has_http_endpoint(text):
            score += self.weights["api_endpoint"]
            artifacts.add("endpoint")
        
        # ITSM ticket patterns
        if self._has_ticket_patterns(text):
            score += self.weights["jira_ticket"] 
            artifacts.add("ticket")
        
        # Step/procedure indicators
        if self._has_step_patterns(text):
            score += self.weights["runbook_step"]
            artifacts.add("runbook")
        
        # Secondary artifacts (capped contribution)
        secondary_score = 0.0
        
        if self._has_form_patterns(text):
            secondary_score += self.weights["form_pipeline"]
            artifacts.add("form")
            
        if self._has_channel_patterns(text):
            secondary_score += self.weights["form_pipeline"]  
            artifacts.add("channel")
        
        if self._has_table_patterns(text):
            secondary_score += self.weights["form_pipeline"]
            artifacts.add("table")
        
        # Cap secondary artifacts contribution
        secondary_score = min(secondary_score, self.weights["secondary_artifacts"])
        score += secondary_score
        
        # Clamp final score to 3.0
        final_score = min(score, 3.0)
        
        logger.debug(f"Actionability: {final_score:.1f} from {artifacts} (text: '{text[:30]}...')")
        
        return final_score, artifacts
    
    def _has_http_endpoint(self, text: str) -> bool:
        """Check for HTTP endpoint patterns."""
        from .patterns import has_pattern_match
        return has_pattern_match(text, "http_endpoint")
    
    def _has_ticket_patterns(self, text: str) -> bool:
        """Check for ticket creation/submission patterns."""
        from .patterns import has_pattern_match
        return has_pattern_match(text, "ticket_verbs")
    
    def _has_step_patterns(self, text: str) -> bool:
        """Check for step/procedure patterns."""
        from .patterns import has_pattern_match
        return has_pattern_match(text, "step_indicators")
    
    def _has_form_patterns(self, text: str) -> bool:
        """Check for form/pipeline patterns."""
        from .patterns import has_pattern_match
        return has_pattern_match(text, "form_patterns")
    
    def _has_channel_patterns(self, text: str) -> bool:
        """Check for channel/communication patterns."""
        from .patterns import has_pattern_match
        return has_pattern_match(text, "channel_patterns")
    
    def _has_table_patterns(self, text: str) -> bool:
        """Check for table/data patterns."""
        # Simple keyword-based detection
        text_lower = text.lower()
        return any(word in text_lower for word in ["table", "spreadsheet", "csv", "data"])