"""Specificity classifier - determines query anchoring from distinct identifiers."""

import logging
from typing import Set

logger = logging.getLogger(__name__)


class SpecificityClassifier:
    """
    Classify query specificity based on distinct anchor patterns.
    
    Counts unique anchoring identifiers to determine how specific
    a query is, affecting search strategy (tight vs broad).
    """
    
    def __init__(self, thresholds: dict):
        """Initialize with detection thresholds from centralized config."""
        self.thresholds = thresholds
    
    def classify(self, text: str) -> str:
        """
        Classify query specificity as low/med/high.
        
        Args:
            text: User query text
            
        Returns:
            Specificity level: "low", "med", or "high"
            
        Examples:
            >>> classifier = SpecificityClassifier({"specificity_high_anchors": 2})
            >>> classifier.classify("ABC-123 v2.1.0 issue")
            "high"
        """
        anchors = self._count_distinct_anchors(text)
        
        if anchors >= self.thresholds["specificity_high_anchors"]:
            specificity = "high"
        elif anchors >= 1:
            specificity = "med"
        else:
            specificity = "low"
        
        logger.debug(f"Specificity: {specificity} ({anchors} anchors in '{text[:30]}...')")
        
        return specificity
    
    def _count_distinct_anchors(self, text: str) -> int:
        """Count distinct anchoring patterns (deduplication prevents inflation)."""
        from .patterns import count_distinct_matches
        
        anchor_patterns = [
            "project_key",  # ABC-123
            "version",      # v1.2.3
            "env_tag",      # prod, staging
            "uuid",         # full UUIDs
            "id_pattern"    # id=something
        ]
        
        return count_distinct_matches(text, anchor_patterns)