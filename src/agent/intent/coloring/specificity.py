"""Specificity classifier - counts distinct anchors to avoid inflation."""

import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class SpecificityClassifier:
    """
    Single responsibility: classify query specificity based on distinct anchor count.
    
    Uses set-based deduplication to avoid multi-hit inflation:
    - Project keys, API paths, UUIDs, versions, env tags → distinct anchors
    - Map: {0: low, 1: med, ≥2: high}
    - Returns both level and anchor count for metrics
    """
    
    def __init__(self, thresholds: dict):
        """Initialize with thresholds from centralized config."""
        self.thresholds = thresholds
    
    def classify(self, text: str) -> Tuple[str, int]:
        """
        Classify specificity level based on distinct anchors.
        
        Args:
            text: User query text
            
        Returns:
            (specificity_level, distinct_anchor_count)
            
        Examples:
            >>> classifier = SpecificityClassifier({\"specificity_med_anchors\": 1, \"specificity_high_anchors\": 2})
            >>> level, count = classifier.classify(\"ABC-123 /api/v1/users\")
            >>> level == \"high\" and count >= 2
            True
        """
        anchors = self._count_distinct_anchors(text)
        
        # Classify based on thresholds
        if anchors >= self.thresholds[\"specificity_high_anchors\"]:
            level = \"high\"
        elif anchors >= self.thresholds[\"specificity_med_anchors\"]:
            level = \"med\"
        else:
            level = \"low\"
        
        logger.debug(f\"Specificity: {level} ({anchors} anchors) for '{text[:30]}...'\")
        
        return level, anchors
    
    def _count_distinct_anchors(self, text: str) -> int:
        \"\"\"Count distinct anchors using set-based deduplication.\"\"\"
        from .patterns import extract_pattern_matches
        
        # Use set to dedupe anchors across all patterns
        distinct_anchors = set()
        
        # Project keys (ABC-123)
        project_keys = extract_pattern_matches(text, \"project_key\")
        distinct_anchors.update(project_keys)
        
        # API paths (dedupe to first occurrence)
        api_paths = extract_pattern_matches(text, \"api_path\")
        if api_paths:
            distinct_anchors.add(\"api_path\")  # Count as single anchor type
        
        # Versions 
        versions = extract_pattern_matches(text, \"version\")
        distinct_anchors.update(versions)
        
        # Environment tags
        env_tags = extract_pattern_matches(text, \"env_tag\")
        distinct_anchors.update(env_tags)
        
        # UUIDs (very specific, count as 2 anchors)
        uuids = extract_pattern_matches(text, \"uuid\")
        for uuid in uuids:
            distinct_anchors.add(f\"uuid:{uuid}\")
            distinct_anchors.add(f\"uuid_bonus:{uuid}\")  # Bonus anchor for high specificity
        
        # ID patterns
        id_matches = extract_pattern_matches(text, \"id_pattern\")
        if id_matches:
            distinct_anchors.add(\"id_pattern\")
        
        return len(distinct_anchors)