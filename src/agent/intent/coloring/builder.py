"""ColorsBuilder orchestrator - coordinates all coloring calculators."""

import logging
from typing import Dict

from ..slotter import Colors, DEFAULT_COLORS
from .actionability import ActionabilityScorer
from .suite_affinity import SuiteAffinityCalculator
from .specificity import SpecificityClassifier
from .safety import SafetyDetector

logger = logging.getLogger(__name__)


class ColorsBuilder:
    """
    Orchestrates coloring calculation with dependency injection.
    
    Single responsibility: coordinate the 4 calculators and assemble Colors result.
    Follows SOLID principles with injected dependencies for testability.
    """
    
    def __init__(self, registry: dict, weights: dict, thresholds: dict):
        """
        Initialize with injected dependencies.
        
        Args:
            registry: Suite registry from YAML manifests
            weights: COLORING_WEIGHTS from config
            thresholds: COLORING_THRESHOLDS from config
        """
        self.weights = weights
        self.thresholds = thresholds
        
        # Initialize calculators with injected dependencies
        self._actionability = ActionabilityScorer(weights)
        self._suite_affinity = SuiteAffinityCalculator(registry, weights)
        self._specificity = SpecificityClassifier(thresholds)
        self._safety = SafetyDetector(thresholds)
    
    def compute(self, text: str, features: Dict[str, int]) -> Colors:
        """
        Compute all coloring attributes for the given text.
        
        Args:
            text: User query text
            features: Binary features from main slotter
            
        Returns:
            Colors object with all attributes computed
            
        Performance target: < 2ms total
        """
        if not text or not text.strip():
            return DEFAULT_COLORS
            
        try:
            # Compute each color dimension (all pure functions)
            actionability_est, artifact_types = self._actionability.score(text, features)
            suite_affinity = self._suite_affinity.calculate(text)
            specificity, anchor_count = self._specificity.classify(text)
            safety_flags = self._safety.detect(text)
            
            # Time urgency (simple keyword-based)
            time_urgency = self._classify_time_urgency(text)
            
            # Troubleshoot flag
            troubleshoot_flag = self._has_troubleshoot_indicators(text)
            
            colors = Colors(
                actionability_est=actionability_est,
                suite_affinity=suite_affinity,
                artifact_types=artifact_types,
                specificity=specificity,
                troubleshoot_flag=troubleshoot_flag,
                time_urgency=time_urgency,
                safety_flags=safety_flags,
            )
            
            logger.debug(f\"Colors computed: act={actionability_est:.1f}, suites={len(suite_affinity)}, spec={specificity}\")
            return colors
            
        except Exception as e:
            logger.error(f\"Colors computation failed: {e}\")
            return DEFAULT_COLORS
    
    def _classify_time_urgency(self, text: str) -> str:
        \"\"\"Classify time urgency from keyword patterns.\"\"\"
        from .patterns import has_pattern_match
        
        if has_pattern_match(text, \"urgency_hard\"):
            return \"hard\"
        elif has_pattern_match(text, \"urgency_soft\"):
            return \"soft\"
        else:
            return \"none\"
    
    def _has_troubleshoot_indicators(self, text: str) -> bool:
        \"\"\"Check for troubleshooting indicators.\"\"\"
        from .patterns import has_pattern_match
        
        return (
            has_pattern_match(text, \"error_codes\") or
            has_pattern_match(text, \"stack_trace\")
        )


# Module-level singleton management
_colors_builder_instance = None


def get_colors_builder() -> ColorsBuilder:
    \"\"\"Get singleton ColorsBuilder instance with loaded configuration.\"\"\"
    global _colors_builder_instance
    
    if _colors_builder_instance is None:
        # Load dependencies
        from src.retrieval.config import COLORING_WEIGHTS, COLORING_THRESHOLDS
        
        # Load suite registry (reuse existing YAML loader)
        try:
            from src.retrieval.suites.registry import get_suite_registry
            registry = get_suite_registry()
        except ImportError:
            logger.warning(\"Suite registry not available, using empty registry\")
            registry = {}
        
        _colors_builder_instance = ColorsBuilder(
            registry=registry,
            weights=COLORING_WEIGHTS, 
            thresholds=COLORING_THRESHOLDS
        )
        
        logger.info(\"ColorsBuilder singleton initialized\")
    
    return _colors_builder_instance


def clear_colors_builder_cache():
    \"\"\"Clear singleton for testing/configuration reload.\"\"\"
    global _colors_builder_instance
    _colors_builder_instance = None