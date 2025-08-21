"""ColorsBuilder orchestrator - coordinates all coloring calculators."""

import logging
from typing import Dict, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Colors:
    """
    Nuanced coloring attributes for retrieval steering.
    
    Provides rich context beyond simple intent classification to enable
    sophisticated retrieval tuning and presentation decisions.
    """
    actionability_est: float  # [0,3] procedure likelihood
    suite_affinity: Dict[str, float]  # {"jira":0.9, "api":0.7}
    artifact_types: Set[str]  # {"endpoint", "ticket", "form"}
    specificity: str  # "low|med|high" - query anchoring
    troubleshoot_flag: bool  # error/debugging context
    time_urgency: str  # "none|soft|hard" - SLA requirements
    safety_flags: Set[str]  # {"pii", "cred", "secrets"}


class ColorsBuilder:
    """
    Orchestrates all coloring calculators with dependency injection.
    
    Coordinates actionability scoring, suite affinity calculation, 
    specificity detection, and safety analysis using SOLID principles.
    """
    
    def __init__(self, registry: dict, weights: dict, thresholds: dict):
        """
        Initialize with injected dependencies.
        
        Args:
            registry: Suite registry for affinity calculation
            weights: Scoring weights for all calculators
            thresholds: Detection thresholds for various systems
        """
        from .actionability import ActionabilityScorer
        from .suite_affinity import SuiteAffinityCalculator
        from .specificity import SpecificityClassifier
        from .safety import SafetyDetector
        
        self.actionability = ActionabilityScorer(weights)
        self.suite_affinity = SuiteAffinityCalculator(registry, weights)
        self.specificity = SpecificityClassifier(thresholds)
        self.safety = SafetyDetector(thresholds)
    
    def compute(self, text: str, features: Dict[str, int]) -> Colors:
        """
        Compute complete Colors object from text and binary features.
        
        Args:
            text: User query text
            features: Binary features from main slotter
            
        Returns:
            Colors object with all computed attributes
        """
        try:
            # Compute all coloring dimensions
            actionability_est, artifact_types = self.actionability.score(text, features)
            suite_affinity = self.suite_affinity.calculate(text)
            specificity = self.specificity.classify(text)
            safety_flags = self.safety.detect(text)
            
            # Classify additional attributes
            time_urgency = self._classify_urgency(text)
            troubleshoot_flag = self._detect_troubleshooting(text)
            
            logger.debug(f"Colors computed: act={actionability_est:.1f}, suites={len(suite_affinity)}, spec={specificity}")
            
            return Colors(
                actionability_est=actionability_est,
                suite_affinity=suite_affinity,
                artifact_types=artifact_types,
                specificity=specificity,
                troubleshoot_flag=troubleshoot_flag,
                time_urgency=time_urgency,
                safety_flags=safety_flags
            )
        except Exception as e:
            logger.error(f"Colors computation failed: {e}")
            return DEFAULT_COLORS
    
    def _classify_urgency(self, text: str) -> str:
        """Classify time urgency from keyword patterns."""
        from .patterns import has_pattern_match
        
        if has_pattern_match(text, "urgency_hard"):
            return "hard"
        elif has_pattern_match(text, "urgency_soft"):
            return "soft"
        else:
            return "none"
    
    def _detect_troubleshooting(self, text: str) -> bool:
        """Check for troubleshooting indicators."""
        from .patterns import has_pattern_match
        
        return (
            has_pattern_match(text, "error_codes") or
            has_pattern_match(text, "stack_trace")
        )


# Global singleton instance
_builder_instance = None


def get_colors_builder() -> ColorsBuilder:
    """Get singleton ColorsBuilder instance with loaded configuration."""
    global _builder_instance
    
    if _builder_instance is None:
        # Load configuration from centralized locations
        try:
            from src.retrieval.config import COLORING_WEIGHTS, COLORING_THRESHOLDS
            weights = COLORING_WEIGHTS
            thresholds = COLORING_THRESHOLDS
        except ImportError:
            logger.warning("Retrieval config not available, using defaults")
            weights = {"api_endpoint": 1.0, "jira_ticket": 1.0, "form_pipeline": 0.5}
            thresholds = {"specificity_high": 2, "cred_min_token_length": 16}
        
        try:
            from src.retrieval.suites.registry import get_suite_registry
            registry = get_suite_registry()
        except ImportError:
            logger.warning("Suite registry not available, using empty registry")
            registry = {}
        
        _builder_instance = ColorsBuilder(
            registry=registry,
            weights=weights,
            thresholds=thresholds
        )
        
        logger.info("ColorsBuilder singleton initialized")
    
    return _builder_instance


def clear_colors_builder():
    """Clear singleton for testing/configuration reload."""
    global _builder_instance
    _builder_instance = None


# Default fallback colors
DEFAULT_COLORS = Colors(
    actionability_est=0.5,
    suite_affinity={},
    artifact_types=set(),
    specificity="med",
    troubleshoot_flag=False,
    time_urgency="none",
    safety_flags=set()
)