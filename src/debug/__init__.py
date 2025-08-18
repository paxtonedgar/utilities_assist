"""Debug module for OpenSearch query analysis and performance monitoring.

This module provides tools for:
- Query explanation and scoring analysis using OpenSearch Explain API
- Performance profiling using OpenSearch Profile API
- Query optimization recommendations
- Search debugging utilities
"""

from .explain_analyzer import ExplainAnalyzer, ExplanationResult, ScoreBreakdown
from .profile_analyzer import ProfileAnalyzer, ProfileResult, QueryProfile
from .debug_client import OpenSearchDebugClient
from .optimization_advisor import OptimizationAdvisor, OptimizationRecommendation

__all__ = [
    'ExplainAnalyzer',
    'ExplanationResult', 
    'ScoreBreakdown',
    'ProfileAnalyzer',
    'ProfileResult',
    'QueryProfile',
    'OpenSearchDebugClient',
    'OptimizationAdvisor',
    'OptimizationRecommendation'
]