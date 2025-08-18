"""Debug module for OpenSearch query analysis and performance monitoring.

This module provides tools for debugging OpenSearch queries using the Explain and Profile APIs,
analyzing query structure and performance, and monitoring search performance over time.
"""

from .debug_client import OpenSearchDebugClient, DebugQuery, DebugSession
from .query_analyzer import QueryAnalyzer, QueryComponent, QueryAnalysis
from .performance_monitor import PerformanceMonitor, PerformanceMetric, PerformanceAlert, PerformanceThreshold

__all__ = [
    'OpenSearchDebugClient',
    'DebugQuery',
    'DebugSession',
    'QueryAnalyzer',
    'QueryComponent',
    'QueryAnalysis',
    'PerformanceMonitor',
    'PerformanceMetric',
    'PerformanceAlert',
    'PerformanceThreshold'
]