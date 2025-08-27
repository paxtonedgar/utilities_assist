"""
Content Quality Analyzer for Multi-Index Search

Provides tools to analyze and monitor search result quality across different indices,
particularly for debugging score calibration issues between confluence and swagger indices.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass, field
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Statistics for a specific index."""
    index_name: str
    total_results: int = 0
    score_distribution: List[float] = field(default_factory=list)
    avg_score: float = 0.0
    median_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    above_threshold_count: int = 0
    threshold_used: float = 0.0


@dataclass
class QualityAnalysis:
    """Complete quality analysis across all indices."""
    by_index: Dict[str, IndexStats] = field(default_factory=dict)
    total_results: int = 0
    quality_filtered_results: int = 0
    calibration_applied: bool = False
    score_balance_ratio: float = 0.0  # confluence_avg / swagger_avg


class ContentQualityAnalyzer:
    """Analyzer for content quality and score calibration issues."""
    
    def __init__(self):
        self.analyses: List[QualityAnalysis] = []
    
    def analyze_search_results(self, results: List[Any], query: str = "") -> QualityAnalysis:
        """
        Analyze search results for quality patterns and score distribution.
        
        Args:
            results: List of search results with .score and .meta attributes
            query: Original search query for context
            
        Returns:
            QualityAnalysis with comprehensive statistics
        """
        analysis = QualityAnalysis()
        analysis.total_results = len(results)
        
        # Group results by index
        results_by_index = defaultdict(list)
        for result in results:
            index_name = result.meta.get("source_index", "unknown")
            results_by_index[index_name].append(result)
        
        # Analyze each index
        confluence_avg = swagger_avg = 0.0
        for index_name, index_results in results_by_index.items():
            stats = self._analyze_index_results(index_name, index_results)
            analysis.by_index[index_name] = stats
            
            if "swagger" in index_name:
                swagger_avg = stats.avg_score
            else:
                confluence_avg = stats.avg_score
        
        # Calculate score balance ratio
        if swagger_avg > 0:
            analysis.score_balance_ratio = confluence_avg / swagger_avg
        
        # Log analysis summary
        self._log_analysis_summary(analysis, query)
        
        # Store for trend analysis
        self.analyses.append(analysis)
        
        return analysis
    
    def _analyze_index_results(self, index_name: str, results: List[Any]) -> IndexStats:
        """Analyze results from a specific index."""
        stats = IndexStats(index_name=index_name)
        
        if not results:
            return stats
        
        stats.total_results = len(results)
        scores = [getattr(r, 'score', 0.0) for r in results]
        stats.score_distribution = scores
        
        if scores:
            stats.avg_score = statistics.mean(scores)
            stats.median_score = statistics.median(scores)
            stats.min_score = min(scores)
            stats.max_score = max(scores)
            
            # Determine appropriate threshold
            if "swagger" in index_name:
                stats.threshold_used = 0.05
            else:
                stats.threshold_used = 0.15
            
            stats.above_threshold_count = sum(1 for s in scores if s >= stats.threshold_used)
        
        return stats
    
    def _log_analysis_summary(self, analysis: QualityAnalysis, query: str = ""):
        """Log summary of quality analysis."""
        query_context = f" for query '{query[:50]}...'" if query else ""
        logger.info(f"Content Quality Analysis{query_context}:")
        logger.info(f"  Total results: {analysis.total_results}")
        
        for index_name, stats in analysis.by_index.items():
            retention_pct = (stats.above_threshold_count / stats.total_results * 100) if stats.total_results > 0 else 0
            logger.info(f"  {index_name}:")
            logger.info(f"    Results: {stats.total_results}, Avg score: {stats.avg_score:.3f}")
            logger.info(f"    Threshold: {stats.threshold_used}, Retained: {stats.above_threshold_count}/{stats.total_results} ({retention_pct:.1f}%)")
        
        if len(analysis.by_index) >= 2:
            logger.info(f"  Score balance ratio: {analysis.score_balance_ratio:.2f}x")
            if analysis.score_balance_ratio > 3.0:
                logger.warning("  ⚠️  High score imbalance detected - consider calibration")
    
    def get_trend_summary(self, last_n: int = 10) -> Dict[str, Any]:
        """Get trend summary across recent analyses."""
        if not self.analyses:
            return {}
        
        recent = self.analyses[-last_n:]
        
        # Calculate averages across recent analyses
        confluence_scores = []
        swagger_scores = []
        balance_ratios = []
        
        for analysis in recent:
            for index_name, stats in analysis.by_index.items():
                if "swagger" in index_name and stats.avg_score > 0:
                    swagger_scores.append(stats.avg_score)
                elif stats.avg_score > 0:
                    confluence_scores.append(stats.avg_score)
            
            if analysis.score_balance_ratio > 0:
                balance_ratios.append(analysis.score_balance_ratio)
        
        summary = {
            "analyses_count": len(recent),
            "confluence_avg_score": statistics.mean(confluence_scores) if confluence_scores else 0,
            "swagger_avg_score": statistics.mean(swagger_scores) if swagger_scores else 0,
            "avg_balance_ratio": statistics.mean(balance_ratios) if balance_ratios else 0,
            "calibration_recommended": statistics.mean(balance_ratios) > 3.0 if balance_ratios else False,
        }
        
        return summary
    
    def suggest_threshold_adjustments(self, target_retention_pct: float = 60.0) -> Dict[str, float]:
        """
        Suggest threshold adjustments based on recent score distributions.
        
        Args:
            target_retention_pct: Target percentage of results to retain per index
            
        Returns:
            Dictionary mapping index names to suggested thresholds
        """
        suggestions = {}
        
        if not self.analyses:
            return suggestions
        
        # Analyze recent score distributions
        recent_scores = defaultdict(list)
        for analysis in self.analyses[-5:]:  # Last 5 analyses
            for index_name, stats in analysis.by_index.items():
                recent_scores[index_name].extend(stats.score_distribution)
        
        # Calculate percentile-based thresholds
        for index_name, scores in recent_scores.items():
            if not scores:
                continue
            
            scores_sorted = sorted(scores, reverse=True)
            target_index = int(len(scores_sorted) * (target_retention_pct / 100))
            
            if target_index < len(scores_sorted):
                suggested_threshold = scores_sorted[target_index]
                suggestions[index_name] = max(suggested_threshold, 0.01)  # Minimum threshold
        
        return suggestions


# Global analyzer instance for use across the application
_quality_analyzer = ContentQualityAnalyzer()

def analyze_results(results: List[Any], query: str = "") -> QualityAnalysis:
    """Convenience function to analyze results with global analyzer."""
    return _quality_analyzer.analyze_search_results(results, query)

def get_quality_trends() -> Dict[str, Any]:
    """Get quality trend summary from global analyzer."""
    return _quality_analyzer.get_trend_summary()

def suggest_thresholds(target_retention_pct: float = 60.0) -> Dict[str, float]:
    """Get threshold suggestions from global analyzer."""
    return _quality_analyzer.suggest_threshold_adjustments(target_retention_pct)