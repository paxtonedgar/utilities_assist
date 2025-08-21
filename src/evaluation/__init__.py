"""Evaluation module for search quality assessment.

Provides tools for evaluating search performance using:
- OpenSearch Rank Evaluation API with nDCG@10 and recall@50 metrics
- Custom evaluation metrics and benchmarking
- Performance monitoring and analysis
"""

from .rank_evaluator import RankEvaluator, EvaluationMetrics, EvaluationResult
from .evaluation_client import OpenSearchEvaluationClient
from .metrics import calculate_ndcg, calculate_recall, calculate_precision

__all__ = [
    "RankEvaluator",
    "EvaluationMetrics",
    "EvaluationResult",
    "OpenSearchEvaluationClient",
    "calculate_ndcg",
    "calculate_recall",
    "calculate_precision",
]
