"""Evaluation module for OpenSearch query analysis and performance monitoring."""

from .evaluation_client import OpenSearchEvaluationClient, EvaluationQuery, EvaluationMetrics, EvaluationResult
from .rank_evaluator import RankEvaluator, EvaluationDataset
from .metrics import (
    calculate_ndcg, calculate_recall, calculate_precision, calculate_mrr,
    calculate_average_precision, calculate_f1_score, calculate_dcg, calculate_idcg,
    calculate_rbp, calculate_err, aggregate_metrics, statistical_significance_test
)

__all__ = [
    'OpenSearchEvaluationClient',
    'EvaluationQuery',
    'EvaluationMetrics', 
    'EvaluationResult',
    'RankEvaluator',
    'EvaluationDataset',
    'calculate_ndcg',
    'calculate_recall',
    'calculate_precision',
    'calculate_mrr',
    'calculate_average_precision',
    'calculate_f1_score',
    'calculate_dcg',
    'calculate_idcg',
    'calculate_rbp',
    'calculate_err',
    'aggregate_metrics',
    'statistical_significance_test'
]