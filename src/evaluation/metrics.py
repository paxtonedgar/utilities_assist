"""Standalone functions for calculating various evaluation metrics."""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from scipy import stats
import numpy as np

logger = logging.getLogger(__name__)


def calculate_dcg(relevance_scores: List[float], k: Optional[int] = None) -> float:
    """Calculate Discounted Cumulative Gain (DCG).
    
    Args:
        relevance_scores: List of relevance scores in ranking order
        k: Calculate DCG@k (if None, uses all scores)
        
    Returns:
        DCG score
    """
    if not relevance_scores:
        return 0.0
        
    scores = relevance_scores[:k] if k else relevance_scores
    dcg = scores[0] if scores else 0.0
    
    for i, score in enumerate(scores[1:], start=2):
        dcg += score / math.log2(i)
        
    return dcg


def calculate_idcg(relevance_scores: List[float], k: Optional[int] = None) -> float:
    """Calculate Ideal Discounted Cumulative Gain (IDCG).
    
    Args:
        relevance_scores: List of relevance scores
        k: Calculate IDCG@k (if None, uses all scores)
        
    Returns:
        IDCG score
    """
    if not relevance_scores:
        return 0.0
        
    # Sort in descending order for ideal ranking
    ideal_scores = sorted(relevance_scores, reverse=True)
    return calculate_dcg(ideal_scores, k)


def calculate_ndcg(relevance_scores: List[float], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain (nDCG).
    
    Args:
        relevance_scores: List of relevance scores in ranking order
        k: Calculate nDCG@k
        
    Returns:
        nDCG@k score between 0 and 1
    """
    if not relevance_scores:
        return 0.0
        
    dcg = calculate_dcg(relevance_scores, k)
    idcg = calculate_idcg(relevance_scores, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_recall(relevant_retrieved: int, total_relevant: int) -> float:
    """Calculate recall.
    
    Args:
        relevant_retrieved: Number of relevant documents retrieved
        total_relevant: Total number of relevant documents
        
    Returns:
        Recall score between 0 and 1
    """
    if total_relevant == 0:
        return 0.0
    return relevant_retrieved / total_relevant


def calculate_precision(relevant_retrieved: int, total_retrieved: int) -> float:
    """Calculate precision.
    
    Args:
        relevant_retrieved: Number of relevant documents retrieved
        total_retrieved: Total number of documents retrieved
        
    Returns:
        Precision score between 0 and 1
    """
    if total_retrieved == 0:
        return 0.0
    return relevant_retrieved / total_retrieved


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score.
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score between 0 and 1
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_average_precision(relevance_scores: List[float]) -> float:
    """Calculate Average Precision (AP).
    
    Args:
        relevance_scores: List of binary relevance scores (0 or 1) in ranking order
        
    Returns:
        Average Precision score
    """
    if not relevance_scores:
        return 0.0
        
    relevant_count = 0
    precision_sum = 0.0
    
    for i, score in enumerate(relevance_scores):
        if score > 0:  # Relevant document
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    total_relevant = sum(1 for score in relevance_scores if score > 0)
    return precision_sum / total_relevant if total_relevant > 0 else 0.0


def calculate_mrr(first_relevant_positions: List[int]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        first_relevant_positions: List of positions (1-indexed) of first relevant document for each query
                                 Use 0 if no relevant document found
        
    Returns:
        MRR score
    """
    if not first_relevant_positions:
        return 0.0
        
    reciprocal_ranks = [1.0 / pos if pos > 0 else 0.0 for pos in first_relevant_positions]
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_rbp(relevance_scores: List[float], p: float = 0.8) -> float:
    """Calculate Rank-Biased Precision (RBP).
    
    Args:
        relevance_scores: List of binary relevance scores (0 or 1) in ranking order
        p: Persistence parameter (probability user continues to next result)
        
    Returns:
        RBP score
    """
    if not relevance_scores:
        return 0.0
        
    rbp = 0.0
    for i, score in enumerate(relevance_scores):
        rbp += score * (p ** i)
        
    return (1 - p) * rbp


def calculate_err(relevance_scores: List[float], max_grade: float = 3.0) -> float:
    """Calculate Expected Reciprocal Rank (ERR).
    
    Args:
        relevance_scores: List of relevance scores in ranking order
        max_grade: Maximum relevance grade
        
    Returns:
        ERR score
    """
    if not relevance_scores:
        return 0.0
        
    err = 0.0
    prob_stop = 0.0
    
    for i, score in enumerate(relevance_scores):
        # Probability of satisfaction at this rank
        prob_sat = (2 ** score - 1) / (2 ** max_grade)
        
        # Probability of reaching this rank
        prob_reach = 1 - prob_stop
        
        # Contribution to ERR
        err += prob_reach * prob_sat / (i + 1)
        
        # Update probability of stopping
        prob_stop += prob_reach * prob_sat
        
        if prob_stop >= 1.0:
            break
            
    return err


def aggregate_metrics(metric_values: List[float]) -> Dict[str, float]:
    """Aggregate metric values across multiple queries.
    
    Args:
        metric_values: List of metric values from different queries
        
    Returns:
        Dictionary with aggregated statistics
    """
    if not metric_values:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'count': 0
        }
    
    values = np.array(metric_values)
    
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'count': len(values)
    }


def statistical_significance_test(
    baseline_scores: List[float],
    treatment_scores: List[float],
    test_type: str = 'paired_t'
) -> float:
    """Perform statistical significance test between two sets of scores.
    
    Args:
        baseline_scores: Baseline metric scores
        treatment_scores: Treatment metric scores
        test_type: Type of test ('paired_t', 'wilcoxon', 'mann_whitney')
        
    Returns:
        p-value from the statistical test
    """
    if len(baseline_scores) != len(treatment_scores):
        raise ValueError("Baseline and treatment scores must have same length")
        
    if len(baseline_scores) < 2:
        return 1.0  # Cannot perform test with insufficient data
    
    try:
        if test_type == 'paired_t':
            _, p_value = stats.ttest_rel(treatment_scores, baseline_scores)
        elif test_type == 'wilcoxon':
            _, p_value = stats.wilcoxon(treatment_scores, baseline_scores)
        elif test_type == 'mann_whitney':
            _, p_value = stats.mannwhitneyu(treatment_scores, baseline_scores, alternative='two-sided')
        else:
            raise ValueError(f"Unknown test type: {test_type}")
            
        return float(p_value)
        
    except Exception as e:
        logger.warning(f"Statistical test failed: {e}")
        return 1.0  # Return non-significant p-value on error