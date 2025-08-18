"""Evaluation metrics calculation utilities."""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RelevanceJudgment:
    """Represents a relevance judgment for a document."""
    doc_id: str
    rating: int  # 0-3 scale (0=not relevant, 3=highly relevant)
    

def calculate_ndcg(retrieved_docs: List[str], relevance_judgments: Dict[str, int], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain (nDCG) at k.
    
    Args:
        retrieved_docs: List of document IDs in retrieval order
        relevance_judgments: Dictionary mapping doc_id to relevance rating (0-3)
        k: Cut-off rank for evaluation
        
    Returns:
        nDCG@k score (0.0 to 1.0)
    """
    if not retrieved_docs or not relevance_judgments:
        return 0.0
    
    # Calculate DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevance_judgments:
            relevance = relevance_judgments[doc_id]
            # DCG formula: (2^rel - 1) / log2(rank + 1)
            dcg += (2 ** relevance - 1) / math.log2(i + 2)
    
    # Calculate IDCG@k (Ideal DCG)
    # Sort relevance judgments by rating in descending order
    sorted_relevance = sorted(relevance_judgments.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(sorted_relevance[:k]):
        idcg += (2 ** relevance - 1) / math.log2(i + 2)
    
    # Return nDCG
    return dcg / idcg if idcg > 0 else 0.0


def calculate_recall(retrieved_docs: List[str], relevance_judgments: Dict[str, int], k: int = 50) -> float:
    """Calculate Recall at k.
    
    Args:
        retrieved_docs: List of document IDs in retrieval order
        relevance_judgments: Dictionary mapping doc_id to relevance rating
        k: Cut-off rank for evaluation
        
    Returns:
        Recall@k score (0.0 to 1.0)
    """
    if not relevance_judgments:
        return 0.0
    
    # Count relevant documents in top-k results
    relevant_retrieved = 0
    for doc_id in retrieved_docs[:k]:
        if doc_id in relevance_judgments and relevance_judgments[doc_id] > 0:
            relevant_retrieved += 1
    
    # Total number of relevant documents
    total_relevant = sum(1 for rating in relevance_judgments.values() if rating > 0)
    
    return relevant_retrieved / total_relevant if total_relevant > 0 else 0.0


def calculate_precision(retrieved_docs: List[str], relevance_judgments: Dict[str, int], k: int = 10) -> float:
    """Calculate Precision at k.
    
    Args:
        retrieved_docs: List of document IDs in retrieval order
        relevance_judgments: Dictionary mapping doc_id to relevance rating
        k: Cut-off rank for evaluation
        
    Returns:
        Precision@k score (0.0 to 1.0)
    """
    if not retrieved_docs:
        return 0.0
    
    # Count relevant documents in top-k results
    relevant_retrieved = 0
    for doc_id in retrieved_docs[:k]:
        if doc_id in relevance_judgments and relevance_judgments[doc_id] > 0:
            relevant_retrieved += 1
    
    return relevant_retrieved / min(len(retrieved_docs), k)


def calculate_mean_reciprocal_rank(retrieved_docs: List[str], relevance_judgments: Dict[str, int]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved_docs: List of document IDs in retrieval order
        relevance_judgments: Dictionary mapping doc_id to relevance rating
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevance_judgments and relevance_judgments[doc_id] > 0:
            return 1.0 / (i + 1)
    
    return 0.0


def calculate_average_precision(retrieved_docs: List[str], relevance_judgments: Dict[str, int]) -> float:
    """Calculate Average Precision (AP).
    
    Args:
        retrieved_docs: List of document IDs in retrieval order
        relevance_judgments: Dictionary mapping doc_id to relevance rating
        
    Returns:
        AP score (0.0 to 1.0)
    """
    if not relevance_judgments:
        return 0.0
    
    total_relevant = sum(1 for rating in relevance_judgments.values() if rating > 0)
    if total_relevant == 0:
        return 0.0
    
    precision_sum = 0.0
    relevant_found = 0
    
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevance_judgments and relevance_judgments[doc_id] > 0:
            relevant_found += 1
            precision_at_i = relevant_found / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / total_relevant


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision score
        recall: Recall score
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def calculate_dcg(retrieved_docs: List[str], relevance_judgments: Dict[str, int], k: int = 10) -> float:
    """Calculate Discounted Cumulative Gain (DCG) at k.
    
    Args:
        retrieved_docs: List of document IDs in retrieval order
        relevance_judgments: Dictionary mapping doc_id to relevance rating
        k: Cut-off rank for evaluation
        
    Returns:
        DCG@k score
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_docs[:k]):
        if doc_id in relevance_judgments:
            relevance = relevance_judgments[doc_id]
            dcg += (2 ** relevance - 1) / math.log2(i + 2)
    
    return dcg


def calculate_ideal_dcg(relevance_judgments: Dict[str, int], k: int = 10) -> float:
    """Calculate Ideal Discounted Cumulative Gain (IDCG) at k.
    
    Args:
        relevance_judgments: Dictionary mapping doc_id to relevance rating
        k: Cut-off rank for evaluation
        
    Returns:
        IDCG@k score
    """
    sorted_relevance = sorted(relevance_judgments.values(), reverse=True)
    idcg = 0.0
    for i, relevance in enumerate(sorted_relevance[:k]):
        idcg += (2 ** relevance - 1) / math.log2(i + 2)
    
    return idcg


def calculate_rank_biased_precision(retrieved_docs: List[str], relevance_judgments: Dict[str, int], p: float = 0.8) -> float:
    """Calculate Rank-Biased Precision (RBP).
    
    Args:
        retrieved_docs: List of document IDs in retrieval order
        relevance_judgments: Dictionary mapping doc_id to relevance rating
        p: Persistence parameter (0 < p < 1)
        
    Returns:
        RBP score
    """
    rbp = 0.0
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevance_judgments:
            # Normalize relevance to 0-1 scale
            relevance = relevance_judgments[doc_id] / 3.0
            rbp += relevance * (p ** i)
    
    return rbp * (1 - p)


def calculate_expected_reciprocal_rank(retrieved_docs: List[str], relevance_judgments: Dict[str, int], max_grade: int = 3) -> float:
    """Calculate Expected Reciprocal Rank (ERR).
    
    Args:
        retrieved_docs: List of document IDs in retrieval order
        relevance_judgments: Dictionary mapping doc_id to relevance rating
        max_grade: Maximum relevance grade
        
    Returns:
        ERR score
    """
    err = 0.0
    prob_stop = 1.0
    
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevance_judgments:
            relevance = relevance_judgments[doc_id]
            # Probability of satisfaction
            prob_sat = (2 ** relevance - 1) / (2 ** max_grade)
            
            err += prob_stop * prob_sat / (i + 1)
            prob_stop *= (1 - prob_sat)
            
            if prob_stop <= 0.001:  # Early stopping
                break
    
    return err


def calculate_metrics_summary(
    retrieved_docs: List[str],
    relevance_judgments: Dict[str, int],
    k_values: List[int] = None
) -> Dict[str, float]:
    """Calculate a comprehensive set of evaluation metrics.
    
    Args:
        retrieved_docs: List of document IDs in retrieval order
        relevance_judgments: Dictionary mapping doc_id to relevance rating
        k_values: List of k values for @k metrics
        
    Returns:
        Dictionary of metric names to scores
    """
    if k_values is None:
        k_values = [1, 5, 10, 20, 50]
    
    metrics = {}
    
    # Calculate @k metrics for each k value
    for k in k_values:
        metrics[f"ndcg_at_{k}"] = calculate_ndcg(retrieved_docs, relevance_judgments, k)
        metrics[f"recall_at_{k}"] = calculate_recall(retrieved_docs, relevance_judgments, k)
        metrics[f"precision_at_{k}"] = calculate_precision(retrieved_docs, relevance_judgments, k)
        
        # F1 score at k
        precision_k = metrics[f"precision_at_{k}"]
        recall_k = metrics[f"recall_at_{k}"]
        metrics[f"f1_at_{k}"] = calculate_f1_score(precision_k, recall_k)
    
    # Single-value metrics
    metrics["mean_reciprocal_rank"] = calculate_mean_reciprocal_rank(retrieved_docs, relevance_judgments)
    metrics["average_precision"] = calculate_average_precision(retrieved_docs, relevance_judgments)
    metrics["rank_biased_precision"] = calculate_rank_biased_precision(retrieved_docs, relevance_judgments)
    metrics["expected_reciprocal_rank"] = calculate_expected_reciprocal_rank(retrieved_docs, relevance_judgments)
    
    return metrics


def aggregate_metrics(query_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across multiple queries.
    
    Args:
        query_metrics: List of metric dictionaries for each query
        
    Returns:
        Dictionary of aggregated metrics (averages)
    """
    if not query_metrics:
        return {}
    
    # Get all metric names
    all_metrics = set()
    for metrics in query_metrics:
        all_metrics.update(metrics.keys())
    
    # Calculate averages
    aggregated = {}
    for metric in all_metrics:
        values = [metrics.get(metric, 0.0) for metrics in query_metrics]
        aggregated[f"avg_{metric}"] = sum(values) / len(values)
        aggregated[f"min_{metric}"] = min(values)
        aggregated[f"max_{metric}"] = max(values)
        
        # Calculate standard deviation
        mean_val = aggregated[f"avg_{metric}"]
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        aggregated[f"std_{metric}"] = math.sqrt(variance)
    
    return aggregated


def statistical_significance_test(
    metrics_a: List[float],
    metrics_b: List[float],
    alpha: float = 0.05
) -> Tuple[bool, float]:
    """Perform paired t-test for statistical significance.
    
    Args:
        metrics_a: Metric values for system A
        metrics_b: Metric values for system B
        alpha: Significance level
        
    Returns:
        Tuple of (is_significant, p_value)
    """
    if len(metrics_a) != len(metrics_b) or len(metrics_a) < 2:
        return False, 1.0
    
    # Calculate differences
    differences = [a - b for a, b in zip(metrics_a, metrics_b)]
    n = len(differences)
    
    # Calculate mean and standard deviation of differences
    mean_diff = sum(differences) / n
    variance = sum((d - mean_diff) ** 2 for d in differences) / (n - 1)
    std_diff = math.sqrt(variance)
    
    if std_diff == 0:
        return False, 1.0
    
    # Calculate t-statistic
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    
    # Degrees of freedom
    df = n - 1
    
    # Approximate p-value calculation (simplified)
    # For a more accurate implementation, use scipy.stats.t.sf
    abs_t = abs(t_stat)
    if df >= 30:
        # Use normal approximation for large samples
        p_value = 2 * (1 - _normal_cdf(abs_t))
    else:
        # Simplified t-distribution approximation
        p_value = 2 * (1 - _t_cdf_approx(abs_t, df))
    
    return p_value < alpha, p_value


def _normal_cdf(x: float) -> float:
    """Approximate normal CDF using error function approximation."""
    return 0.5 * (1 + _erf(x / math.sqrt(2)))


def _erf(x: float) -> float:
    """Approximate error function."""
    # Abramowitz and Stegun approximation
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    
    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    
    return sign * y


def _t_cdf_approx(t: float, df: int) -> float:
    """Approximate t-distribution CDF."""
    # Simple approximation for small degrees of freedom
    if df >= 30:
        return _normal_cdf(t)
    
    # Use normal approximation with correction
    correction = 1 + (t * t) / (4 * df)
    return _normal_cdf(t / math.sqrt(correction))