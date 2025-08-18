"""Example usage of the search evaluation tools."""

import logging
from datetime import datetime
from typing import List, Dict, Any

from .rank_evaluator import RankEvaluator, EvaluationDataset
from .evaluation_client import EvaluationQuery
from .metrics import (
    calculate_ndcg, calculate_recall, calculate_precision, calculate_mrr,
    aggregate_metrics, statistical_significance_test
)
from src.infra.opensearch_client import SearchFilters

logger = logging.getLogger(__name__)


def create_example_dataset() -> EvaluationDataset:
    """Create an example evaluation dataset."""
    queries = [
        EvaluationQuery(
            query_id="q1",
            query_text="OpenSearch configuration",
            relevant_docs=[
                {"_id": "doc1", "rating": 3},
                {"_id": "doc2", "rating": 2},
                {"_id": "doc3", "rating": 1}
            ]
        ),
        EvaluationQuery(
            query_id="q2",
            query_text="API documentation search",
            relevant_docs=[
                {"_id": "doc4", "rating": 3},
                {"_id": "doc5", "rating": 2}
            ],
            filters=SearchFilters(content_type="api_spec")
        ),
        EvaluationQuery(
            query_id="q3",
            query_text="troubleshooting connection issues",
            relevant_docs=[
                {"_id": "doc6", "rating": 3},
                {"_id": "doc7", "rating": 3},
                {"_id": "doc8", "rating": 1}
            ]
        )
    ]
    
    return EvaluationDataset(
        name="example_dataset",
        description="Example dataset for demonstrating evaluation tools",
        queries=queries,
        created_at=datetime.now(),
        metadata={"source": "example", "version": "1.0"}
    )


def run_basic_evaluation():
    """Demonstrate basic evaluation workflow."""
    print("=== Basic Evaluation Example ===")
    
    # Create example dataset
    dataset = create_example_dataset()
    print(f"Created dataset with {len(dataset.queries)} queries")
    
    # Initialize evaluator
    evaluator = RankEvaluator()
    
    # Evaluate multiple strategies
    strategies = ["enhanced_rrf", "bm25", "knn"]
    print(f"\nEvaluating strategies: {strategies}")
    
    try:
        results = evaluator.evaluate_search_strategies(
            dataset=dataset,
            strategies=strategies,
            k=50
        )
        
        # Display results
        print("\nEvaluation Results:")
        for strategy, result in results.items():
            metrics = result.overall_metrics
            print(f"\n{strategy}:")
            print(f"  nDCG@10: {metrics.get('ndcg_at_10', 0):.3f}")
            print(f"  Recall@50: {metrics.get('recall_at_50', 0):.3f}")
            print(f"  Precision@10: {metrics.get('precision_at_10', 0):.3f}")
            print(f"  MRR: {metrics.get('mean_reciprocal_rank', 0):.3f}")
        
        # Compare strategies
        if len(results) > 1:
            comparison = evaluator.compare_strategies(results)
            print("\nStrategy Rankings:")
            for i, (strategy, metrics) in enumerate(comparison['rankings'], 1):
                print(f"{i}. {strategy}: {metrics.get('ndcg_at_10', 0):.3f}")
                
    except Exception as e:
        print(f"Evaluation failed: {e}")


def demonstrate_standalone_metrics():
    """Demonstrate standalone metric calculations."""
    print("\n=== Standalone Metrics Example ===")
    
    # Example relevance scores (in ranking order)
    relevance_scores = [3, 2, 0, 1, 3, 0, 2, 1, 0, 0]
    
    # Calculate various metrics
    ndcg_10 = calculate_ndcg(relevance_scores, k=10)
    print(f"nDCG@10: {ndcg_10:.3f}")
    
    # Binary relevance for precision/recall
    binary_scores = [1 if score > 0 else 0 for score in relevance_scores]
    relevant_retrieved = sum(binary_scores[:10])
    total_relevant = sum(binary_scores)
    
    recall = calculate_recall(relevant_retrieved, total_relevant)
    precision = calculate_precision(relevant_retrieved, 10)
    
    print(f"Recall@10: {recall:.3f}")
    print(f"Precision@10: {precision:.3f}")
    
    # MRR example
    first_relevant_positions = [1, 3, 2, 0, 1]  # 0 means no relevant found
    mrr = calculate_mrr(first_relevant_positions)
    print(f"MRR: {mrr:.3f}")
    
    # Aggregate metrics
    metric_values = [0.8, 0.7, 0.9, 0.6, 0.85]
    aggregated = aggregate_metrics(metric_values)
    print(f"\nAggregated metrics:")
    print(f"  Mean: {aggregated['mean']:.3f}")
    print(f"  Std: {aggregated['std']:.3f}")
    print(f"  Min/Max: {aggregated['min']:.3f}/{aggregated['max']:.3f}")


def demonstrate_strategy_comparison():
    """Demonstrate strategy comparison with statistical significance."""
    print("\n=== Strategy Comparison Example ===")
    
    # Simulate metric scores for two strategies
    baseline_scores = [0.7, 0.6, 0.8, 0.5, 0.9, 0.7, 0.6, 0.8]
    treatment_scores = [0.8, 0.7, 0.85, 0.6, 0.95, 0.75, 0.65, 0.82]
    
    # Statistical significance test
    p_value = statistical_significance_test(baseline_scores, treatment_scores)
    
    baseline_mean = sum(baseline_scores) / len(baseline_scores)
    treatment_mean = sum(treatment_scores) / len(treatment_scores)
    improvement = treatment_mean - baseline_mean
    
    print(f"Baseline mean: {baseline_mean:.3f}")
    print(f"Treatment mean: {treatment_mean:.3f}")
    print(f"Improvement: {improvement:.3f}")
    print(f"P-value: {p_value:.3f}")
    print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")


def demonstrate_dataset_from_logs():
    """Demonstrate creating dataset from search logs."""
    print("\n=== Dataset from Logs Example ===")
    
    # Example search logs
    search_logs = [
        {
            "query": "opensearch configuration",
            "clicked_documents": ["doc1", "doc2"],
            "filters": {"content_type": "confluence"}
        },
        {
            "query": "opensearch configuration",
            "clicked_documents": ["doc1"],
            "filters": {"content_type": "confluence"}
        },
        {
            "query": "api documentation",
            "clicked_documents": ["doc3", "doc4", "doc3"],
            "filters": {"content_type": "api_spec"}
        },
        {
            "query": "api documentation",
            "clicked_documents": ["doc3"],
            "filters": {"content_type": "api_spec"}
        }
    ]
    
    # Create dataset from logs
    evaluator = RankEvaluator()
    dataset = evaluator.create_dataset_from_search_logs(
        search_logs=search_logs,
        name="log_derived_dataset",
        description="Dataset created from search logs",
        relevance_threshold=0.5,
        min_results_per_query=2
    )
    
    print(f"Created dataset with {len(dataset.queries)} queries")
    for query in dataset.queries:
        print(f"\nQuery: {query.query_text}")
        print(f"Relevant docs: {len(query.relevant_docs)}")
        for doc in query.relevant_docs:
            print(f"  {doc['_id']}: rating {doc['rating']}")


def demonstrate_continuous_monitoring():
    """Demonstrate continuous evaluation and monitoring."""
    print("\n=== Continuous Monitoring Example ===")
    
    # Create dataset
    dataset = create_example_dataset()
    evaluator = RankEvaluator()
    
    # Simulate baseline results (you would load these from previous evaluation)
    baseline_results = None  # In practice, load from file
    
    try:
        # Run continuous evaluation
        results = evaluator.run_continuous_evaluation(
            dataset=dataset,
            strategies=["enhanced_rrf", "bm25"],
            baseline_results=baseline_results,
            degradation_threshold=0.05
        )
        
        print(f"Evaluation completed at: {results['evaluation_timestamp']}")
        print(f"Degradation detected: {results['has_degradation']}")
        
        if results['alerts']:
            print("\nAlerts:")
            for alert in results['alerts']:
                print(f"  {alert['strategy']}: {alert['degradation']:.3f} degradation")
        else:
            print("No performance alerts")
            
    except Exception as e:
        print(f"Continuous evaluation failed: {e}")


def main():
    """Run all evaluation examples."""
    print("Search Evaluation Tools - Examples")
    print("=" * 50)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Run examples
        run_basic_evaluation()
        demonstrate_standalone_metrics()
        demonstrate_strategy_comparison()
        demonstrate_dataset_from_logs()
        demonstrate_continuous_monitoring()
        
        print("\n=== All Examples Completed ===")
        
    except Exception as e:
        print(f"Example execution failed: {e}")
        raise


if __name__ == "__main__":
    main()