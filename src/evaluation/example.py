#!/usr/bin/env python3
"""Example usage of the search evaluation tools."""

import logging
from datetime import datetime

from .rank_evaluator import RankEvaluator, EvaluationDataset
from .evaluation_client import EvaluationQuery
from .metrics import calculate_metrics_summary

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_example_dataset() -> EvaluationDataset:
    """Create an example evaluation dataset."""

    # Define example queries with relevance judgments
    example_queries = [
        EvaluationQuery(
            query_id="q1",
            query_text="OpenSearch configuration tutorial",
            relevant_docs=[
                {"_id": "doc_opensearch_config_1", "rating": 3},
                {"_id": "doc_opensearch_setup_2", "rating": 2},
                {"_id": "doc_elasticsearch_config_3", "rating": 1},
            ],
        ),
        EvaluationQuery(
            query_id="q2",
            query_text="Python REST API authentication",
            relevant_docs=[
                {"_id": "doc_python_auth_1", "rating": 3},
                {"_id": "doc_rest_security_2", "rating": 2},
                {"_id": "doc_api_tokens_3", "rating": 2},
                {"_id": "doc_oauth_python_4", "rating": 1},
            ],
        ),
        EvaluationQuery(
            query_id="q3",
            query_text="database connection pooling best practices",
            relevant_docs=[
                {"_id": "doc_db_pooling_1", "rating": 3},
                {"_id": "doc_connection_mgmt_2", "rating": 2},
                {"_id": "doc_performance_db_3", "rating": 1},
            ],
        ),
        EvaluationQuery(
            query_id="q4",
            query_text="error handling patterns microservices",
            relevant_docs=[
                {"_id": "doc_error_patterns_1", "rating": 3},
                {"_id": "doc_microservice_errors_2", "rating": 3},
                {"_id": "doc_resilience_patterns_3", "rating": 2},
                {"_id": "doc_circuit_breaker_4", "rating": 1},
            ],
        ),
        EvaluationQuery(
            query_id="q5",
            query_text="Docker container optimization",
            relevant_docs=[
                {"_id": "doc_docker_optimize_1", "rating": 3},
                {"_id": "doc_container_perf_2", "rating": 2},
                {"_id": "doc_dockerfile_best_3", "rating": 2},
            ],
        ),
    ]

    return EvaluationDataset(
        name="example_tech_docs_dataset",
        description="Example dataset for evaluating technical documentation search",
        queries=example_queries,
        created_at=datetime.now(),
        version="1.0",
    )


def example_basic_evaluation():
    """Example of basic evaluation workflow."""
    print("\n" + "=" * 60)
    print("BASIC EVALUATION EXAMPLE")
    print("=" * 60)

    # Create example dataset
    dataset = create_example_dataset()
    print(f"Created dataset '{dataset.name}' with {len(dataset.queries)} queries")

    # Initialize evaluator
    evaluator = RankEvaluator()

    # Save dataset for later use
    dataset_path = "example_dataset.json"
    evaluator.save_dataset(dataset, dataset_path)
    print(f"Dataset saved to {dataset_path}")

    # Note: In a real scenario, you would run evaluation against actual search results
    print("\n⚠️  Note: This example creates the dataset structure.")
    print("   To run actual evaluation, you need:")
    print("   1. A running OpenSearch instance")
    print("   2. Indexed documents matching the relevance judgments")
    print("   3. Configured search strategies")

    # Example of what the evaluation call would look like:
    print("\n📝 Example evaluation call:")
    print("   results = evaluator.evaluate_search_strategies(")
    print("       dataset=dataset,")
    print("       strategies=['enhanced_rrf', 'bm25'],")
    print("       index='your_index_name'")
    print("   )")


def example_metrics_calculation():
    """Example of standalone metrics calculation."""
    print("\n" + "=" * 60)
    print("METRICS CALCULATION EXAMPLE")
    print("=" * 60)

    # Example retrieved documents (in order of retrieval)
    retrieved_docs = [
        "doc_opensearch_config_1",  # Highly relevant (rating 3)
        "doc_irrelevant_1",  # Not relevant (not in judgments)
        "doc_opensearch_setup_2",  # Moderately relevant (rating 2)
        "doc_irrelevant_2",  # Not relevant
        "doc_elasticsearch_config_3",  # Slightly relevant (rating 1)
        "doc_irrelevant_3",  # Not relevant
        "doc_irrelevant_4",  # Not relevant
        "doc_irrelevant_5",  # Not relevant
        "doc_irrelevant_6",  # Not relevant
        "doc_irrelevant_7",  # Not relevant
    ]

    # Relevance judgments
    relevance_judgments = {
        "doc_opensearch_config_1": 3,
        "doc_opensearch_setup_2": 2,
        "doc_elasticsearch_config_3": 1,
    }

    # Calculate comprehensive metrics
    metrics = calculate_metrics_summary(retrieved_docs, relevance_judgments)

    print("\nCalculated metrics for example query:")
    print("Query: 'OpenSearch configuration tutorial'")
    print(f"Retrieved docs: {len(retrieved_docs)}")
    print(f"Relevant docs: {len(relevance_judgments)}")
    print("\nMetrics:")

    key_metrics = [
        "ndcg_at_10",
        "recall_at_10",
        "precision_at_10",
        "mean_reciprocal_rank",
        "average_precision",
    ]

    for metric in key_metrics:
        if metric in metrics:
            print(f"  {metric:20}: {metrics[metric]:.3f}")

    # Show interpretation
    print("\nInterpretation:")
    print(
        f"  • nDCG@10 = {metrics.get('ndcg_at_10', 0):.3f} - Measures ranking quality with position discount"
    )
    print(
        f"  • Recall@10 = {metrics.get('recall_at_10', 0):.3f} - Fraction of relevant docs found in top 10"
    )
    print(
        f"  • Precision@10 = {metrics.get('precision_at_10', 0):.3f} - Fraction of top 10 that are relevant"
    )
    print(
        f"  • MRR = {metrics.get('mean_reciprocal_rank', 0):.3f} - Reciprocal rank of first relevant doc"
    )


def example_strategy_comparison():
    """Example of comparing multiple search strategies."""
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON EXAMPLE")
    print("=" * 60)

    # Simulate results from different strategies
    strategy_results = {
        "enhanced_rrf": {
            "query_metrics": [
                {"ndcg_at_10": 0.85, "recall_at_50": 0.90, "precision_at_10": 0.70},
                {"ndcg_at_10": 0.78, "recall_at_50": 0.85, "precision_at_10": 0.60},
                {"ndcg_at_10": 0.92, "recall_at_50": 0.95, "precision_at_10": 0.80},
                {"ndcg_at_10": 0.88, "recall_at_50": 0.88, "precision_at_10": 0.75},
                {"ndcg_at_10": 0.82, "recall_at_50": 0.92, "precision_at_10": 0.65},
            ]
        },
        "bm25": {
            "query_metrics": [
                {"ndcg_at_10": 0.75, "recall_at_50": 0.80, "precision_at_10": 0.60},
                {"ndcg_at_10": 0.68, "recall_at_50": 0.75, "precision_at_10": 0.50},
                {"ndcg_at_10": 0.82, "recall_at_50": 0.85, "precision_at_10": 0.70},
                {"ndcg_at_10": 0.78, "recall_at_50": 0.78, "precision_at_10": 0.65},
                {"ndcg_at_10": 0.72, "recall_at_50": 0.82, "precision_at_10": 0.55},
            ]
        },
    }

    print("Comparing search strategies:")
    print("\nStrategy Performance:")

    for strategy, data in strategy_results.items():
        metrics_list = data["query_metrics"]

        # Calculate averages
        avg_ndcg = sum(m["ndcg_at_10"] for m in metrics_list) / len(metrics_list)
        avg_recall = sum(m["recall_at_50"] for m in metrics_list) / len(metrics_list)
        avg_precision = sum(m["precision_at_10"] for m in metrics_list) / len(
            metrics_list
        )

        print(f"\n{strategy.upper()}:")
        print(f"  Average nDCG@10:     {avg_ndcg:.3f}")
        print(f"  Average Recall@50:   {avg_recall:.3f}")
        print(f"  Average Precision@10: {avg_precision:.3f}")

    # Calculate improvements
    enhanced_rrf_ndcg = (
        sum(m["ndcg_at_10"] for m in strategy_results["enhanced_rrf"]["query_metrics"])
        / 5
    )
    bm25_ndcg = (
        sum(m["ndcg_at_10"] for m in strategy_results["bm25"]["query_metrics"]) / 5
    )

    improvement = ((enhanced_rrf_ndcg - bm25_ndcg) / bm25_ndcg) * 100

    print("\nIMPROVEMENT ANALYSIS:")
    print("  Enhanced RRF vs BM25:")
    print(f"  nDCG@10 improvement: {improvement:+.1f}%")

    if improvement > 5:
        print("  ✅ Significant improvement detected")
    elif improvement > 0:
        print("  ⚠️  Modest improvement")
    else:
        print("  ❌ No improvement or degradation")


def example_dataset_creation_from_logs():
    """Example of creating dataset from search logs."""
    print("\n" + "=" * 60)
    print("DATASET CREATION FROM LOGS EXAMPLE")
    print("=" * 60)

    # Example search log queries
    log_queries = [
        "how to install docker",
        "python virtual environment setup",
        "git merge conflict resolution",
        "kubernetes deployment yaml",
        "nginx load balancer configuration",
        "postgresql backup and restore",
        "redis caching strategies",
        "elasticsearch mapping configuration",
        "jenkins pipeline best practices",
        "terraform aws infrastructure",
    ]

    print(f"Example log queries ({len(log_queries)} total):")
    for i, query in enumerate(log_queries[:5], 1):
        print(f"  {i}. {query}")
    print(f"  ... and {len(log_queries) - 5} more")

    print("\n📝 To create dataset from actual logs:")
    print("   evaluator = RankEvaluator()")
    print("   dataset = evaluator.create_dataset_from_search_logs(")
    print("       log_queries=log_queries,")
    print("       relevance_threshold=0.7")
    print("   )")

    print("\n⚠️  Note: This requires:")
    print("   1. OpenSearch instance with indexed documents")
    print("   2. Documents that match the log queries")
    print(
        "   3. The system will auto-generate relevance judgments based on search scores"
    )


def example_continuous_monitoring():
    """Example of continuous evaluation monitoring."""
    print("\n" + "=" * 60)
    print("CONTINUOUS MONITORING EXAMPLE")
    print("=" * 60)

    print("Continuous evaluation monitors search quality over time:")
    print("\n📊 Key Features:")
    print("   • Automated evaluation runs")
    print("   • Performance threshold alerts")
    print("   • Historical trend tracking")
    print("   • Multi-strategy comparison")

    print("\n📝 Example usage:")
    print("   evaluator = RankEvaluator()")
    print("   summary = evaluator.run_continuous_evaluation(")
    print("       dataset=dataset,")
    print("       strategies=['enhanced_rrf', 'bm25'],")
    print("       baseline_threshold=0.8")
    print("   )")

    print("\n🚨 Alert Examples:")
    print("   ⚠️  enhanced_rrf: nDCG@10 (0.75) below threshold (0.80)")
    print("   ⚠️  bm25: Recall@50 (0.72) below threshold (0.80)")

    print("\n💡 Use Cases:")
    print("   • CI/CD pipeline integration")
    print("   • Production monitoring")
    print("   • A/B testing validation")
    print("   • Performance regression detection")


def main():
    """Run all examples."""
    print("🔍 SEARCH EVALUATION TOOLS - EXAMPLES")
    print("This script demonstrates the key features of the evaluation system.")

    try:
        example_basic_evaluation()
        example_metrics_calculation()
        example_strategy_comparison()
        example_dataset_creation_from_logs()
        example_continuous_monitoring()

        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("\n1. 📋 Create your evaluation dataset:")
        print("   python -m src.evaluation.cli create-sample --output my_dataset.json")

        print("\n2. 🧪 Run evaluation (requires OpenSearch + indexed docs):")
        print("   python -m src.evaluation.cli evaluate --dataset my_dataset.json")

        print("\n3. 📊 Set up continuous monitoring:")
        print("   python -m src.evaluation.cli continuous --dataset my_dataset.json")

        print("\n4. 📚 Check the documentation for advanced usage")

        print("\n✅ Examples completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise


if __name__ == "__main__":
    main()
