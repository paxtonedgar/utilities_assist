#!/usr/bin/env python3
"""Command-line interface for search evaluation tools."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .rank_evaluator import RankEvaluator, EvaluationDataset
from .evaluation_client import EvaluationQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_sample_dataset(output_path: str, num_queries: int = 10) -> None:
    """Create a sample evaluation dataset for testing.

    Args:
        output_path: Path to save the sample dataset
        num_queries: Number of sample queries to generate
    """
    sample_queries = [
        {
            "query_text": "How to configure OpenSearch cluster",
            "relevant_docs": [
                {"_id": "doc1", "rating": 3},
                {"_id": "doc2", "rating": 2},
            ],
        },
        {
            "query_text": "Python API documentation",
            "relevant_docs": [
                {"_id": "doc3", "rating": 3},
                {"_id": "doc4", "rating": 1},
            ],
        },
        {
            "query_text": "REST API authentication",
            "relevant_docs": [
                {"_id": "doc5", "rating": 3},
                {"_id": "doc6", "rating": 2},
                {"_id": "doc7", "rating": 1},
            ],
        },
        {
            "query_text": "database connection pooling",
            "relevant_docs": [
                {"_id": "doc8", "rating": 3},
                {"_id": "doc9", "rating": 2},
            ],
        },
        {
            "query_text": "error handling best practices",
            "relevant_docs": [
                {"_id": "doc10", "rating": 3},
                {"_id": "doc11", "rating": 2},
                {"_id": "doc12", "rating": 1},
            ],
        },
    ]

    # Extend or truncate to desired number of queries
    if num_queries > len(sample_queries):
        # Duplicate queries with modified text
        base_queries = sample_queries.copy()
        for i in range(num_queries - len(sample_queries)):
            base_query = base_queries[i % len(base_queries)]
            modified_query = {
                "query_text": f"{base_query['query_text']} variant {i + 1}",
                "relevant_docs": [
                    {"_id": f"{doc['_id']}_v{i + 1}", "rating": doc["rating"]}
                    for doc in base_query["relevant_docs"]
                ],
            }
            sample_queries.append(modified_query)

    sample_queries = sample_queries[:num_queries]

    # Convert to EvaluationQuery objects
    evaluation_queries = []
    for i, q in enumerate(sample_queries):
        query = EvaluationQuery(
            query_id=f"sample_query_{i + 1}",
            query_text=q["query_text"],
            relevant_docs=q["relevant_docs"],
        )
        evaluation_queries.append(query)

    # Create dataset
    from datetime import datetime

    dataset = EvaluationDataset(
        name="sample_evaluation_dataset",
        description=f"Sample dataset with {num_queries} queries for testing evaluation tools",
        queries=evaluation_queries,
        created_at=datetime.now(),
    )

    # Save dataset
    evaluator = RankEvaluator()
    evaluator.save_dataset(dataset, output_path)

    logger.info(f"Sample dataset with {num_queries} queries saved to {output_path}")


def run_evaluation(
    dataset_path: str,
    strategies: List[str],
    index: Optional[str] = None,
    k: int = 50,
    output_dir: str = "evaluation_results",
) -> None:
    """Run evaluation on specified strategies.

    Args:
        dataset_path: Path to evaluation dataset
        strategies: List of search strategies to evaluate
        index: OpenSearch index to evaluate against
        k: Number of results to retrieve
        output_dir: Directory to save results
    """
    try:
        # Initialize evaluator
        evaluator = RankEvaluator()

        # Load dataset
        dataset = evaluator.load_dataset(dataset_path)
        logger.info(
            f"Loaded dataset '{dataset.name}' with {len(dataset.queries)} queries"
        )

        # Run evaluation
        results = evaluator.evaluate_search_strategies(
            dataset=dataset, strategies=strategies, index=index, k=k
        )

        if not results:
            logger.error("No evaluation results generated")
            return

        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)

        # Save individual results
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for strategy, result in results.items():
            output_path = f"{output_dir}/{strategy}_{timestamp}.json"
            evaluator.evaluation_client.save_evaluation_result(result, output_path)
            logger.info(f"Results for {strategy} saved to {output_path}")

        # Generate comparison report
        comparison = evaluator.compare_strategies(results)
        comparison_path = f"{output_dir}/comparison_{timestamp}.json"
        with open(comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison report saved to {comparison_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)

        for strategy, result in results.items():
            metrics = result.overall_metrics
            print(f"\n{strategy.upper()}:")
            print(f"  nDCG@10:     {metrics.get('avg_ndcg_at_10', 0):.3f}")
            print(f"  Recall@50:   {metrics.get('avg_recall_at_50', 0):.3f}")
            print(f"  Precision@10: {metrics.get('avg_precision_at_10', 0):.3f}")
            print(f"  MRR:         {metrics.get('avg_mean_reciprocal_rank', 0):.3f}")

        # Show best performing strategy
        if len(results) > 1:
            best_ndcg = max(
                results.items(),
                key=lambda x: x[1].overall_metrics.get("avg_ndcg_at_10", 0),
            )
            best_recall = max(
                results.items(),
                key=lambda x: x[1].overall_metrics.get("avg_recall_at_50", 0),
            )

            print("\nBEST PERFORMERS:")
            print(
                f"  nDCG@10:  {best_ndcg[0]} ({best_ndcg[1].overall_metrics.get('avg_ndcg_at_10', 0):.3f})"
            )
            print(
                f"  Recall@50: {best_recall[0]} ({best_recall[1].overall_metrics.get('avg_recall_at_50', 0):.3f})"
            )

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def run_continuous_evaluation(
    dataset_path: str,
    strategies: List[str],
    output_dir: str = "evaluation_results",
    threshold: float = 0.8,
) -> None:
    """Run continuous evaluation with performance monitoring.

    Args:
        dataset_path: Path to evaluation dataset
        strategies: List of search strategies to evaluate
        output_dir: Directory to save results
        threshold: Performance threshold for alerts
    """
    try:
        evaluator = RankEvaluator()
        dataset = evaluator.load_dataset(dataset_path)

        logger.info(f"Running continuous evaluation on {len(strategies)} strategies")

        summary = evaluator.run_continuous_evaluation(
            dataset=dataset,
            strategies=strategies,
            output_dir=output_dir,
            baseline_threshold=threshold,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("CONTINUOUS EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Timestamp: {summary['timestamp']}")
        print(f"Strategies: {', '.join(summary['strategies_evaluated'])}")
        print(f"Total Queries: {summary['total_queries']}")

        if summary["alerts"]:
            print("\nALERTS:")
            for alert in summary["alerts"]:
                print(f"  {alert}")
        else:
            print("\n✅ All strategies performing within acceptable thresholds")

        print("\nPERFORMANCE SUMMARY:")
        for strategy, metrics in summary["results_summary"].items():
            print(f"  {strategy}:")
            print(f"    nDCG@10:     {metrics['ndcg_at_10']:.3f}")
            print(f"    Recall@50:   {metrics['recall_at_50']:.3f}")
            print(f"    Precision@10: {metrics['precision_at_10']:.3f}")

        print("\n" + "=" * 60)

    except Exception as e:
        logger.error(f"Continuous evaluation failed: {e}")
        sys.exit(1)


def create_dataset_from_logs(
    log_file: str,
    output_path: str,
    max_queries: int = 100,
    relevance_threshold: float = 0.7,
) -> None:
    """Create evaluation dataset from search logs.

    Args:
        log_file: Path to search log file (one query per line)
        output_path: Path to save generated dataset
        max_queries: Maximum number of queries to process
        relevance_threshold: Threshold for relevance judgments
    """
    try:
        # Read queries from log file
        with open(log_file, "r") as f:
            queries = [line.strip() for line in f if line.strip()]

        queries = queries[:max_queries]
        logger.info(f"Processing {len(queries)} queries from log file")

        # Create dataset
        evaluator = RankEvaluator()
        dataset = evaluator.create_dataset_from_search_logs(
            log_queries=queries, relevance_threshold=relevance_threshold
        )

        # Save dataset
        evaluator.save_dataset(dataset, output_path)

        logger.info(
            f"Dataset created with {len(dataset.queries)} queries and saved to {output_path}"
        )

    except Exception as e:
        logger.error(f"Failed to create dataset from logs: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Search Evaluation Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sample dataset
  python -m src.evaluation.cli create-sample --output sample_dataset.json --queries 20
  
  # Run evaluation
  python -m src.evaluation.cli evaluate --dataset sample_dataset.json --strategies enhanced_rrf bm25
  
  # Run continuous evaluation
  python -m src.evaluation.cli continuous --dataset sample_dataset.json --strategies enhanced_rrf
  
  # Create dataset from logs
  python -m src.evaluation.cli from-logs --log-file search.log --output log_dataset.json
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create sample dataset command
    sample_parser = subparsers.add_parser(
        "create-sample", help="Create sample evaluation dataset"
    )
    sample_parser.add_argument(
        "--output", "-o", required=True, help="Output path for dataset"
    )
    sample_parser.add_argument(
        "--queries", "-q", type=int, default=10, help="Number of sample queries"
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run evaluation on search strategies"
    )
    eval_parser.add_argument(
        "--dataset", "-d", required=True, help="Path to evaluation dataset"
    )
    eval_parser.add_argument(
        "--strategies",
        "-s",
        nargs="+",
        default=["enhanced_rrf", "bm25"],
        help="Search strategies to evaluate",
    )
    eval_parser.add_argument(
        "--index", "-i", help="OpenSearch index to evaluate against"
    )
    eval_parser.add_argument(
        "--k", type=int, default=50, help="Number of results to retrieve"
    )
    eval_parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Output directory for results",
    )

    # Continuous evaluation command
    continuous_parser = subparsers.add_parser(
        "continuous", help="Run continuous evaluation"
    )
    continuous_parser.add_argument(
        "--dataset", "-d", required=True, help="Path to evaluation dataset"
    )
    continuous_parser.add_argument(
        "--strategies",
        "-s",
        nargs="+",
        default=["enhanced_rrf"],
        help="Search strategies to evaluate",
    )
    continuous_parser.add_argument(
        "--output-dir",
        default="evaluation_results",
        help="Output directory for results",
    )
    continuous_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.8,
        help="Performance threshold for alerts",
    )

    # Create dataset from logs command
    logs_parser = subparsers.add_parser(
        "from-logs", help="Create dataset from search logs"
    )
    logs_parser.add_argument(
        "--log-file", "-l", required=True, help="Path to search log file"
    )
    logs_parser.add_argument(
        "--output", "-o", required=True, help="Output path for dataset"
    )
    logs_parser.add_argument(
        "--max-queries", type=int, default=100, help="Maximum queries to process"
    )
    logs_parser.add_argument(
        "--threshold", type=float, default=0.7, help="Relevance threshold"
    )

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        if args.command == "create-sample":
            create_sample_dataset(args.output, args.queries)

        elif args.command == "evaluate":
            run_evaluation(
                dataset_path=args.dataset,
                strategies=args.strategies,
                index=args.index,
                k=args.k,
                output_dir=args.output_dir,
            )

        elif args.command == "continuous":
            run_continuous_evaluation(
                dataset_path=args.dataset,
                strategies=args.strategies,
                output_dir=args.output_dir,
                threshold=args.threshold,
            )

        elif args.command == "from-logs":
            create_dataset_from_logs(
                log_file=args.log_file,
                output_path=args.output,
                max_queries=args.max_queries,
                relevance_threshold=args.threshold,
            )

    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
