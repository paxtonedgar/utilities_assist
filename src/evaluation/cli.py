"""Command-line interface for the search evaluation tools."""

import click
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from .rank_evaluator import RankEvaluator, EvaluationDataset
from .evaluation_client import EvaluationQuery
from src.infra.opensearch_client import SearchFilters

logger = logging.getLogger(__name__)


@click.group()
def evaluation():
    """Search evaluation tools."""
    pass


@evaluation.command()
@click.option('--name', required=True, help='Dataset name')
@click.option('--description', required=True, help='Dataset description')
@click.option('--output', required=True, help='Output file path')
@click.option('--num-queries', default=10, help='Number of sample queries to generate')
def create_sample_dataset(name: str, description: str, output: str, num_queries: int):
    """Create a sample evaluation dataset."""
    try:
        # Generate sample queries
        sample_queries = []
        query_templates = [
            "How to configure OpenSearch",
            "API documentation for search",
            "Troubleshooting connection issues",
            "Performance optimization tips",
            "Security best practices",
            "Index mapping configuration",
            "Query DSL examples",
            "Aggregation queries",
            "Cluster monitoring",
            "Backup and restore procedures"
        ]
        
        for i in range(min(num_queries, len(query_templates))):
            query = EvaluationQuery(
                query_id=f"sample_{i+1}",
                query_text=query_templates[i],
                relevant_docs=[
                    {"_id": f"doc_{i+1}_1", "rating": 3},
                    {"_id": f"doc_{i+1}_2", "rating": 2},
                    {"_id": f"doc_{i+1}_3", "rating": 1}
                ]
            )
            sample_queries.append(query)
        
        # Create dataset
        dataset = EvaluationDataset(
            name=name,
            description=description,
            queries=sample_queries,
            created_at=datetime.now(),
            metadata={'source': 'cli_generated', 'type': 'sample'}
        )
        
        # Save dataset
        evaluator = RankEvaluator()
        evaluator.save_dataset(dataset, output)
        
        click.echo(f"Sample dataset created with {len(sample_queries)} queries")
        click.echo(f"Saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error creating sample dataset: {e}", err=True)
        raise click.Abort()


@evaluation.command()
@click.option('--dataset', required=True, help='Path to evaluation dataset')
@click.option('--strategies', required=True, help='Comma-separated list of strategies to evaluate')
@click.option('--index', help='OpenSearch index to evaluate against')
@click.option('--k', default=50, help='Number of results to retrieve')
@click.option('--output', help='Output file for results')
def run_evaluation(dataset: str, strategies: str, index: str, k: int, output: str):
    """Run evaluation against multiple search strategies."""
    try:
        # Load dataset
        evaluator = RankEvaluator()
        eval_dataset = evaluator.load_dataset(dataset)
        
        # Parse strategies
        strategy_list = [s.strip() for s in strategies.split(',')]
        
        click.echo(f"Evaluating {len(strategy_list)} strategies against {len(eval_dataset.queries)} queries")
        
        # Run evaluation
        results = evaluator.evaluate_search_strategies(
            dataset=eval_dataset,
            strategies=strategy_list,
            index=index,
            k=k
        )
        
        # Display results
        click.echo("\nEvaluation Results:")
        click.echo("=" * 50)
        
        for strategy, result in results.items():
            metrics = result.overall_metrics
            click.echo(f"\n{strategy}:")
            click.echo(f"  nDCG@10: {metrics.get('ndcg_at_10', 0):.3f}")
            click.echo(f"  Recall@50: {metrics.get('recall_at_50', 0):.3f}")
            click.echo(f"  Precision@10: {metrics.get('precision_at_10', 0):.3f}")
            click.echo(f"  MRR: {metrics.get('mean_reciprocal_rank', 0):.3f}")
            click.echo(f"  Failed queries: {len(result.failed_queries)}")
        
        # Compare strategies
        if len(results) > 1:
            comparison = evaluator.compare_strategies(results)
            
            click.echo("\nStrategy Rankings:")
            click.echo("-" * 30)
            for i, (strategy, metrics) in enumerate(comparison['rankings'], 1):
                click.echo(f"{i}. {strategy}: {metrics.get('ndcg_at_10', 0):.3f}")
        
        # Save results if output specified
        if output:
            results_data = {
                'evaluation_timestamp': datetime.now().isoformat(),
                'dataset_name': eval_dataset.name,
                'strategies': strategy_list,
                'results': {k: {
                    'overall_metrics': v.overall_metrics,
                    'total_queries': v.total_queries,
                    'failed_queries': v.failed_queries
                } for k, v in results.items()}
            }
            
            with open(output, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            click.echo(f"\nResults saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error running evaluation: {e}", err=True)
        raise click.Abort()


@evaluation.command()
@click.option('--dataset', required=True, help='Path to evaluation dataset')
@click.option('--strategies', required=True, help='Comma-separated list of strategies')
@click.option('--baseline', help='Path to baseline results for comparison')
@click.option('--threshold', default=0.05, help='Degradation threshold for alerts')
@click.option('--index', help='OpenSearch index to evaluate against')
def continuous_evaluation(dataset: str, strategies: str, baseline: str, threshold: float, index: str):
    """Run continuous evaluation with degradation detection."""
    try:
        # Load dataset
        evaluator = RankEvaluator()
        eval_dataset = evaluator.load_dataset(dataset)
        
        # Parse strategies
        strategy_list = [s.strip() for s in strategies.split(',')]
        
        # Load baseline if provided
        baseline_results = None
        if baseline:
            with open(baseline, 'r') as f:
                baseline_data = json.load(f)
                # Convert to EvaluationResult objects if needed
                # This is simplified - in practice you'd need proper deserialization
                baseline_results = baseline_data.get('results', {})
        
        click.echo(f"Running continuous evaluation for {len(strategy_list)} strategies")
        
        # Run evaluation
        results = evaluator.run_continuous_evaluation(
            dataset=eval_dataset,
            strategies=strategy_list,
            baseline_results=baseline_results,
            degradation_threshold=threshold,
            index=index
        )
        
        # Display results
        click.echo("\nContinuous Evaluation Results:")
        click.echo("=" * 40)
        
        current_results = results['current_results']
        for strategy, result in current_results.items():
            metrics = result.overall_metrics
            click.echo(f"\n{strategy}:")
            click.echo(f"  nDCG@10: {metrics.get('ndcg_at_10', 0):.3f}")
        
        # Display alerts
        alerts = results['alerts']
        if alerts:
            click.echo("\n⚠️  PERFORMANCE ALERTS:")
            click.echo("=" * 30)
            for alert in alerts:
                click.echo(f"Strategy: {alert['strategy']}")
                click.echo(f"Metric: {alert['metric']}")
                click.echo(f"Degradation: {alert['degradation']:.3f}")
                click.echo(f"Severity: {alert['severity']}")
                click.echo("-" * 20)
        else:
            click.echo("\n✅ No performance degradation detected")
        
    except Exception as e:
        click.echo(f"Error in continuous evaluation: {e}", err=True)
        raise click.Abort()


@evaluation.command()
@click.option('--logs', required=True, help='Path to search logs JSON file')
@click.option('--name', required=True, help='Dataset name')
@click.option('--description', required=True, help='Dataset description')
@click.option('--output', required=True, help='Output file path')
@click.option('--threshold', default=0.7, help='Relevance threshold (CTR)')
@click.option('--min-results', default=5, help='Minimum results per query')
def create_dataset_from_logs(logs: str, name: str, description: str, output: str, threshold: float, min_results: int):
    """Create evaluation dataset from search logs."""
    try:
        # Load search logs
        with open(logs, 'r') as f:
            search_logs = json.load(f)
        
        if not isinstance(search_logs, list):
            raise ValueError("Search logs must be a list of log entries")
        
        # Create dataset
        evaluator = RankEvaluator()
        dataset = evaluator.create_dataset_from_search_logs(
            search_logs=search_logs,
            name=name,
            description=description,
            relevance_threshold=threshold,
            min_results_per_query=min_results
        )
        
        # Save dataset
        evaluator.save_dataset(dataset, output)
        
        click.echo(f"Dataset created from {len(search_logs)} log entries")
        click.echo(f"Generated {len(dataset.queries)} evaluation queries")
        click.echo(f"Saved to: {output}")
        
        # Display metadata
        if dataset.metadata:
            click.echo("\nDataset metadata:")
            for key, value in dataset.metadata.items():
                click.echo(f"  {key}: {value}")
        
    except Exception as e:
        click.echo(f"Error creating dataset from logs: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    evaluation()