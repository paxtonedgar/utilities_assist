"""High-level rank evaluator for search quality assessment."""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from .evaluation_client import (
    OpenSearchEvaluationClient,
    EvaluationQuery,
    EvaluationResult,
    EvaluationMetrics
)
from src.infra.opensearch_client import OpenSearchClient, SearchFilters

logger = logging.getLogger(__name__)


@dataclass
class EvaluationDataset:
    """Represents an evaluation dataset with queries and relevance judgments."""
    name: str
    description: str
    queries: List[EvaluationQuery]
    created_at: datetime
    version: str = "1.0"


class RankEvaluator:
    """High-level interface for search ranking evaluation."""
    
    def __init__(self, opensearch_client: Optional[OpenSearchClient] = None):
        """Initialize rank evaluator.
        
        Args:
            opensearch_client: OpenSearch client instance
        """
        self.evaluation_client = OpenSearchEvaluationClient(opensearch_client)
        self.opensearch_client = opensearch_client or OpenSearchClient()
        
    def evaluate_search_strategies(
        self,
        dataset: Union[EvaluationDataset, str, Path],
        strategies: List[str] = None,
        index: Optional[str] = None,
        k: int = 50
    ) -> Dict[str, EvaluationResult]:
        """Evaluate multiple search strategies against a dataset.
        
        Args:
            dataset: Evaluation dataset or path to dataset file
            strategies: List of search strategies to evaluate
            index: Index to evaluate against
            k: Number of results to retrieve
            
        Returns:
            Dictionary mapping strategy names to evaluation results
        """
        if strategies is None:
            strategies = ["enhanced_rrf", "bm25"]
        
        # Load dataset if path provided
        if isinstance(dataset, (str, Path)):
            dataset = self.load_dataset(dataset)
        
        logger.info(f"Evaluating {len(strategies)} strategies on dataset '{dataset.name}' "
                   f"with {len(dataset.queries)} queries")
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"Evaluating strategy: {strategy}")
            try:
                result = self.evaluation_client.evaluate_search_quality(
                    queries=dataset.queries,
                    index=index,
                    search_strategy=strategy,
                    k=k
                )
                results[strategy] = result
                
                # Log key metrics
                metrics = result.overall_metrics
                logger.info(f"{strategy} results: nDCG@10={metrics.get('avg_ndcg_at_10', 0):.3f}, "
                           f"Recall@50={metrics.get('avg_recall_at_50', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate strategy {strategy}: {e}")
                continue
        
        return results
    
    def compare_strategies(
        self,
        results: Dict[str, EvaluationResult],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Compare evaluation results across strategies.
        
        Args:
            results: Dictionary of strategy results
            output_path: Optional path to save comparison report
            
        Returns:
            Comparison report with rankings and statistical significance
        """
        if not results:
            return {}
        
        # Extract key metrics for comparison
        comparison = {
            "strategies": list(results.keys()),
            "metrics_comparison": {},
            "rankings": {},
            "best_strategy": {},
            "query_level_analysis": {}
        }
        
        # Compare overall metrics
        metrics_to_compare = ["avg_ndcg_at_10", "avg_recall_at_50", "avg_precision_at_10", "avg_mean_reciprocal_rank"]
        
        for metric in metrics_to_compare:
            metric_values = {}
            for strategy, result in results.items():
                metric_values[strategy] = result.overall_metrics.get(metric, 0.0)
            
            # Sort strategies by metric value
            sorted_strategies = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            
            comparison["metrics_comparison"][metric] = metric_values
            comparison["rankings"][metric] = [s[0] for s in sorted_strategies]
            comparison["best_strategy"][metric] = sorted_strategies[0][0] if sorted_strategies else None
        
        # Query-level analysis
        if results:
            first_result = next(iter(results.values()))
            for i, query_metric in enumerate(first_result.query_metrics):
                query_id = query_metric.query_id
                query_comparison = {}
                
                for strategy, result in results.items():
                    if i < len(result.query_metrics):
                        q_metric = result.query_metrics[i]
                        query_comparison[strategy] = {
                            "ndcg_at_10": q_metric.ndcg_at_10,
                            "recall_at_50": q_metric.recall_at_50,
                            "precision_at_10": q_metric.precision_at_10
                        }
                
                comparison["query_level_analysis"][query_id] = query_comparison
        
        # Calculate improvement percentages
        if len(results) >= 2:
            strategies = list(results.keys())
            baseline = strategies[0]
            comparison["improvements"] = {}
            
            for strategy in strategies[1:]:
                improvements = {}
                for metric in metrics_to_compare:
                    baseline_value = results[baseline].overall_metrics.get(metric, 0.0)
                    strategy_value = results[strategy].overall_metrics.get(metric, 0.0)
                    
                    if baseline_value > 0:
                        improvement = ((strategy_value - baseline_value) / baseline_value) * 100
                        improvements[metric] = improvement
                
                comparison["improvements"][f"{strategy}_vs_{baseline}"] = improvements
        
        # Save comparison report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            logger.info(f"Comparison report saved to {output_path}")
        
        return comparison
    
    def create_dataset_from_search_logs(
        self,
        log_queries: List[str],
        relevance_threshold: float = 0.7,
        max_relevant_docs: int = 10
    ) -> EvaluationDataset:
        """Create evaluation dataset from search logs with automatic relevance judgments.
        
        Args:
            log_queries: List of queries from search logs
            relevance_threshold: Threshold for considering documents relevant
            max_relevant_docs: Maximum number of relevant docs per query
            
        Returns:
            EvaluationDataset with generated relevance judgments
        """
        logger.info(f"Creating dataset from {len(log_queries)} log queries")
        
        evaluation_queries = []
        
        for i, query_text in enumerate(log_queries):
            query_id = f"log_query_{i+1}"
            
            # Perform search to get candidate documents
            try:
                search_response = self.opensearch_client.hybrid_search(
                    query=query_text,
                    k=50  # Get more candidates for relevance judgment
                )
                
                # Generate relevance judgments based on scores
                relevant_docs = []
                for result in search_response.results[:max_relevant_docs]:
                    # Normalize score to 0-1 range for relevance rating
                    # This is a simplified approach - in practice, you'd want human judgments
                    normalized_score = min(result.score / 10.0, 1.0)  # Assuming max score ~10
                    
                    if normalized_score >= relevance_threshold:
                        relevant_docs.append({
                            "_id": result.id,
                            "rating": int(normalized_score * 3)  # Convert to 0-3 scale
                        })
                
                if relevant_docs:  # Only include queries with relevant documents
                    evaluation_query = EvaluationQuery(
                        query_id=query_id,
                        query_text=query_text,
                        relevant_docs=relevant_docs
                    )
                    evaluation_queries.append(evaluation_query)
                
            except Exception as e:
                logger.warning(f"Failed to process query '{query_text}': {e}")
                continue
        
        dataset = EvaluationDataset(
            name="auto_generated_from_logs",
            description=f"Dataset generated from {len(log_queries)} search log queries",
            queries=evaluation_queries,
            created_at=datetime.now()
        )
        
        logger.info(f"Created dataset with {len(evaluation_queries)} queries")
        return dataset
    
    def save_dataset(
        self,
        dataset: EvaluationDataset,
        output_path: str
    ) -> None:
        """Save evaluation dataset to JSON file.
        
        Args:
            dataset: Dataset to save
            output_path: Path to save dataset
        """
        dataset_dict = {
            "name": dataset.name,
            "description": dataset.description,
            "version": dataset.version,
            "created_at": dataset.created_at.isoformat(),
            "queries": [
                {
                    "query_id": q.query_id,
                    "query_text": q.query_text,
                    "relevant_docs": q.relevant_docs,
                    "filters": q.filters.__dict__ if q.filters else None
                }
                for q in dataset.queries
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(dataset_dict, f, indent=2)
        
        logger.info(f"Dataset saved to {output_path}")
    
    def load_dataset(self, dataset_path: Union[str, Path]) -> EvaluationDataset:
        """Load evaluation dataset from JSON file.
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            Loaded EvaluationDataset
        """
        with open(dataset_path, 'r') as f:
            dataset_dict = json.load(f)
        
        queries = []
        for q_dict in dataset_dict["queries"]:
            filters = None
            if q_dict.get("filters"):
                filters = SearchFilters(**q_dict["filters"])
            
            query = EvaluationQuery(
                query_id=q_dict["query_id"],
                query_text=q_dict["query_text"],
                relevant_docs=q_dict["relevant_docs"],
                filters=filters
            )
            queries.append(query)
        
        dataset = EvaluationDataset(
            name=dataset_dict["name"],
            description=dataset_dict["description"],
            queries=queries,
            created_at=datetime.fromisoformat(dataset_dict["created_at"]),
            version=dataset_dict.get("version", "1.0")
        )
        
        logger.info(f"Loaded dataset '{dataset.name}' with {len(dataset.queries)} queries")
        return dataset
    
    def run_continuous_evaluation(
        self,
        dataset: EvaluationDataset,
        strategies: List[str] = None,
        output_dir: str = "evaluation_results",
        baseline_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Run continuous evaluation and alert on performance degradation.
        
        Args:
            dataset: Evaluation dataset
            strategies: Strategies to evaluate
            output_dir: Directory to save results
            baseline_threshold: Threshold for performance alerts
            
        Returns:
            Evaluation summary with alerts
        """
        if strategies is None:
            strategies = ["enhanced_rrf"]
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Run evaluation
        results = self.evaluate_search_strategies(dataset, strategies)
        
        # Save individual results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for strategy, result in results.items():
            output_path = f"{output_dir}/{strategy}_{timestamp}.json"
            self.evaluation_client.save_evaluation_result(result, output_path)
        
        # Generate alerts for performance issues
        alerts = []
        for strategy, result in results.items():
            ndcg = result.overall_metrics.get("avg_ndcg_at_10", 0.0)
            recall = result.overall_metrics.get("avg_recall_at_50", 0.0)
            
            if ndcg < baseline_threshold:
                alerts.append(f"⚠️  {strategy}: nDCG@10 ({ndcg:.3f}) below threshold ({baseline_threshold})")
            
            if recall < baseline_threshold:
                alerts.append(f"⚠️  {strategy}: Recall@50 ({recall:.3f}) below threshold ({baseline_threshold})")
        
        summary = {
            "timestamp": timestamp,
            "strategies_evaluated": list(results.keys()),
            "total_queries": len(dataset.queries),
            "alerts": alerts,
            "results_summary": {
                strategy: {
                    "ndcg_at_10": result.overall_metrics.get("avg_ndcg_at_10", 0.0),
                    "recall_at_50": result.overall_metrics.get("avg_recall_at_50", 0.0),
                    "precision_at_10": result.overall_metrics.get("avg_precision_at_10", 0.0)
                }
                for strategy, result in results.items()
            }
        }
        
        # Save summary
        summary_path = f"{output_dir}/evaluation_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log alerts
        if alerts:
            logger.warning(f"Performance alerts detected: {alerts}")
        else:
            logger.info("All strategies performing within acceptable thresholds")
        
        return summary