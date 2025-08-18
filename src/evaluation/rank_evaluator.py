"""High-level interface for running search quality evaluations and managing evaluation datasets."""

import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .evaluation_client import OpenSearchEvaluationClient, EvaluationQuery, EvaluationResult
from .metrics import aggregate_metrics, statistical_significance_test
from src.infra.opensearch_client import OpenSearchClient, SearchFilters

logger = logging.getLogger(__name__)


@dataclass
class EvaluationDataset:
    """Dataset for evaluation with queries and relevance judgments."""
    name: str
    description: str
    queries: List[EvaluationQuery]
    created_at: datetime
    version: str = "1.0"
    metadata: Optional[Dict[str, Any]] = None


class RankEvaluator:
    """High-level interface for search quality evaluation."""
    
    def __init__(self, opensearch_client: Optional[OpenSearchClient] = None):
        """Initialize rank evaluator.
        
        Args:
            opensearch_client: OpenSearch client instance
        """
        self.evaluation_client = OpenSearchEvaluationClient(opensearch_client)
        self.opensearch_client = opensearch_client or OpenSearchClient()
        
    def evaluate_search_strategies(
        self,
        dataset: EvaluationDataset,
        strategies: List[str],
        index: Optional[str] = None,
        k: int = 50
    ) -> Dict[str, EvaluationResult]:
        """Evaluate multiple search strategies against a dataset.
        
        Args:
            dataset: Evaluation dataset with queries and relevance judgments
            strategies: List of search strategy names to evaluate
            index: Index to evaluate against
            k: Number of results to retrieve for evaluation
            
        Returns:
            Dictionary mapping strategy names to evaluation results
        """
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
                logger.info(f"Strategy {strategy} - nDCG@10: {result.overall_metrics.get('ndcg_at_10', 0):.3f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate strategy {strategy}: {e}")
                
        return results
    
    def compare_strategies(
        self,
        results: Dict[str, EvaluationResult],
        primary_metric: str = "ndcg_at_10"
    ) -> Dict[str, Any]:
        """Compare evaluation results across strategies.
        
        Args:
            results: Dictionary of strategy evaluation results
            primary_metric: Primary metric for comparison
            
        Returns:
            Comparison analysis with rankings and statistical significance
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 strategies to compare")
            
        # Extract metrics for comparison
        strategy_metrics = {}
        for strategy, result in results.items():
            strategy_metrics[strategy] = {
                'overall': result.overall_metrics,
                'query_level': [asdict(qm) for qm in result.query_metrics]
            }
        
        # Rank strategies by primary metric
        rankings = sorted(
            strategy_metrics.items(),
            key=lambda x: x[1]['overall'].get(primary_metric, 0),
            reverse=True
        )
        
        # Statistical significance testing
        significance_tests = {}
        if len(rankings) >= 2:
            best_strategy = rankings[0][0]
            best_scores = [qm[primary_metric] for qm in strategy_metrics[best_strategy]['query_level']]
            
            for strategy, _ in rankings[1:]:
                strategy_scores = [qm[primary_metric] for qm in strategy_metrics[strategy]['query_level']]
                p_value = statistical_significance_test(best_scores, strategy_scores)
                significance_tests[f"{best_strategy}_vs_{strategy}"] = {
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return {
            'rankings': [(strategy, metrics['overall']) for strategy, metrics in rankings],
            'primary_metric': primary_metric,
            'significance_tests': significance_tests,
            'summary': {
                'best_strategy': rankings[0][0],
                'best_score': rankings[0][1]['overall'].get(primary_metric, 0),
                'improvement_over_worst': (
                    rankings[0][1]['overall'].get(primary_metric, 0) - 
                    rankings[-1][1]['overall'].get(primary_metric, 0)
                ) if len(rankings) > 1 else 0
            }
        }
    
    def create_dataset_from_search_logs(
        self,
        search_logs: List[Dict[str, Any]],
        name: str,
        description: str,
        relevance_threshold: float = 0.7,
        min_results_per_query: int = 5
    ) -> EvaluationDataset:
        """Create evaluation dataset from search logs with automatic relevance judgments.
        
        Args:
            search_logs: List of search log entries with queries and clicked results
            name: Dataset name
            description: Dataset description
            relevance_threshold: Minimum click-through rate to consider relevant
            min_results_per_query: Minimum number of results needed per query
            
        Returns:
            EvaluationDataset with generated relevance judgments
        """
        query_data = {}
        
        # Aggregate search logs by query
        for log_entry in search_logs:
            query_text = log_entry.get('query', '').strip().lower()
            if not query_text:
                continue
                
            if query_text not in query_data:
                query_data[query_text] = {
                    'total_searches': 0,
                    'clicked_docs': {},
                    'filters': log_entry.get('filters')
                }
            
            query_data[query_text]['total_searches'] += 1
            
            # Track clicked documents
            clicked_docs = log_entry.get('clicked_documents', [])
            for doc_id in clicked_docs:
                if doc_id not in query_data[query_text]['clicked_docs']:
                    query_data[query_text]['clicked_docs'][doc_id] = 0
                query_data[query_text]['clicked_docs'][doc_id] += 1
        
        # Generate evaluation queries
        evaluation_queries = []
        for query_text, data in query_data.items():
            if data['total_searches'] < min_results_per_query:
                continue
                
            # Calculate relevance scores based on click-through rates
            relevant_docs = []
            for doc_id, clicks in data['clicked_docs'].items():
                ctr = clicks / data['total_searches']
                if ctr >= relevance_threshold:
                    # Convert CTR to relevance rating (0-3 scale)
                    rating = min(3, int(ctr * 4))
                    relevant_docs.append({
                        "_id": doc_id,
                        "rating": rating
                    })
            
            if relevant_docs:
                # Convert filters if present
                filters = None
                if data['filters']:
                    filters = SearchFilters(**data['filters'])
                    
                evaluation_queries.append(EvaluationQuery(
                    query_id=str(uuid.uuid4()),
                    query_text=query_text,
                    relevant_docs=relevant_docs,
                    filters=filters
                ))
        
        return EvaluationDataset(
            name=name,
            description=description,
            queries=evaluation_queries,
            created_at=datetime.now(),
            metadata={
                'source': 'search_logs',
                'relevance_threshold': relevance_threshold,
                'min_results_per_query': min_results_per_query,
                'total_log_entries': len(search_logs),
                'generated_queries': len(evaluation_queries)
            }
        )
    
    def save_dataset(self, dataset: EvaluationDataset, file_path: str) -> None:
        """Save evaluation dataset to file.
        
        Args:
            dataset: Dataset to save
            file_path: Path to save dataset
        """
        try:
            # Convert to serializable format
            dataset_dict = asdict(dataset)
            dataset_dict['created_at'] = dataset.created_at.isoformat()
            
            # Convert EvaluationQuery objects
            for query in dataset_dict['queries']:
                if 'filters' in query and query['filters']:
                    # Convert SearchFilters to dict
                    query['filters'] = asdict(query['filters'])
            
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(dataset_dict, f, indent=2)
                
            logger.info(f"Dataset saved to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            raise
    
    def load_dataset(self, file_path: str) -> EvaluationDataset:
        """Load evaluation dataset from file.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Loaded EvaluationDataset
        """
        try:
            with open(file_path, 'r') as f:
                dataset_dict = json.load(f)
            
            # Convert back to objects
            dataset_dict['created_at'] = datetime.fromisoformat(dataset_dict['created_at'])
            
            # Convert queries back to EvaluationQuery objects
            queries = []
            for query_dict in dataset_dict['queries']:
                filters = None
                if query_dict.get('filters'):
                    filters = SearchFilters(**query_dict['filters'])
                    
                queries.append(EvaluationQuery(
                    query_id=query_dict['query_id'],
                    query_text=query_dict['query_text'],
                    relevant_docs=query_dict['relevant_docs'],
                    filters=filters
                ))
            
            dataset_dict['queries'] = queries
            return EvaluationDataset(**dataset_dict)
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def run_continuous_evaluation(
        self,
        dataset: EvaluationDataset,
        strategies: List[str],
        baseline_results: Optional[Dict[str, EvaluationResult]] = None,
        degradation_threshold: float = 0.05,
        index: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run continuous evaluation and detect performance degradation.
        
        Args:
            dataset: Evaluation dataset
            strategies: Search strategies to evaluate
            baseline_results: Previous evaluation results for comparison
            degradation_threshold: Threshold for detecting significant degradation
            index: Index to evaluate against
            
        Returns:
            Evaluation results with degradation alerts
        """
        # Run current evaluation
        current_results = self.evaluate_search_strategies(
            dataset=dataset,
            strategies=strategies,
            index=index
        )
        
        alerts = []
        
        # Compare with baseline if provided
        if baseline_results:
            for strategy in strategies:
                if strategy not in baseline_results or strategy not in current_results:
                    continue
                    
                baseline_ndcg = baseline_results[strategy].overall_metrics.get('ndcg_at_10', 0)
                current_ndcg = current_results[strategy].overall_metrics.get('ndcg_at_10', 0)
                
                degradation = baseline_ndcg - current_ndcg
                if degradation > degradation_threshold:
                    alerts.append({
                        'strategy': strategy,
                        'metric': 'ndcg_at_10',
                        'baseline_value': baseline_ndcg,
                        'current_value': current_ndcg,
                        'degradation': degradation,
                        'severity': 'high' if degradation > 0.1 else 'medium'
                    })
        
        return {
            'current_results': current_results,
            'alerts': alerts,
            'evaluation_timestamp': datetime.now().isoformat(),
            'has_degradation': len(alerts) > 0
        }