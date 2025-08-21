"""OpenSearch Rank Evaluation API client for search quality assessment."""

import logging
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import requests
from src.infra.settings import get_settings
from src.infra.opensearch_client import OpenSearchClient, SearchFilters

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuery:
    """Represents a query for evaluation with expected relevant documents."""

    query_id: str
    query_text: str
    relevant_docs: List[
        Dict[str, Any]
    ]  # List of {"_id": doc_id, "rating": relevance_score}
    filters: Optional[SearchFilters] = None


@dataclass
class EvaluationMetrics:
    """Evaluation metrics from OpenSearch Rank Evaluation API."""

    ndcg_at_10: float
    recall_at_50: float
    precision_at_10: float
    mean_reciprocal_rank: float
    query_id: str
    total_hits: int
    evaluation_timestamp: datetime


@dataclass
class EvaluationResult:
    """Complete evaluation result for a set of queries."""

    overall_metrics: Dict[str, float]
    query_metrics: List[EvaluationMetrics]
    evaluation_id: str
    index_name: str
    search_strategy: str
    total_queries: int
    failed_queries: List[str]
    evaluation_timestamp: datetime


class OpenSearchEvaluationClient:
    """Client for OpenSearch Rank Evaluation API integration."""

    def __init__(self, opensearch_client: Optional[OpenSearchClient] = None):
        """Initialize evaluation client.

        Args:
            opensearch_client: OpenSearch client instance
        """
        self.settings = get_settings()
        self.opensearch_client = opensearch_client or OpenSearchClient()
        self.base_url = self.settings.opensearch.endpoint

    def evaluate_search_quality(
        self,
        queries: List[EvaluationQuery],
        index: Optional[str] = None,
        search_strategy: str = "enhanced_rrf",
        k: int = 50,
    ) -> EvaluationResult:
        """Evaluate search quality using OpenSearch Rank Evaluation API.

        Args:
            queries: List of evaluation queries with expected relevant documents
            index: Index to evaluate (defaults to main search index)
            search_strategy: Search strategy to evaluate
            k: Number of results to retrieve for evaluation

        Returns:
            EvaluationResult with comprehensive metrics
        """
        index = index or self.settings.opensearch.index_name
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting evaluation {evaluation_id} with {len(queries)} queries")

        # Build evaluation request for OpenSearch Rank Evaluation API
        evaluation_request = self._build_evaluation_request(queries, search_strategy, k)

        try:
            # Execute evaluation using OpenSearch Rank Evaluation API
            response = self._execute_rank_evaluation(index, evaluation_request)

            # Parse results and calculate metrics
            query_metrics = self._parse_evaluation_response(response, queries)
            overall_metrics = self._calculate_overall_metrics(query_metrics)

            result = EvaluationResult(
                overall_metrics=overall_metrics,
                query_metrics=query_metrics,
                evaluation_id=evaluation_id,
                index_name=index,
                search_strategy=search_strategy,
                total_queries=len(queries),
                failed_queries=[],
                evaluation_timestamp=datetime.now(),
            )

            logger.info(
                f"Evaluation completed: nDCG@10={overall_metrics.get('avg_ndcg_at_10', 0):.3f}, "
                f"Recall@50={overall_metrics.get('avg_recall_at_50', 0):.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

    def _build_evaluation_request(
        self, queries: List[EvaluationQuery], search_strategy: str, k: int
    ) -> Dict[str, Any]:
        """Build OpenSearch Rank Evaluation API request."""

        # Define evaluation metrics
        metrics = {
            "ndcg": {"k": 10, "normalize": True},
            "recall": {"k": 50},
            "precision": {"k": 10},
            "mean_reciprocal_rank": {},
        }

        # Build requests for each query
        requests = []
        for query in queries:
            # Get the appropriate search query based on strategy
            search_query = self._get_search_query(query, search_strategy, k)

            request = {
                "id": query.query_id,
                "request": search_query,
                "ratings": query.relevant_docs,
            }
            requests.append(request)

        return {"requests": requests, "metric": metrics}

    def _get_search_query(
        self, query: EvaluationQuery, search_strategy: str, k: int
    ) -> Dict[str, Any]:
        """Get search query based on strategy."""

        if search_strategy == "enhanced_rrf":
            # Use hybrid search query
            return self.opensearch_client._build_hybrid_query(
                query.query_text,
                query_vector=None,  # Will be generated internally
                k=k,
            )
        elif search_strategy == "bm25":
            # Use BM25 search query
            return self.opensearch_client._build_simple_bm25_query(
                query.query_text, k=k
            )
        elif search_strategy == "knn":
            # Use kNN search query (requires vector)
            # This would need vector generation
            raise NotImplementedError("kNN evaluation requires vector generation")
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")

    def _execute_rank_evaluation(
        self, index: str, evaluation_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute rank evaluation using OpenSearch API."""

        url = f"{self.base_url}/{index}/_rank_eval"

        headers = {"Content-Type": "application/json"}

        # Add authentication if configured
        auth = None
        if (
            hasattr(self.settings.opensearch, "username")
            and self.settings.opensearch.username
        ):
            auth = (
                self.settings.opensearch.username,
                self.settings.opensearch.password,
            )

        response = requests.post(
            url, json=evaluation_request, headers=headers, auth=auth, timeout=30
        )

        if response.status_code != 200:
            raise Exception(
                f"Rank evaluation failed: {response.status_code} - {response.text}"
            )

        return response.json()

    def _parse_evaluation_response(
        self, response: Dict[str, Any], queries: List[EvaluationQuery]
    ) -> List[EvaluationMetrics]:
        """Parse OpenSearch rank evaluation response."""

        query_metrics = []
        details = response.get("details", {})

        for query in queries:
            query_detail = details.get(query.query_id, {})

            # Extract metrics from response
            metric_scores = query_detail.get("metric_score", 0.0)
            hits = query_detail.get("hits", [])

            # Parse individual metrics if available
            ndcg_at_10 = (
                query_detail.get("metric_details", {})
                .get("ndcg", {})
                .get("ndcg", metric_scores)
            )
            recall_at_50 = (
                query_detail.get("metric_details", {})
                .get("recall", {})
                .get("recall", 0.0)
            )
            precision_at_10 = (
                query_detail.get("metric_details", {})
                .get("precision", {})
                .get("precision", 0.0)
            )
            mrr = (
                query_detail.get("metric_details", {})
                .get("mean_reciprocal_rank", {})
                .get("mean_reciprocal_rank", 0.0)
            )

            metrics = EvaluationMetrics(
                ndcg_at_10=ndcg_at_10,
                recall_at_50=recall_at_50,
                precision_at_10=precision_at_10,
                mean_reciprocal_rank=mrr,
                query_id=query.query_id,
                total_hits=len(hits),
                evaluation_timestamp=datetime.now(),
            )

            query_metrics.append(metrics)

        return query_metrics

    def _calculate_overall_metrics(
        self, query_metrics: List[EvaluationMetrics]
    ) -> Dict[str, float]:
        """Calculate overall metrics across all queries."""

        if not query_metrics:
            return {}

        total_queries = len(query_metrics)

        return {
            "avg_ndcg_at_10": sum(m.ndcg_at_10 for m in query_metrics) / total_queries,
            "avg_recall_at_50": sum(m.recall_at_50 for m in query_metrics)
            / total_queries,
            "avg_precision_at_10": sum(m.precision_at_10 for m in query_metrics)
            / total_queries,
            "avg_mean_reciprocal_rank": sum(
                m.mean_reciprocal_rank for m in query_metrics
            )
            / total_queries,
            "total_queries": total_queries,
            "avg_total_hits": sum(m.total_hits for m in query_metrics) / total_queries,
        }

    def save_evaluation_result(
        self, result: EvaluationResult, output_path: str
    ) -> None:
        """Save evaluation result to JSON file."""

        # Convert dataclasses to dict for JSON serialization
        result_dict = {
            "evaluation_id": result.evaluation_id,
            "index_name": result.index_name,
            "search_strategy": result.search_strategy,
            "total_queries": result.total_queries,
            "failed_queries": result.failed_queries,
            "evaluation_timestamp": result.evaluation_timestamp.isoformat(),
            "overall_metrics": result.overall_metrics,
            "query_metrics": [
                {
                    "query_id": m.query_id,
                    "ndcg_at_10": m.ndcg_at_10,
                    "recall_at_50": m.recall_at_50,
                    "precision_at_10": m.precision_at_10,
                    "mean_reciprocal_rank": m.mean_reciprocal_rank,
                    "total_hits": m.total_hits,
                    "evaluation_timestamp": m.evaluation_timestamp.isoformat(),
                }
                for m in result.query_metrics
            ],
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

        logger.info(f"Evaluation result saved to {output_path}")
