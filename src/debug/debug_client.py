"""OpenSearch debug client for query analysis and performance monitoring."""

import logging
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime

from src.infra.opensearch_client import OpenSearchClient, SearchFilters

logger = logging.getLogger(__name__)


@dataclass
class DebugQuery:
    """Represents a query for debugging analysis."""
    query_text: str
    search_strategy: str
    filters: Optional[SearchFilters] = None
    index: Optional[str] = None
    document_id: Optional[str] = None  # For explain API
    

@dataclass
class DebugSession:
    """Represents a debugging session with multiple queries."""
    session_id: str
    created_at: datetime
    queries: List[DebugQuery]
    results: Dict[str, Any]
    

class OpenSearchDebugClient:
    """Client for debugging OpenSearch queries using Explain and Profile APIs."""
    
    def __init__(self, opensearch_client: Optional[OpenSearchClient] = None):
        """Initialize debug client.
        
        Args:
            opensearch_client: OpenSearch client instance
        """
        self.opensearch_client = opensearch_client or OpenSearchClient()
        self.client = self.opensearch_client.client
        
    def explain_query(
        self,
        query: Union[str, Dict[str, Any]],
        document_id: str,
        index: Optional[str] = None,
        search_strategy: str = "bm25",
        filters: Optional[SearchFilters] = None
    ) -> Dict[str, Any]:
        """Explain why a document matches (or doesn't match) a query.
        
        Args:
            query: Query text or OpenSearch query dict
            document_id: ID of document to explain
            index: Index to search in
            search_strategy: Search strategy to use
            filters: Optional search filters
            
        Returns:
            Explanation result from OpenSearch
        """
        try:
            # Build query based on strategy
            if isinstance(query, str):
                opensearch_query = self._build_query_for_strategy(
                    query, search_strategy, filters
                )
            else:
                opensearch_query = query
            
            # Use default index if not specified
            target_index = index or self.opensearch_client.default_index
            
            # Call OpenSearch Explain API
            response = self.client.explain(
                index=target_index,
                id=document_id,
                body=opensearch_query
            )
            
            logger.info(f"Explained query for document {document_id} in index {target_index}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to explain query: {e}")
            raise
    
    def profile_query(
        self,
        query: Union[str, Dict[str, Any]],
        index: Optional[str] = None,
        search_strategy: str = "bm25",
        filters: Optional[SearchFilters] = None,
        size: int = 10
    ) -> Dict[str, Any]:
        """Profile query performance using OpenSearch Profile API.
        
        Args:
            query: Query text or OpenSearch query dict
            index: Index to search in
            search_strategy: Search strategy to use
            filters: Optional search filters
            size: Number of results to return
            
        Returns:
            Profile result from OpenSearch
        """
        try:
            # Build query based on strategy
            if isinstance(query, str):
                opensearch_query = self._build_query_for_strategy(
                    query, search_strategy, filters
                )
            else:
                opensearch_query = query
            
            # Add profiling to query
            search_body = {
                "profile": True,
                "query": opensearch_query.get("query", opensearch_query),
                "size": size
            }
            
            # Add other query components if present
            if "_source" in opensearch_query:
                search_body["_source"] = opensearch_query["_source"]
            if "sort" in opensearch_query:
                search_body["sort"] = opensearch_query["sort"]
            
            # Use default index if not specified
            target_index = index or self.opensearch_client.default_index
            
            # Execute search with profiling
            response = self.client.search(
                index=target_index,
                body=search_body
            )
            
            logger.info(f"Profiled query in index {target_index}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to profile query: {e}")
            raise
    
    def debug_search_strategy(
        self,
        query_text: str,
        strategies: List[str] = None,
        index: Optional[str] = None,
        filters: Optional[SearchFilters] = None,
        top_docs: int = 5
    ) -> Dict[str, Any]:
        """Debug multiple search strategies for comparison.
        
        Args:
            query_text: Query to debug
            strategies: List of strategies to compare
            index: Index to search in
            filters: Optional search filters
            top_docs: Number of top documents to analyze
            
        Returns:
            Comparison of strategies with profiling and explanation data
        """
        if strategies is None:
            strategies = ["enhanced_rrf", "bm25"]
        
        results = {
            "query": query_text,
            "strategies": {},
            "comparison": {},
            "timestamp": datetime.now().isoformat()
        }
        
        for strategy in strategies:
            logger.info(f"Debugging strategy: {strategy}")
            
            try:
                # Profile the query
                profile_result = self.profile_query(
                    query=query_text,
                    index=index,
                    search_strategy=strategy,
                    filters=filters,
                    size=top_docs
                )
                
                # Extract key information
                strategy_result = {
                    "profile": profile_result.get("profile", {}),
                    "hits": profile_result.get("hits", {}),
                    "took": profile_result.get("took", 0),
                    "explanations": {}
                }
                
                # Get explanations for top documents
                hits = profile_result.get("hits", {}).get("hits", [])
                for hit in hits[:3]:  # Explain top 3 documents
                    doc_id = hit["_id"]
                    try:
                        explanation = self.explain_query(
                            query=query_text,
                            document_id=doc_id,
                            index=index,
                            search_strategy=strategy,
                            filters=filters
                        )
                        strategy_result["explanations"][doc_id] = explanation
                    except Exception as e:
                        logger.warning(f"Failed to explain document {doc_id}: {e}")
                        continue
                
                results["strategies"][strategy] = strategy_result
                
            except Exception as e:
                logger.error(f"Failed to debug strategy {strategy}: {e}")
                results["strategies"][strategy] = {"error": str(e)}
        
        # Generate comparison insights
        results["comparison"] = self._compare_strategies(results["strategies"])
        
        return results
    
    def analyze_slow_queries(
        self,
        queries: List[str],
        threshold_ms: int = 1000,
        index: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze queries that exceed performance thresholds.
        
        Args:
            queries: List of queries to analyze
            threshold_ms: Performance threshold in milliseconds
            index: Index to search in
            
        Returns:
            Analysis of slow queries with optimization suggestions
        """
        slow_queries = []
        analysis = {
            "threshold_ms": threshold_ms,
            "total_queries": len(queries),
            "slow_queries": [],
            "performance_summary": {},
            "optimization_suggestions": []
        }
        
        for i, query in enumerate(queries):
            logger.info(f"Analyzing query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                # Profile the query
                profile_result = self.profile_query(
                    query=query,
                    index=index,
                    search_strategy="bm25",  # Use BM25 as baseline
                    size=10
                )
                
                took_ms = profile_result.get("took", 0)
                
                if took_ms > threshold_ms:
                    slow_query_analysis = {
                        "query": query,
                        "took_ms": took_ms,
                        "profile": profile_result.get("profile", {}),
                        "hits_total": profile_result.get("hits", {}).get("total", {}).get("value", 0)
                    }
                    
                    # Analyze bottlenecks
                    bottlenecks = self._analyze_query_bottlenecks(profile_result.get("profile", {}))
                    slow_query_analysis["bottlenecks"] = bottlenecks
                    
                    slow_queries.append(slow_query_analysis)
                
            except Exception as e:
                logger.error(f"Failed to analyze query '{query}': {e}")
                continue
        
        analysis["slow_queries"] = slow_queries
        analysis["performance_summary"] = self._generate_performance_summary(slow_queries)
        analysis["optimization_suggestions"] = self._generate_optimization_suggestions(slow_queries)
        
        return analysis
    
    def _build_query_for_strategy(
        self,
        query_text: str,
        strategy: str,
        filters: Optional[SearchFilters] = None
    ) -> Dict[str, Any]:
        """Build OpenSearch query based on strategy.
        
        Args:
            query_text: Query text
            strategy: Search strategy
            filters: Optional filters
            
        Returns:
            OpenSearch query dictionary
        """
        if strategy == "enhanced_rrf":
            return self.opensearch_client._build_hybrid_query(
                query=query_text,
                k=50,
                filters=filters
            )
        elif strategy == "bm25":
            return self.opensearch_client._build_simple_bm25_query(
                query=query_text,
                k=50,
                filters=filters
            )
        elif strategy == "knn":
            # For KNN, we need to generate embeddings first
            # This is a simplified version - in practice, you'd use the embedding service
            return {
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": [0.0] * 384,  # Placeholder vector
                            "k": 50
                        }
                    }
                }
            }
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
    
    def _compare_strategies(self, strategies_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance and results across strategies.
        
        Args:
            strategies_results: Results from different strategies
            
        Returns:
            Comparison analysis
        """
        comparison = {
            "performance": {},
            "result_overlap": {},
            "recommendations": []
        }
        
        # Performance comparison
        for strategy, result in strategies_results.items():
            if "error" not in result:
                comparison["performance"][strategy] = {
                    "took_ms": result.get("took", 0),
                    "total_hits": result.get("hits", {}).get("total", {}).get("value", 0),
                    "max_score": result.get("hits", {}).get("max_score", 0)
                }
        
        # Result overlap analysis
        if len(strategies_results) >= 2:
            strategy_names = list(strategies_results.keys())
            for i, strategy1 in enumerate(strategy_names):
                for strategy2 in strategy_names[i+1:]:
                    if "error" not in strategies_results[strategy1] and "error" not in strategies_results[strategy2]:
                        overlap = self._calculate_result_overlap(
                            strategies_results[strategy1].get("hits", {}).get("hits", []),
                            strategies_results[strategy2].get("hits", {}).get("hits", [])
                        )
                        comparison["result_overlap"][f"{strategy1}_vs_{strategy2}"] = overlap
        
        # Generate recommendations
        comparison["recommendations"] = self._generate_strategy_recommendations(comparison)
        
        return comparison
    
    def _calculate_result_overlap(
        self,
        hits1: List[Dict[str, Any]],
        hits2: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overlap between two sets of search results.
        
        Args:
            hits1: First set of hits
            hits2: Second set of hits
            
        Returns:
            Overlap analysis
        """
        ids1 = set(hit["_id"] for hit in hits1)
        ids2 = set(hit["_id"] for hit in hits2)
        
        intersection = ids1.intersection(ids2)
        union = ids1.union(ids2)
        
        return {
            "jaccard_similarity": len(intersection) / len(union) if union else 0,
            "overlap_count": len(intersection),
            "unique_to_first": len(ids1 - ids2),
            "unique_to_second": len(ids2 - ids1),
            "total_unique": len(union)
        }
    
    def _analyze_query_bottlenecks(self, profile: Dict[str, Any]) -> List[str]:
        """Analyze query profile to identify performance bottlenecks.
        
        Args:
            profile: OpenSearch profile data
            
        Returns:
            List of identified bottlenecks
        """
        bottlenecks = []
        
        # Analyze shard-level performance
        shards = profile.get("shards", [])
        for shard in shards:
            searches = shard.get("searches", [])
            for search in searches:
                query_info = search.get("query", [])
                
                # Check for expensive operations
                for query_component in query_info:
                    query_type = query_component.get("type", "")
                    time_ms = query_component.get("time_in_nanos", 0) / 1_000_000
                    
                    if time_ms > 100:  # More than 100ms
                        bottlenecks.append(f"Slow {query_type} query component: {time_ms:.1f}ms")
                    
                    # Check for specific problematic patterns
                    if query_type == "TermQuery" and time_ms > 50:
                        bottlenecks.append("Slow term query - consider using filters")
                    elif query_type == "BooleanQuery" and time_ms > 200:
                        bottlenecks.append("Complex boolean query - consider simplification")
                    elif "wildcard" in query_type.lower() and time_ms > 100:
                        bottlenecks.append("Expensive wildcard query - consider alternatives")
        
        return bottlenecks
    
    def _generate_performance_summary(self, slow_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate performance summary from slow query analysis.
        
        Args:
            slow_queries: List of slow query analyses
            
        Returns:
            Performance summary
        """
        if not slow_queries:
            return {"message": "No slow queries detected"}
        
        times = [q["took_ms"] for q in slow_queries]
        
        return {
            "slow_query_count": len(slow_queries),
            "avg_time_ms": sum(times) / len(times),
            "max_time_ms": max(times),
            "min_time_ms": min(times),
            "common_bottlenecks": self._find_common_bottlenecks(slow_queries)
        }
    
    def _find_common_bottlenecks(self, slow_queries: List[Dict[str, Any]]) -> List[str]:
        """Find common bottlenecks across slow queries.
        
        Args:
            slow_queries: List of slow query analyses
            
        Returns:
            List of common bottlenecks
        """
        bottleneck_counts = {}
        
        for query in slow_queries:
            for bottleneck in query.get("bottlenecks", []):
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
        
        # Return bottlenecks that appear in at least 25% of slow queries
        threshold = max(1, len(slow_queries) * 0.25)
        return [bottleneck for bottleneck, count in bottleneck_counts.items() if count >= threshold]
    
    def _generate_optimization_suggestions(self, slow_queries: List[Dict[str, Any]]) -> List[str]:
        """Generate optimization suggestions based on slow query analysis.
        
        Args:
            slow_queries: List of slow query analyses
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        if not slow_queries:
            return ["No optimization needed - all queries performing well"]
        
        # Analyze patterns in slow queries
        avg_time = sum(q["took_ms"] for q in slow_queries) / len(slow_queries)
        
        if avg_time > 2000:
            suggestions.append("Consider adding more specific filters to reduce search scope")
            suggestions.append("Review index mapping and consider field optimization")
        
        if len(slow_queries) > 5:
            suggestions.append("High number of slow queries detected - consider query optimization")
            suggestions.append("Review search strategy selection logic")
        
        # Check for common bottlenecks
        common_bottlenecks = self._find_common_bottlenecks(slow_queries)
        if "wildcard" in str(common_bottlenecks).lower():
            suggestions.append("Reduce wildcard query usage - use prefix queries or filters instead")
        
        if "boolean" in str(common_bottlenecks).lower():
            suggestions.append("Simplify complex boolean queries - break into multiple simpler queries")
        
        # General suggestions
        suggestions.extend([
            "Consider implementing query result caching for frequently used queries",
            "Monitor index size and consider archiving old documents",
            "Review field boosting values for optimal relevance vs performance balance"
        ])
        
        return suggestions
    
    def _generate_strategy_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on strategy comparison.
        
        Args:
            comparison: Strategy comparison data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        performance = comparison.get("performance", {})
        if not performance:
            return recommendations
        
        # Find fastest strategy
        fastest_strategy = min(performance.items(), key=lambda x: x[1].get("took_ms", float('inf')))
        slowest_strategy = max(performance.items(), key=lambda x: x[1].get("took_ms", 0))
        
        if len(performance) > 1:
            recommendations.append(
                f"Fastest strategy: {fastest_strategy[0]} ({fastest_strategy[1]['took_ms']}ms)"
            )
            
            time_diff = slowest_strategy[1]['took_ms'] - fastest_strategy[1]['took_ms']
            if time_diff > 500:  # Significant difference
                recommendations.append(
                    f"Consider using {fastest_strategy[0]} for better performance "
                    f"({time_diff}ms improvement)"
                )
        
        # Analyze result overlap
        result_overlap = comparison.get("result_overlap", {})
        for comparison_key, overlap_data in result_overlap.items():
            jaccard = overlap_data.get("jaccard_similarity", 0)
            if jaccard < 0.3:
                recommendations.append(
                    f"Low result similarity between {comparison_key.replace('_vs_', ' and ')} "
                    f"(Jaccard: {jaccard:.2f}) - review strategy selection criteria"
                )
            elif jaccard > 0.8:
                recommendations.append(
                    f"High result similarity between {comparison_key.replace('_vs_', ' and ')} "
                    f"(Jaccard: {jaccard:.2f}) - consider using faster strategy only"
                )
        
        return recommendations
    
    def save_debug_session(
        self,
        session_data: Dict[str, Any],
        output_path: str
    ) -> None:
        """Save debug session data to file.
        
        Args:
            session_data: Debug session data
            output_path: Path to save session data
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            logger.info(f"Debug session saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save debug session: {e}")
            raise