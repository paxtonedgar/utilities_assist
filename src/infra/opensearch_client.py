# src/infra/opensearch_client.py
"""
OpenSearch client with enterprise-grade search capabilities.

All filters must be applied to both BM25 and kNN to avoid leakage.

Features:
- Pooled session with JPMC authentication and proxy support
- BM25 search with ACL filters and time decay
- kNN vector search with ef_search optimization
- Pure Python RRF fusion for hybrid search
- Comprehensive filtering (ACL, space, time ranges)
- Function score time decay for recency boosting
"""

import json
import logging
import math
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

import requests

from src.infra.config import SearchCfg
from src.infra.clients import make_search_session

logger = logging.getLogger(__name__)


@dataclass
class SearchFilters:
    """Search filters for ACL, space, and time-based filtering."""
    acl_hash: Optional[str] = None
    space_key: Optional[str] = None
    content_type: Optional[str] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None


@dataclass 
class SearchResult:
    """Individual search result with score and metadata."""
    doc_id: str
    score: float
    title: str
    body: str
    metadata: Dict[str, Any]
    
    
@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    results: List[SearchResult]
    total_hits: int
    took_ms: int
    method: str


class OpenSearchClient:
    """Production-ready OpenSearch client with enterprise features."""
    
    def __init__(self, config: SearchCfg):
        self.config = config
        self.session = make_search_session(config)
        self.base_url = config.host.rstrip('/')
        
    def bm25_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        index: str = "confluence_current",
        k: int = 50,
        time_decay_half_life_days: int = 120
    ) -> SearchResponse:
        """
        BM25 full-text search with ACL filters and time decay.
        
        Args:
            query: Search query string
            filters: ACL, space, and time filters
            index: Index name or alias to search
            k: Number of results to return
            time_decay_half_life_days: Half-life for time decay in days
            
        Returns:
            SearchResponse with BM25 results
        """
        start_time = time.time()
        
        # Build simple BM25 query (like main branch)
        search_body = self._build_simple_bm25_query(query, k)
        
        # Debug: log the exact query being sent
        logger.info(f"OpenSearch BM25 query: {json.dumps(search_body, indent=2)}")
        
        try:
            url = f"{self.base_url}/{index}/_search"
            
            # Use direct requests with auth (like main branch) instead of session
            from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
            _setup_jpmc_proxy()  # Ensure proxy is configured
            aws_auth = _get_aws_auth()
            if aws_auth:
                logger.info("Using direct AWS4Auth for BM25 OpenSearch request")
                response = requests.post(url, json=search_body, auth=aws_auth, timeout=self.config.timeout_s)
            else:
                logger.warning("No AWS auth available, using session without auth for BM25")
                response = self.session.post(url, json=search_body, timeout=self.config.timeout_s)
            
            response.raise_for_status()
            
            data = response.json()
            results = self._parse_search_response(data)
            
            return SearchResponse(
                results=results,
                total_hits=data["hits"]["total"]["value"],
                took_ms=int((time.time() - start_time) * 1000),
                method="bm25"
            )
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return SearchResponse(results=[], total_hits=0, took_ms=0, method="bm25")
    
    def knn_search(
        self,
        query_vector: List[float],
        filters: Optional[SearchFilters] = None,
        index: str = "confluence_current", 
        k: int = 50,
        ef_search: int = 256,
        time_decay_half_life_days: int = 120
    ) -> SearchResponse:
        """
        kNN vector similarity search with ACL filters and time decay.
        
        Args:
            query_vector: Query embedding vector (1536 dims)
            filters: ACL, space, and time filters
            index: Index name or alias to search
            k: Number of results to return
            ef_search: ef_search parameter for HNSW (trade-off: accuracy vs speed)
            time_decay_half_life_days: Half-life for time decay in days (uniform with BM25)
            
        Returns:
            SearchResponse with kNN results
        """
        start_time = time.time()
        
        # Build simple kNN query (like main branch)
        search_body = self._build_simple_knn_query(query_vector, k)
        
        # Debug: log the exact kNN query being sent
        logger.info(f"OpenSearch kNN query: {json.dumps(search_body, indent=2)}")
        
        try:
            url = f"{self.base_url}/{index}/_search"
            
            # Use direct requests with auth (like main branch) instead of session
            from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
            _setup_jpmc_proxy()  # Ensure proxy is configured
            aws_auth = _get_aws_auth()
            if aws_auth:
                logger.info("Using direct AWS4Auth for kNN OpenSearch request")
                response = requests.post(url, json=search_body, auth=aws_auth, timeout=self.config.timeout_s)
            else:
                logger.warning("No AWS auth available, using session without auth for kNN")
                response = self.session.post(url, json=search_body, timeout=self.config.timeout_s)
            
            response.raise_for_status()
            
            data = response.json()
            results = self._parse_search_response(data)
            
            return SearchResponse(
                results=results,
                total_hits=data["hits"]["total"]["value"],
                took_ms=int((time.time() - start_time) * 1000),
                method="knn"
            )
            
        except Exception as e:
            logger.error(f"kNN search failed: {e}")
            return SearchResponse(results=[], total_hits=0, took_ms=0, method="knn")
    
    def rrf_fuse(
        self,
        bm25_response: SearchResponse,
        knn_response: SearchResponse,
        k: int = 8,
        rrf_k: int = 60
    ) -> SearchResponse:
        """
        Reciprocal Rank Fusion (RRF) for hybrid search.
        
        Pure Python implementation combining BM25 and kNN results.
        
        Args:
            bm25_response: BM25 search results
            knn_response: kNN search results
            k: Final number of results to return
            rrf_k: RRF constant (typically 60, higher = less aggressive fusion)
            
        Returns:
            SearchResponse with fused results
        """
        start_time = time.time()
        
        # Create rank maps for both result sets
        bm25_ranks = {result.doc_id: idx + 1 for idx, result in enumerate(bm25_response.results)}
        knn_ranks = {result.doc_id: idx + 1 for idx, result in enumerate(knn_response.results)}
        
        # Create document map for result data
        doc_map = {}
        for result in bm25_response.results:
            doc_map[result.doc_id] = result
        for result in knn_response.results:
            doc_map[result.doc_id] = result
        
        # Calculate RRF scores
        rrf_scores = {}
        all_doc_ids = set(bm25_ranks.keys()) | set(knn_ranks.keys())
        
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            
            # Add BM25 contribution: 1 / (k + rank)
            if doc_id in bm25_ranks:
                rrf_score += 1.0 / (rrf_k + bm25_ranks[doc_id])
            
            # Add kNN contribution: 1 / (k + rank)
            if doc_id in knn_ranks:
                rrf_score += 1.0 / (rrf_k + knn_ranks[doc_id])
            
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score (descending) and take top k
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Build final results with RRF scores
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            if doc_id in doc_map:
                result = doc_map[doc_id]
                # Create new result with RRF score
                fused_result = SearchResult(
                    doc_id=result.doc_id,
                    score=rrf_score,
                    title=result.title,
                    body=result.body,
                    metadata=result.metadata
                )
                fused_results.append(fused_result)
        
        logger.info(f"RRF fusion: {len(bm25_response.results)} BM25 + {len(knn_response.results)} kNN → {len(fused_results)} fused")
        
        return SearchResponse(
            results=fused_results,
            total_hits=len(fused_results),
            took_ms=int((time.time() - start_time) * 1000) + bm25_response.took_ms + knn_response.took_ms,
            method="rrf"
        )
    
    def _extract_key_terms(self, query_text: str) -> str:
        """Extract key terms from natural language queries."""
        import re
        # Remove question words and common stopwords
        stopwords = {'what', 'how', 'do', 'i', 'is', 'are', 'the', 'to', 'for', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'with', 'from', 'me', 'through', 'of'}
        
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', '', query_text.lower())
        
        # Split into words and filter stopwords
        words = [word for word in cleaned.split() if word not in stopwords and len(word) > 2]
        
        # Return key terms as space-separated string
        return ' '.join(words)

    def _build_bm25_query(
        self,
        query: str,
        filters: Optional[SearchFilters],
        k: int,
        time_decay_half_life_days: int
    ) -> Dict[str, Any]:
        """Build tuned BM25 query with multi_match, dynamic minimum_should_match, and phrase boosting."""
        
        # Extract key terms from natural language query
        key_terms = self._extract_key_terms(query)
        
        # Extract key phrases for proximity boosting (from original query)
        key_phrases = self._extract_key_phrases(query)
        
        # Use extracted key terms for main query with moderate boosts
        # Build multi_match query with reduced field boosts for better recall  
        must_clauses = [
            {
                "multi_match": {
                    "query": key_terms,  # Use extracted key terms instead of full query
                    "type": "best_fields",
                    "fields": ["title^5", "section^2", "body^1"],  # Reduced boosts: 10→5, 4→2
                    "tie_breaker": 0.3
                }
            }
        ]
        
        # Add gentle proximity boosts without strict phrase requirements
        should_clauses = []
        
        # Gentle title proximity matching (not strict phrase)
        should_clauses.append({
            "match": {
                "title": {
                    "query": key_terms,
                    "boost": 2  # Reduced from 6 to 2
                }
            }
        })
        
        # Key term proximity boosting in body (not strict phrases)
        should_clauses.append({
            "match": {
                "body": {
                    "query": key_terms,
                    "boost": 1.5  # Gentle boost for body matches
                }
            }
        })
        
        # Build base query structure
        base_query = {
            "bool": {
                "must": must_clauses,
                "should": should_clauses,
                "minimum_should_match": 1  # At least one must clause required
            }
        }
        
        # Apply filters to query
        filter_clauses = []
        if filters:
            filter_clauses = self._build_filter_clauses(filters)
        
        # Wrap in bool query for filters
        if filter_clauses:
            bool_query = {
                "bool": {
                    "must": [base_query],
                    "filter": filter_clauses
                }
            }
        else:
            bool_query = {
                "bool": {
                    "must": [base_query]
                }
            }
        
        # Apply function_score with time decay and generic doc penalties
        functions = []
        
        # Gentle time decay function - capped to avoid suppressing authoritative older docs
        if time_decay_half_life_days > 0:
            # Use longer half-life and gentler decay to preserve older authoritative content
            functions.append({
                "exp": {
                    "updated_at": {
                        "scale": "90d",  # Longer half-life: 75d → 90d  
                        "decay": 0.6     # Gentler decay: 0.4 → 0.6
                    }
                },
                "weight": 1.1  # Reduced recency weight: 1.2 → 1.1
            })
        
        # Generic document penalty - use positive weights < 1.0 for penalties
        functions.append({
            "filter": {
                "terms": {
                    "section": ["global", "overview", "platform", "general", "introduction", "welcome"]
                }
            },
            "weight": 0.3  # Penalty via low positive weight (instead of negative)
        })
        
        # Enhanced generic title penalty with more patterns
        functions.append({
            "filter": {
                "regexp": {
                    "title": ".*(Overview|Introduction|Welcome|Platform|Global|General).*"
                }
            },
            "weight": 0.5  # Penalty via low positive weight (instead of negative)
        })
        
        final_query = {
            "function_score": {
                "query": bool_query,
                "boost_mode": "multiply",  # Changed from "sum" to "multiply" for penalty weights
                "functions": functions
            }
        }
        
        return {
            "query": final_query,
            "size": k,
            "sort": [{"_score": {"order": "desc"}}],
            "_source": ["title", "body", "metadata", "updated_at", "page_id", "canonical_id", "section"],
            "track_scores": True
        }
    
    def _extract_key_phrases(self, query: str, max_phrases: int = 3) -> List[str]:
        """Extract key phrases from query for proximity boosting."""
        # Simple stopword list
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you',
            'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        # Tokenize and clean query
        words = re.findall(r'\b\w+\b', query.lower())
        words = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Generate bigrams as key phrases
        key_phrases = []
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            key_phrases.append(bigram)
        
        # Also include important single terms
        for word in words:
            if len(word) > 4:  # Longer words are likely more important
                key_phrases.append(word)
        
        # Return top phrases (prioritize bigrams)
        return key_phrases[:max_phrases]
    
    def _build_knn_query(
        self,
        query_vector: List[float],
        filters: Optional[SearchFilters],
        k: int,
        ef_search: int,
        time_decay_half_life_days: int = 120
    ) -> Dict[str, Any]:
        """Build kNN query with ACL filtering, time decay, and optimized parameters."""
        
        # Build filter clauses
        filter_clauses = []
        if filters:
            filter_clauses = self._build_filter_clauses(filters)
        
        # Base kNN query structure
        base_query = {
            "size": k,
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": k,
                "num_candidates": max(200, k * 4)  # Ensure good candidate pool
            },
            "_source": ["title", "body", "metadata", "updated_at", "page_id", "canonical_id", "section"],
            "track_scores": True
        }
        
        # Apply filters if present
        if filter_clauses:
            base_query["query"] = {
                "bool": {
                    "filter": filter_clauses
                }
            }
        else:
            base_query["query"] = {"match_all": {}}
        
        # Apply time decay function score (uniform with BM25)
        if time_decay_half_life_days > 0:
            # Wrap the query in function_score for time decay
            functions = [{
                "exp": {
                    "updated_at": {
                        "scale": "75d",  # Same as BM25 tuned mode
                        "decay": 0.4     # Same decay factor as BM25
                    }
                },
                "weight": 1.2  # Same weight as BM25
            }]
            
            # Generic document penalty (same as BM25) - use positive weights < 1.0
            functions.extend([
                {
                    "filter": {
                        "terms": {
                            "section": ["global", "overview", "platform", "general", "introduction", "welcome"]
                        }
                    },
                    "weight": 0.3  # Penalty via low positive weight (consistent with BM25)
                },
                {
                    "filter": {
                        "regexp": {
                            "title": ".*(Overview|Introduction|Welcome|Platform|Global|General).*"
                        }
                    },
                    "weight": 0.5  # Penalty via low positive weight (consistent with BM25)
                }
            ])
            
            base_query["query"] = {
                "function_score": {
                    "query": base_query["query"],
                    "boost_mode": "multiply",  # Changed from "sum" to "multiply" for penalty weights
                    "functions": functions
                }
            }
        
        return base_query
    
    def _build_filter_clauses(self, filters: SearchFilters) -> List[Dict[str, Any]]:
        """Build filter clauses from SearchFilters."""
        clauses = []
        
        # ACL hash filter
        if filters.acl_hash:
            clauses.append({
                "term": {"acl_hash": filters.acl_hash}
            })
        
        # Space key filter
        if filters.space_key:
            clauses.append({
                "term": {"metadata.space_key": filters.space_key}
            })
        
        # Content type filter
        if filters.content_type:
            clauses.append({
                "term": {"content_type": filters.content_type}
            })
        
        # Time range filters - use consistent YYYY-MM-DD format
        if filters.updated_after or filters.updated_before:
            range_filter = {"range": {"updated_at": {}}}
            
            if filters.updated_after:
                range_filter["range"]["updated_at"]["gte"] = filters.updated_after.strftime('%Y-%m-%d')
            
            if filters.updated_before:
                range_filter["range"]["updated_at"]["lte"] = filters.updated_before.strftime('%Y-%m-%d')
            
            clauses.append(range_filter)
        
        return clauses
    
    def _parse_search_response(self, data: Dict[str, Any]) -> List[SearchResult]:
        """Parse OpenSearch response into SearchResult objects."""
        results = []
        
        for hit in data.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            
            result = SearchResult(
                doc_id=hit["_id"],
                score=hit["_score"],
                title=source.get("title", ""),
                body=source.get("body", ""),
                metadata=source.get("metadata", {})
            )
            results.append(result)
        
        return results
    
    def _build_simple_bm25_query(self, query: str, k: int) -> Dict[str, Any]:
        """Build simple BM25 query exactly like main branch - no complex features."""
        # Simple multi_match query like main branch
        search_body = {
            "size": k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "title"],  # Use main branch field names
                    "boost": 1.0
                }
            }
        }
        return search_body
    
    def _build_simple_knn_query(self, query_vector: List[float], k: int) -> Dict[str, Any]:
        """Build simple kNN query exactly like main branch - no complex features."""
        # Simple kNN query like main branch
        search_body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            }
        }
        return search_body
    
    def health_check(self) -> Dict[str, Any]:
        """Check OpenSearch cluster health and connectivity."""
        try:
            # Test basic connectivity
            url = f"{self.base_url}/_cluster/health"
            response = self.session.get(url, timeout=5.0)
            response.raise_for_status()
            
            health = response.json()
            
            # Test index existence
            alias_url = f"{self.base_url}/confluence_current"
            alias_response = self.session.head(alias_url, timeout=5.0)
            index_exists = alias_response.status_code == 200
            
            return {
                "status": "healthy" if health["status"] in ["green", "yellow"] else "unhealthy",
                "cluster_name": health["cluster_name"],
                "cluster_status": health["status"],
                "node_count": health["number_of_nodes"],
                "data_nodes": health["number_of_data_nodes"],
                "index_exists": index_exists,
                "authentication": "configured" if self.session.auth else "none"
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "authentication": "configured" if self.session.auth else "none"
            }


def create_search_client(config: SearchCfg) -> OpenSearchClient:
    """Factory function to create OpenSearch client with proper configuration."""
    return OpenSearchClient(config)