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

from src.infra.settings import get_settings
from src.telemetry.logger import log_event, stage
from services.models import SearchResult as ServiceSearchResult  # Use service model

logger = logging.getLogger(__name__)


def get_total_hits(response_data: Dict[str, Any]) -> int:
    """
    Safely extract total hits count from OpenSearch response.
    
    Handles both formats:
    - Legacy: {"hits": {"total": 42}}
    - Modern: {"hits": {"total": {"value": 42, "relation": "eq"}}}
    """
    try:
        hits_section = response_data.get("hits", {})
        total = hits_section.get("total", 0)
        
        if isinstance(total, int):
            return total
        elif isinstance(total, dict):
            return total.get("value", 0)
        else:
            return 0
    except (KeyError, TypeError, AttributeError):
        return 0


@dataclass
class SearchFilters:
    """Search filters for ACL, space, and time-based filtering."""
    acl_hash: Optional[str] = None
    space_key: Optional[str] = None
    content_type: Optional[str] = None
    updated_after: Optional[datetime] = None
    updated_before: Optional[datetime] = None


@dataclass
class SearchResponse:
    """Search response with results and metadata."""
    results: List[ServiceSearchResult]  # Use the service model
    total_hits: int
    took_ms: int
    method: str


class OpenSearchClient:
    """Production-ready OpenSearch client with enterprise features."""
    
    def __init__(self, settings: Optional[object] = None):
        """Initialize OpenSearch client with centralized settings."""
        if settings is None:
            settings = get_settings()
        
        self.settings = settings
        self.base_url = settings.opensearch_host.rstrip('/')
        
        # Create session with proper authentication for the current profile
        if settings.requires_aws_auth:
            # Will use AWS authentication via clients module
            from src.infra.clients import _setup_jpmc_proxy
            _setup_jpmc_proxy()
            self.session = None  # Use direct requests with AWS auth
        else:
            # Local development - create simple session
            import requests
            self.session = requests.Session()
            self.session.headers.update({'Content-Type': 'application/json'})
        
    @stage("bm25")
    def bm25_search(
        self,
        query: str,
        filters: Optional[SearchFilters] = None,
        index: Optional[str] = None,
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
        # Use configured index alias if not specified
        if index is None:
            index = self.settings.search_index_alias
        
        # Log search start
        log_event(
            stage="bm25",
            event="start",
            index=index,
            query_type="simple_match",
            k=k,
            filters_enabled=filters is not None,
            query_length=len(query)
        )
        
        # Build BM25 query matching working v1 branch structure
        search_body = self._build_simple_bm25_query(query, k)
        
        try:
            url = f"{self.base_url}/{index}/_search"
            start_time = time.time()
            
            # EXPLICIT OS QUERY LOG - for debugging query preservation
            logger.info(
                "OS_QUERY index=%s strategy=%s q=%r query_len=%d",
                index, "bm25", query, len(query)
            )
            
            # Use GET request like working v1 branch
            from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
            _setup_jpmc_proxy()  # Ensure proxy is configured
            aws_auth = _get_aws_auth()
            if aws_auth:
                response = requests.get(url, json=search_body, auth=aws_auth, timeout=30.0)
            else:
                response = self.session.get(url, json=search_body, timeout=30.0)
            
            took_ms = (time.time() - start_time) * 1000
            status_code = response.status_code
            
            # Log HTTP response details
            log_event(
                stage="bm25", 
                event="http_response",
                status=status_code,
                took_ms=took_ms,
                index=index
            )
            
            # Handle HTTP errors
            if not response.ok:
                error_body = response.text[:500]  # Snippet only
                log_event(
                    stage="bm25",
                    event="error", 
                    status=status_code,
                    took_ms=took_ms,
                    err=True,
                    error_type="HTTPError",
                    error_message=f"HTTP {status_code}",
                    error_body_snippet=error_body,
                    index=index
                )
                response.raise_for_status()
            
            data = response.json()
            results = self._parse_search_response(data)
            
            # EXPLICIT OS RESPONSE LOG - for debugging query results
            top_result_title = results[0].title if results else None
            logger.info(
                "OS_RESPONSE index=%s took_ms=%.1f status=%s hits=%d top=%r",
                index, took_ms, status_code, len(results), top_result_title
            )
            
            # Log successful completion
            total_hits = get_total_hits(data)
            log_event(
                stage="bm25",
                event="success",
                took_ms=took_ms,
                result_count=len(results),
                total_hits=total_hits,
                index=index,
                elasticsearch_took=data.get("took", 0)
            )
            
            return SearchResponse(
                results=results,
                total_hits=total_hits,
                took_ms=int(took_ms),
                method="bm25"
            )
            
        except requests.exceptions.HTTPError as e:
            # Special handling for 404 - index not found
            if hasattr(e.response, 'status_code') and e.response.status_code == 404:
                logger.warning(f"Index not found: {index}. Continuing with empty results.")
                log_event(
                    stage="bm25",
                    event="index_not_found",
                    index=index,
                    status_code=404,
                    message=f"Index {index} not found, returning empty results"
                )
            else:
                log_event(
                    stage="bm25",
                    event="error",
                    err=True,
                    error_type="HTTPError",
                    error_message=str(e)[:200],
                    index=index,
                    status_code=e.response.status_code if hasattr(e.response, 'status_code') else None
                )
            
            return SearchResponse(results=[], total_hits=0, took_ms=0, method="bm25")
            
        except Exception as e:
            # Log other errors
            log_event(
                stage="bm25",
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                index=index,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None
            )
            
            return SearchResponse(results=[], total_hits=0, took_ms=0, method="bm25")
    
    @stage("knn")
    def knn_search(
        self,
        query_vector: List[float],
        filters: Optional[SearchFilters] = None,
        index: Optional[str] = None, 
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
        # Use configured index alias if not specified
        if index is None:
            index = self.settings.search_index_alias
        
        # Log search start
        log_event(
            stage="knn",
            event="start",
            index=index,
            query_type="vector",
            k=k,
            vector_dims=len(query_vector),
            ef_search=ef_search,
            filters_enabled=filters is not None
        )
        
        # Build simple kNN query (like main branch)
        search_body = self._build_simple_knn_query(query_vector, k)
        
        try:
            url = f"{self.base_url}/{index}/_search"
            start_time = time.time()
            
            # Use direct requests with auth (like main branch) instead of session
            from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
            _setup_jpmc_proxy()  # Ensure proxy is configured
            aws_auth = _get_aws_auth()
            if aws_auth:
                response = requests.post(url, json=search_body, auth=aws_auth, timeout=30.0)
            else:
                response = self.session.post(url, json=search_body, timeout=30.0)
            
            took_ms = (time.time() - start_time) * 1000
            status_code = response.status_code
            
            # Log HTTP response details
            log_event(
                stage="knn",
                event="http_response", 
                status=status_code,
                took_ms=took_ms,
                index=index
            )
            
            # Handle HTTP errors
            if not response.ok:
                error_body = response.text[:500]  # Snippet only
                log_event(
                    stage="knn",
                    event="error",
                    status=status_code,
                    took_ms=took_ms,
                    err=True,
                    error_type="HTTPError",
                    error_message=f"HTTP {status_code}",
                    error_body_snippet=error_body,
                    index=index
                )
                response.raise_for_status()
            
            data = response.json()
            results = self._parse_search_response(data)
            
            # Log successful completion
            total_hits = get_total_hits(data)
            log_event(
                stage="knn",
                event="success",
                took_ms=took_ms,
                result_count=len(results),
                total_hits=total_hits,
                index=index,
                elasticsearch_took=data.get("took", 0)
            )
            
            return SearchResponse(
                results=results,
                total_hits=total_hits,
                took_ms=int(took_ms),
                method="knn"
            )
            
        except requests.exceptions.HTTPError as e:
            # Special handling for 404 - index not found
            if hasattr(e.response, 'status_code') and e.response.status_code == 404:
                logger.warning(f"Index not found: {index}. Continuing with empty results.")
                log_event(
                    stage="knn",
                    event="index_not_found",
                    index=index,
                    status_code=404,
                    message=f"Index {index} not found, returning empty results"
                )
            else:
                log_event(
                    stage="knn",
                    event="error",
                    err=True,
                    error_type="HTTPError",
                    error_message=str(e)[:200],
                    index=index,
                    status_code=e.response.status_code if hasattr(e.response, 'status_code') else None
                )
            
            return SearchResponse(results=[], total_hits=0, took_ms=0, method="knn")
            
        except Exception as e:
            # Log other errors
            log_event(
                stage="knn",
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:200],
                index=index,
                status_code=getattr(response, 'status_code', None) if 'response' in locals() else None
            )
            
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
                fused_result = ServiceSearchResult(
                    doc_id=result.doc_id,
                    score=rrf_score,
                    content=result.content,  # Use content field
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
    
    @stage("hybrid")
    def hybrid_search(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        filters: Optional[SearchFilters] = None,
        index: Optional[str] = None,
        k: int = 50,
        time_decay_half_life_days: int = 120
    ) -> SearchResponse:
        """
        True hybrid search combining BM25 and KNN in a single query.
        Falls back to BM25-only if no vector is provided or if KNN fails.
        
        Args:
            query: Search query string
            query_vector: Optional query embedding vector
            filters: ACL, space, and time filters
            index: Index name or alias to search
            k: Number of results to return
            time_decay_half_life_days: Half-life for time decay in days
            
        Returns:
            SearchResponse with hybrid search results
        """
        # Use configured index alias if not specified
        if index is None:
            index = self.settings.search_index_alias
        
        # Log search start
        log_event(
            stage="hybrid",
            event="start",
            index=index,
            query_type="hybrid" if query_vector else "bm25_fallback",
            k=k,
            filters_enabled=filters is not None,
            query_length=len(query),
            has_vector=query_vector is not None
        )
        
        # Build hybrid query
        search_body = self._build_hybrid_query(
            query=query,
            query_vector=query_vector,
            k=k
        )
        
        try:
            url = f"{self.base_url}/{index}/_search"
            start_time = time.time()
            
            logger.info(
                "OS_QUERY index=%s strategy=%s q=%r query_len=%d has_vector=%s",
                index, "hybrid", query, len(query), query_vector is not None
            )
            
            # DEBUG: Log the actual query body for troubleshooting
            logger.info(f"HYBRID_QUERY_BODY: {json.dumps(search_body, indent=2)}")
            
            from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
            _setup_jpmc_proxy()
            
            try:
                response = requests.get(
                    url,
                    auth=_get_aws_auth() if self.settings.requires_aws_auth else None,
                    json=search_body,
                    timeout=30
                )
                
                # Log HTTP response details before raising for status
                logger.info(f"HYBRID_HTTP_RESPONSE: status={response.status_code} took={time.time() - start_time:.3f}s")
                
                response.raise_for_status()
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Hybrid search HTTP request failed: {req_err}")
                logger.error(f"Request URL: {url}")
                logger.error(f"Response status: {getattr(response, 'status_code', 'N/A')}")
                logger.error(f"Response text: {getattr(response, 'text', 'N/A')[:500]}")
                raise
            
            data = response.json()
            took_ms = int((time.time() - start_time) * 1000)
            
            # Log success with detailed result info
            total_hits = get_total_hits(data)
            
            logger.info(
                "OS_RESPONSE index=%s strategy=hybrid status=200 hits=%d took_ms=%d",
                index, total_hits, took_ms
            )
            
            log_event(
                stage="hybrid",
                event="success",
                index=index,
                total_hits=total_hits,
                took_ms=took_ms,
                has_vector=query_vector is not None
            )
            
            # Parse results - handle both root and nested content
            results = self._parse_search_response(data)
            
            return SearchResponse(
                results=results,
                total_hits=total_hits,
                took_ms=took_ms,
                method="hybrid" if query_vector else "bm25_fallback"
            )
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            log_event(
                stage="hybrid",
                event="error",
                err=True,
                error_type=type(e).__name__,
                error_message=str(e)[:100]
            )
            
            # If hybrid fails and we have no vector, fallback to simple BM25
            if not query_vector:
                logger.info("Falling back to simple BM25 search")
                return self.bm25_search(query, filters, index, k, time_decay_half_life_days)
            
            return SearchResponse(
                results=[],
                total_hits=0,
                took_ms=int((time.time() - start_time) * 1000),
                method="hybrid_error"
            )
    
    def _build_hybrid_query(
        self,
        query: str,
        query_vector: Optional[List[float]] = None,
        k: int = 50
    ) -> Dict[str, Any]:
        """
        Build a hybrid query combining BM25 and KNN in a single bool.should query.
        This follows the fix pattern for root-level hybrid search.
        """
        from agent.acronym_map import expand_acronym
        
        # Expand acronyms if present
        expanded_query, expansions = expand_acronym(query)
        
        # Build should clauses for hybrid search
        should_clauses = []
        
        # BM25 clause - multi_match over root fields
        should_clauses.append({
            "multi_match": {
                "query": expanded_query if expansions else query,
                "fields": [
                    "content^3",
                    "body^3",
                    "text^2",
                    "title^4",
                    "name^2"
                ],
                "type": "best_fields"
            }
        })
        
        # KNN clause - only if vector is provided
        if query_vector:
            should_clauses.append({
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": k
                    }
                }
            })
        
        # Build complete query with namespace filter
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1,
                    "filter": [
                        {
                            "bool": {
                                "should": [
                                    {"term": {"space.keyword": "Utilities"}},
                                    {"term": {"labels.keyword": "utilities"}},
                                    {"wildcard": {"path.keyword": "*/utilities/*"}},
                                    {"prefix": {"title.keyword": "utilities/"}}
                                ],
                                "minimum_should_match": 1
                            }
                        }
                    ]
                }
            },
            "_source": [
                "title", "name", "page_url", "path",
                "content", "body", "text",
                # Include sections in case some docs have it
                "sections", "sections.heading", "sections.content"
            ],
            "highlight": {
                "fields": {
                    "content": {},
                    "body": {},
                    "text": {}
                },
                "fragment_size": 160,
                "number_of_fragments": 2
            }
        }
        
        return search_body
    
    def _extract_key_terms(self, query_text: str) -> str:
        """Extract key terms from natural language queries, optimized for BM25-friendly keywords.
        
        Strips verbose LLM narrations to compact entities/keywords as per user requirements.
        """
        import re
        
        if not query_text or not query_text.strip():
            return ""
        
        # Normalize the text
        text = query_text.lower().strip()
        
        # Remove common LLM narration patterns that hurt BM25 performance
        llm_patterns = [
            r'\b(tell me about|what is|explain|describe|show me|help me understand|i need to know|can you|please)\b',
            r'\b(how do i|how to|steps to|process for|way to)\b',
            r'\b(i want to|i would like to|i need to|let me)\b',
            r'\b(the user is asking|the question is|this query is about)\b'
        ]
        
        for pattern in llm_patterns:
            text = re.sub(pattern, '', text)
        
        # Enhanced stopwords including verbose terms that don't help BM25
        expanded_stopwords = {
            'what', 'how', 'do', 'i', 'is', 'are', 'the', 'to', 'for', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'with', 'from', 'me', 'through', 'of', 'this', 'that', 'these', 'those',
            'about', 'can', 'could', 'would', 'should', 'will', 'was', 'were', 'been', 'have', 'has', 'had', 'does', 'did', 'get', 'got', 'use', 'using', 'used',
            'tell', 'show', 'give', 'find', 'see', 'know', 'help', 'want', 'need', 'like', 'please', 'understand', 'explain', 'describe'
        }
        
        # Clean punctuation but preserve important separators temporarily
        text = re.sub(r'[^\w\s\-\_]', ' ', text)
        
        # Split into words and filter
        words = []
        for word in text.split():
            word = word.strip('-_')  # Clean separators from edges
            if (len(word) > 2 and 
                word not in expanded_stopwords and 
                not word.isdigit() and  # Skip pure numbers
                re.match(r'^[a-zA-Z][a-zA-Z0-9]*$', word)):  # Keep alphanumeric starting with letter
                words.append(word)
        
        # Prioritize potential business terms and acronyms
        priority_words = []
        regular_words = []
        
        for word in words:
            # Prioritize: uppercase terms, compound terms with separators, longer terms
            if (word.upper() == word and len(word) > 1) or len(word) > 6:
                priority_words.append(word)
            else:
                regular_words.append(word)
        
        # Combine priority words first, then regular words
        result_words = priority_words + regular_words
        
        # Return top keywords to avoid over-long queries
        return ' '.join(result_words[:8])  # Limit to 8 key terms for BM25 performance

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
                "field": "embedding",  # Use root-level embedding field
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
    
    def _parse_search_response(self, data: Dict[str, Any]) -> List[ServiceSearchResult]:
        """Parse OpenSearch response into SearchResult objects with canonical schema."""
        results = []
        
        # Safely extract hits array
        hits = data.get("hits", {}).get("hits", [])
        if not isinstance(hits, list):
            logger.warning("Invalid hits format in OpenSearch response")
            return []
        
        for hit in hits:
            source = hit.get("_source", {})
            
            # Extract title with fallbacks - REQUIRED field in canonical schema
            title = (source.get("api_name") or 
                    source.get("title") or 
                    source.get("utility_name") or 
                    f"Document {len(results)+1}")  # Always provide a title
            
            # Handle nested structure with inner_hits (like v1 working branch)
            body_parts = []
            
            # Extract content from inner_hits (matched sections) with section metadata
            inner_hits = hit.get("inner_hits", {}).get("matched_sections", {}).get("hits", {}).get("hits", [])
            section_paths = []
            anchors = []
            
            # If we have inner_hits, use nested content structure
            if inner_hits:
                for section_hit in inner_hits:
                    section_source = section_hit.get("_source", {})
                    heading = section_source.get("heading", "")
                    content = section_source.get("content", "")
                    anchor = section_source.get("anchor", "")  # URL anchor/slug
                    section_path = section_source.get("section_path", "")  # e.g., "CIU > Onboarding > Create Client IDs"
                    
                    if content:
                        section_text = f"{heading}\n{content}" if heading else content
                        body_parts.append(section_text)
                        
                        # Track section metadata for actionable answers
                        if section_path:
                            section_paths.append(section_path)
                        if anchor:
                            anchors.append(anchor)
            else:
                # Fallback: Extract from root-level fields (for hybrid search results)
                content_fields = ['content', 'body', 'text', 'description']
                for field in content_fields:
                    if field in source and source[field]:
                        body_parts.append(source[field])
                        break  # Use the first available content field
                
                # Also check if there are sections array at root level
                sections = source.get('sections', [])
                if sections and isinstance(sections, list):
                    for section in sections[:3]:  # Limit to first 3 sections
                        if isinstance(section, dict):
                            heading = section.get('heading', '')
                            content = section.get('content', '')
                            if content:
                                section_text = f"{heading}\n{content}" if heading else content
                                body_parts.append(section_text)
            
            body = "\n\n".join(body_parts) if body_parts else ""
            
            # Extract real URL with fallbacks - REQUIRED field in canonical schema
            url = source.get("page_url") or source.get("url")
            if not url or url == "#":
                # Construct a meaningful URL using doc_id
                doc_id = hit.get("_id") or f"doc_{len(results)+1}"
                url = f"#doc-{doc_id}"
            
            # Ensure doc_id is always valid and non-empty - CRITICAL for TurnResult validation
            doc_id = hit.get("_id") or f"doc_{len(results)+1}"
            if not doc_id or doc_id.strip() == "":
                doc_id = f"doc_{len(results)+1}"
            
            # Build metadata from source fields (preserving existing structure)
            metadata = {
                "page_url": source.get("page_url", ""),
                "utility_name": source.get("utility_name", ""),
                "api_name": source.get("api_name", ""),
                "sections_matched": len(inner_hits),
                "section_paths": section_paths,  # For actionable content detection
                "anchors": anchors,  # For direct section linking
                "space": source.get("space", ""),  # Track namespace
                # Preserve backward compatibility
                "page_title": title,
            }
            
            # CANONICAL SCHEMA: Always populate required fields
            result = ServiceSearchResult(
                doc_id=doc_id,      # REQUIRED: Always populated and non-empty
                title=title,        # REQUIRED: Always populated with fallback
                url=url,           # REQUIRED: Always populated with fallback  
                score=hit.get("_score", 0.0),  # REQUIRED: Handle missing scores
                content=body,       # REQUIRED: Use body as content
                metadata=metadata   # Additional fields for backward compatibility
            )
            results.append(result)
        
        return results
    
    def _build_simple_bm25_query(self, query: str, k: int) -> Dict[str, Any]:
        """Build optimized BM25 query with namespace filtering, acronym expansion, and title boosting."""
        from agent.acronym_map import expand_acronym, is_short_acronym_query
        
        # Expand acronyms if present (e.g., "CIU" -> "Customer Interaction Utility CIU")
        expanded_query, expansions = expand_acronym(query)
        is_acronym = is_short_acronym_query(query)
        
        # Extract key terms for BM25-friendly searching
        key_terms = self._extract_key_terms(expanded_query if expansions else query)
        
        # Build filter clauses - CRITICAL: Filter to Utilities namespace
        filter_clauses = []
        
        # NAMESPACE FILTER: Only search Utilities docs (prevents medical CIU match)
        filter_clauses.append({
            "bool": {
                "should": [
                    {"term": {"space.keyword": "Utilities"}},
                    {"term": {"labels.keyword": "utilities"}},
                    {"wildcard": {"path.keyword": "*/utilities/*"}},
                    {"prefix": {"title.keyword": "utilities/"}}
                ],
                "minimum_should_match": 1
            }
        })
        
        # Build should clauses for scoring
        should_clauses = []
        
        # ACRONYM HANDLING: Add exact title matches with high boosts
        if expansions:
            for expansion in expansions:
                # Exact title match gets highest boost
                should_clauses.append({
                    "term": {
                        "title.keyword": {
                            "value": expansion,
                            "boost": 20
                        }
                    }
                })
                # Also boost the acronym itself in title
                should_clauses.append({
                    "term": {
                        "title.keyword": {
                            "value": query.upper(),
                            "boost": 15
                        }
                    }
                })
                # Phrase match for the expansion
                should_clauses.append({
                    "match_phrase": {
                        "title": {
                            "query": expansion,
                            "boost": 10
                        }
                    }
                })
                
                # Boost documents containing associated API names from swagger_keyword.json
                from agent.acronym_map import get_apis_for_acronym
                acronym = query.upper().split()[0]  # Get first word as potential acronym
                api_names = get_apis_for_acronym(acronym)
                if api_names:
                    # High boost for exact API name matches in title/headings
                    for api_name in api_names[:5]:  # Limit to top 5 APIs to avoid query bloat
                        should_clauses.append({
                            "match_phrase": {
                                "title": {
                                    "query": api_name,
                                    "boost": 12
                                }
                            }
                        })
                        # Use root-level fields instead of nested
                        should_clauses.append({
                            "match_phrase": {
                                "title": {
                                    "query": api_name,
                                    "boost": 8
                                }
                            }
                        })
                        should_clauses.append({
                            "match_phrase": {
                                "content": {
                                    "query": api_name,
                                    "boost": 3
                                }
                            }
                        })
        
        # Enhanced multi_match query with expanded terms from synonyms
        multi_match_query = expanded_query if expansions else key_terms
        should_clauses.append({
            "multi_match": {
                "query": multi_match_query,
                "type": "most_fields",  # Use most_fields for better expansion matching
                "fields": [
                    "title^8" if is_acronym else "title^6",  # Extra boost for acronym queries
                    "name^4",
                    "content^3",
                    "body^3",
                    "text^2"
                ],
                "tie_breaker": 0.3,
                "minimum_should_match": "2<-1 5<-2" if len(multi_match_query.split()) > 4 else "1"
            }
        })
        
        # Build the complete bool query
        bool_query = {
            "bool": {
                "filter": filter_clauses,  # Namespace filtering
                "should": should_clauses,   # Scoring with acronym boosts
                "minimum_should_match": 1
            }
        }
        
        # PAYLOAD REDUCTION: Reduced size, essential fields, enable total tracking for error handling
        search_body = {
            "size": min(k, 20),  # Limit to 20 instead of 50 to reduce over-fetching
            "_source": ["page_url", "api_name", "utility_name", "title", "space", "labels", "path"],  # Essential fields + namespace info
            "track_total_hits": True,  # Enable to properly handle empty results
            "query": bool_query
        }
        return search_body
    
    def _build_simple_knn_query(self, query_vector: List[float], k: int) -> Dict[str, Any]:
        """Build optimized nested kNN query with namespace filtering and reduced payload size."""
        
        # NAMESPACE FILTER: Same as BM25 to prevent cross-domain matches
        filter_clauses = [{
            "bool": {
                "should": [
                    {"term": {"space.keyword": "Utilities"}},
                    {"term": {"labels.keyword": "utilities"}},
                    {"wildcard": {"path.keyword": "*/utilities/*"}},
                    {"prefix": {"title.keyword": "utilities/"}}
                ],
                "minimum_should_match": 1
            }
        }]
        
        # Use nested query structure for JPMC production index with optimizations
        search_body = {
            "size": min(k, 20),  # Limit to 20 instead of 50 to reduce over-fetching
            "_source": ["page_url", "api_name", "utility_name", "title", "space", "labels", "path"],  # Essential fields + namespace
            "track_total_hits": True,  # Enable for proper error handling
            "query": {
                "bool": {
                    "filter": filter_clauses,  # Apply namespace filter
                    "must": [{
                        "nested": {
                            "path": "sections",
                            "query": {
                                "knn": {
                                    "sections.embedding": {
                                        "vector": query_vector,
                                        "k": 5
                                    }
                                }
                            },
                            "inner_hits": {
                                "name": "matched_sections", 
                                "size": 3,  # Reduced from 5 to minimize payload
                                "sort": [{"_score": "desc"}],
                                "_source": ["heading", "content"]  # Only fetch needed fields
                            }
                        }
                    }]
                }
            }
        }
        return search_body
    
    def get_index_mapping(self, index_name: str) -> Dict[str, Any]:
        """Get the mapping (schema) for a specific index to discover field names."""
        try:
            url = f"{self.base_url}/{index_name}/_mapping"
            
            # Use direct requests with auth (like main branch)
            from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
            _setup_jpmc_proxy()  # Ensure proxy is configured
            aws_auth = _get_aws_auth()
            if aws_auth:
                logger.info("Using direct AWS4Auth for mapping request")
                response = requests.get(url, auth=aws_auth, timeout=30.0)
            else:
                logger.warning("No AWS auth available, using session for mapping")
                response = self.session.get(url, timeout=30.0)
            
            if not response.ok:
                error_body = response.text
                logger.error(f"Get mapping failed with {response.status_code}: {error_body}")
            response.raise_for_status()
            
            mapping_data = response.json()
            logger.info(f"Retrieved mapping for index {index_name}")
            return mapping_data
            
        except Exception as e:
            logger.error(f"Failed to get index mapping: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """Check OpenSearch cluster health and connectivity."""
        try:
            # Test basic connectivity
            url = f"{self.base_url}/_cluster/health"
            
            # Handle JPMC authentication (session can be None)
            if self.session is None:
                # Use direct requests with AWS auth for JPMC
                from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
                _setup_jpmc_proxy()
                aws_auth = _get_aws_auth()
                if aws_auth:
                    response = requests.get(url, auth=aws_auth, timeout=5.0)
                else:
                    # No auth available
                    raise ValueError("No authentication available for JPMC profile")
            else:
                response = self.session.get(url, timeout=5.0)
                
            response.raise_for_status()
            
            health = response.json()
            
            # Check if health data is valid
            if not health or not isinstance(health, dict):
                raise ValueError("Invalid health response from OpenSearch")
            
            # Test index existence using configured alias
            index_alias = self.settings.search_index_alias
            alias_url = f"{self.base_url}/{index_alias}"
            
            # Handle authentication for index check too
            if self.session is None:
                from src.infra.clients import _get_aws_auth
                aws_auth = _get_aws_auth()
                if aws_auth:
                    alias_response = requests.head(alias_url, auth=aws_auth, timeout=5.0)
                else:
                    alias_response = requests.head(alias_url, timeout=5.0)
            else:
                alias_response = self.session.head(alias_url, timeout=5.0)
                
            index_exists = alias_response.status_code == 200
            
            # Determine authentication status safely
            auth_status = "none"
            if self.session is not None and hasattr(self.session, 'auth') and self.session.auth:
                auth_status = "session_configured"
            elif self.settings.requires_aws_auth:
                auth_status = "aws_auth_configured"
            
            return {
                "status": "healthy" if health.get("status") in ["green", "yellow"] else "unhealthy",
                "cluster_name": health.get("cluster_name", "unknown"),
                "cluster_status": health.get("status", "unknown"),
                "node_count": health.get("number_of_nodes", 0),
                "data_nodes": health.get("number_of_data_nodes", 0),
                "index_exists": index_exists,
                "authentication": auth_status
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            
            # Safe authentication status for error case
            auth_status = "none"
            try:
                if self.session is not None and hasattr(self.session, 'auth') and self.session.auth:
                    auth_status = "session_configured"
                elif self.settings.requires_aws_auth:
                    auth_status = "aws_auth_configured"
            except:
                pass
                
            return {
                "status": "unhealthy",
                "error": str(e),
                "authentication": auth_status
            }


def create_search_client(settings=None) -> OpenSearchClient:
    """Factory function to create OpenSearch client with centralized settings."""
    return OpenSearchClient(settings)