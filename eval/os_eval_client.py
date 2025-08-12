#!/usr/bin/env python3
"""
OpenSearch evaluation client that calls actual OpenSearch with tuned vs simple BM25 queries.
Uses the same pooled session as the main app for realistic testing.
"""

import logging
import time
import re
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.infra.config import get_settings
from src.infra.opensearch_client import OpenSearchClient, SearchFilters


logger = logging.getLogger(__name__)


@dataclass
class Hit:
    """Search hit result."""
    doc_id: str
    score: float
    title: str
    body: str
    section: str
    metadata: Dict[str, Any]


class OpenSearchEvalClient:
    """Evaluation client that calls actual OpenSearch with different query strategies."""
    
    def __init__(self):
        """Initialize with same config as main app."""
        self.settings = get_settings()
        self.os_client = OpenSearchClient(self.settings.search)
        
    def search(self, 
               index_alias: str, 
               body: Dict[str, Any], 
               size: int = 10, 
               explain: bool = False) -> List[Hit]:
        """
        Execute OpenSearch query and return hits.
        
        Args:
            index_alias: OpenSearch index or alias name
            body: OpenSearch query body
            size: Number of results to return
            explain: Whether to include score explanations
            
        Returns:
            List of Hit objects with doc_id, score, and content
        """
        try:
            # Add size to body
            body["size"] = size
            if explain:
                body["explain"] = True
                
            # Execute search using the session from OpenSearchClient
            url = f"{self.os_client.base_url}/{index_alias}/_search"
            response = self.os_client.session.post(
                url, 
                json=body, 
                timeout=self.os_client.config.timeout_s
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Parse hits
            hits = []
            for hit in data["hits"]["hits"]:
                source = hit["_source"]
                
                hit_obj = Hit(
                    doc_id=source.get("canonical_id", hit["_id"]),
                    score=hit["_score"],
                    title=source.get("title", ""),
                    body=source.get("body", ""),
                    section=source.get("section", ""),
                    metadata=source.get("metadata", {})
                )
                hits.append(hit_obj)
                
            logger.info(f"OpenSearch query returned {len(hits)}/{data['hits']['total']['value']} hits")
            return hits
            
        except Exception as e:
            logger.error(f"OpenSearch query failed: {e}")
            return []
    
    def rank_docs(self, 
                  query_text: str, 
                  mode: str = "tuned", 
                  acl_hash: Optional[str] = None,
                  index_alias: str = "confluence_current") -> List[Tuple[str, float]]:
        """
        Rank documents using simple or tuned BM25 query.
        
        Args:
            query_text: Search query string
            mode: "simple" or "tuned" query building mode
            acl_hash: ACL hash for filtering (None for public access)
            index_alias: OpenSearch index or alias name
            
        Returns:
            List of (doc_id, score) tuples ordered by relevance
        """
        
        # Build search filters - always apply ACL filtering (including "public")
        filters = SearchFilters(acl_hash=acl_hash) if acl_hash else None
        
        # Build query based on mode
        if mode == "simple":
            query_body = self._build_bm25_simple(query_text, filters)
        elif mode == "tuned":
            query_body = self._build_bm25_tuned(query_text, filters)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'simple' or 'tuned'")
        
        # Execute search
        hits = self.search(index_alias, query_body, size=10)
        
        # Return (doc_id, score) tuples
        return [(hit.doc_id, hit.score) for hit in hits]
    
    def extract_key_terms(self, query_text: str) -> str:
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

    def _build_bm25_simple(self, 
                          query: str, 
                          filters: Optional[SearchFilters]) -> Dict[str, Any]:
        """Build simple BM25 query (baseline) with proven multi_match approach."""
        
        # Extract key terms from natural language
        key_terms = self.extract_key_terms(query)
        
        # Use proven multi_match approach instead of simple_query_string
        base_query = {
            "multi_match": {
                "query": key_terms,
                "fields": ["title^5", "body^1", "section^2"],
                "type": "best_fields"
            }
        }
        
        # Apply filters
        filter_clauses = []
        if filters:
            filter_clauses = self._build_filter_clauses(filters)
        
        # Wrap in bool query
        if filter_clauses:
            bool_query = {
                "bool": {
                    "must": [base_query],
                    "filter": filter_clauses
                }
            }
        else:
            bool_query = base_query
        
        return {
            "query": bool_query,
            "sort": [{"_score": {"order": "desc"}}],
            "_source": ["title", "body", "metadata", "updated_at", "canonical_id", "section"],
            "track_scores": True
        }
    
    def _build_bm25_tuned(self, 
                         query: str, 
                         filters: Optional[SearchFilters]) -> Dict[str, Any]:
        """Build tuned BM25 query using the OpenSearchClient implementation."""
        
        # Use the tuned query building from OpenSearchClient
        # We'll call the internal method directly for evaluation
        return self.os_client._build_bm25_query(
            query=query,
            filters=filters,
            k=10,
            time_decay_half_life_days=75  # Our tuned value
        )
    
    def _build_filter_clauses(self, filters: SearchFilters) -> List[Dict[str, Any]]:
        """Build filter clauses from SearchFilters."""
        clauses = []
        
        if filters.acl_hash:
            clauses.append({
                "term": {"acl_hash": filters.acl_hash}
            })
            
        if filters.space_key:
            clauses.append({
                "term": {"space_key": filters.space_key}
            })
            
        if filters.content_type:
            clauses.append({
                "term": {"content_type": filters.content_type}
            })
            
        if filters.updated_after:
            clauses.append({
                "range": {
                    "updated_at": {
                        "gte": filters.updated_after.isoformat()
                    }
                }
            })
            
        if filters.updated_before:
            clauses.append({
                "range": {
                    "updated_at": {
                        "lte": filters.updated_before.isoformat()
                    }
                }
            })
            
        return clauses
    
    def compare_modes(self, 
                     query_text: str, 
                     expected_doc_ids: List[str],
                     acl_hash: Optional[str] = None,
                     index_alias: str = "confluence_current") -> Dict[str, Any]:
        """
        Compare simple vs tuned query performance for a single query.
        
        Args:
            query_text: Query to test
            expected_doc_ids: List of relevant document IDs  
            acl_hash: ACL filter
            index_alias: OpenSearch index
            
        Returns:
            Comparison results with precision metrics
        """
        
        # Get results from both modes
        simple_results = self.rank_docs(query_text, "simple", acl_hash, index_alias)
        tuned_results = self.rank_docs(query_text, "tuned", acl_hash, index_alias)
        
        # Calculate precision@5 for both
        def calc_precision_at_5(results: List[Tuple[str, float]], relevant: List[str]) -> float:
            top_5_ids = [doc_id for doc_id, _ in results[:5]]
            hits = sum(1 for doc_id in top_5_ids if doc_id in relevant)
            return hits / 5.0
        
        simple_p5 = calc_precision_at_5(simple_results, expected_doc_ids)
        tuned_p5 = calc_precision_at_5(tuned_results, expected_doc_ids)
        
        return {
            "query": query_text,
            "expected_docs": expected_doc_ids,
            "simple": {
                "results": simple_results[:5],
                "precision_at_5": simple_p5
            },
            "tuned": {
                "results": tuned_results[:5], 
                "precision_at_5": tuned_p5
            },
            "improvement": tuned_p5 - simple_p5
        }


def test_eval_client():
    """Quick test of the evaluation client."""
    print("🔍 Testing OpenSearch Evaluation Client")
    print("=" * 50)
    
    client = OpenSearchEvalClient()
    
    # Test query
    test_query = "API Authentication Guide"
    expected_docs = ["UTILS:API:CUSTOMER_SUMMARY:v2"]
    
    try:
        # Compare modes
        result = client.compare_modes(test_query, expected_docs)
        
        print(f"Query: {result['query']}")
        print(f"Expected: {result['expected_docs']}")
        print()
        print(f"Simple P@5: {result['simple']['precision_at_5']:.3f}")
        print(f"Tuned P@5:  {result['tuned']['precision_at_5']:.3f}")
        print(f"Improvement: {result['improvement']:+.3f}")
        
        print("\nTop 3 Results (Simple):")
        for i, (doc_id, score) in enumerate(result['simple']['results'][:3], 1):
            print(f"  {i}. {doc_id} (score: {score:.2f})")
            
        print("\nTop 3 Results (Tuned):")
        for i, (doc_id, score) in enumerate(result['tuned']['results'][:3], 1):
            print(f"  {i}. {doc_id} (score: {score:.2f})")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This is expected if OpenSearch is not running or accessible")
        

if __name__ == "__main__":
    test_eval_client()