#!/usr/bin/env python3
"""
Hybrid evaluation client that runs minimal BM25 + tuned BM25 in parallel
and merges results with Reciprocal Rank Fusion (RRF).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

from eval.os_eval_client import OpenSearchEvalClient
from src.infra.opensearch_client import SearchFilters
from typing import Dict, Any, Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)

class HybridEvalClient(OpenSearchEvalClient):
    """Hybrid evaluation client using RRF fusion of minimal + tuned BM25."""
    
    def rank_docs_hybrid(self, 
                        query_text: str, 
                        acl_hash: Optional[str] = None,
                        index_alias: str = "confluence_current",
                        rrf_k: int = 60) -> List[Tuple[str, float]]:
        """
        Rank documents using hybrid RRF fusion of minimal + tuned BM25.
        
        Args:
            query_text: Search query string
            acl_hash: ACL hash for filtering
            index_alias: OpenSearch index or alias name  
            rrf_k: RRF parameter (typically 60)
            
        Returns:
            List of (doc_id, rrf_score) tuples ordered by RRF score
        """
        
        # Build search filters
        filters = SearchFilters(acl_hash=acl_hash) if acl_hash else None
        
        # Run minimal BM25 query (proven baseline)
        minimal_query = self._build_bm25_simple(query_text, filters)
        minimal_hits = self.search(index_alias, minimal_query, size=10)
        
        # Run tuned BM25 query (gentle improvements)
        tuned_query = self._build_bm25_tuned(query_text, filters)
        tuned_hits = self.search(index_alias, tuned_query, size=10)
        
        logger.info(f"Hybrid search: {len(minimal_hits)} minimal + {len(tuned_hits)} tuned hits")
        
        # Create rank mappings
        minimal_ranks = {hit.doc_id: i+1 for i, hit in enumerate(minimal_hits)}
        tuned_ranks = {hit.doc_id: i+1 for i, hit in enumerate(tuned_hits)}
        
        # Create document mapping for score retrieval
        doc_map = {}
        for hit in minimal_hits:
            doc_map[hit.doc_id] = hit
        for hit in tuned_hits:
            doc_map[hit.doc_id] = hit  # Tuned score overwrites if same doc
        
        # Apply RRF fusion
        rrf_scores = {}
        all_doc_ids = set(minimal_ranks.keys()) | set(tuned_ranks.keys())
        
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            
            # Add minimal BM25 contribution: 1 / (k + rank)
            if doc_id in minimal_ranks:
                rrf_score += 1.0 / (rrf_k + minimal_ranks[doc_id])
            
            # Add tuned BM25 contribution: 1 / (k + rank)  
            if doc_id in tuned_ranks:
                rrf_score += 1.0 / (rrf_k + tuned_ranks[doc_id])
            
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score (descending) and take top 10
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        
        logger.info(f"RRF fusion: {len(minimal_hits)} + {len(tuned_hits)} → {len(sorted_docs)} fused")
        
        # Return (doc_id, rrf_score) tuples
        return sorted_docs
    
    def compare_hybrid_modes(self, 
                           query_text: str, 
                           expected_doc_ids: List[str],
                           acl_hash: Optional[str] = None,
                           index_alias: str = "confluence_current") -> Dict[str, Any]:
        """
        Compare minimal, tuned, and hybrid performance for a single query.
        
        Returns:
            Comparison results with precision metrics for all three modes
        """
        
        # Get results from all three modes
        minimal_results = self.rank_docs(query_text, "simple", acl_hash, index_alias)
        tuned_results = self.rank_docs(query_text, "tuned", acl_hash, index_alias) 
        hybrid_results = self.rank_docs_hybrid(query_text, acl_hash, index_alias)
        
        # Calculate precision@5 for all modes
        def calc_precision_at_5(results: List[Tuple[str, float]], relevant: List[str]) -> float:
            top_5_ids = [doc_id for doc_id, _ in results[:5]]
            hits = sum(1 for doc_id in top_5_ids if doc_id in relevant)
            return hits / 5.0
        
        minimal_p5 = calc_precision_at_5(minimal_results, expected_doc_ids)
        tuned_p5 = calc_precision_at_5(tuned_results, expected_doc_ids)
        hybrid_p5 = calc_precision_at_5(hybrid_results, expected_doc_ids)
        
        return {
            "query": query_text,
            "expected_docs": expected_doc_ids,
            "minimal": {
                "results": minimal_results[:5],
                "precision_at_5": minimal_p5
            },
            "tuned": {
                "results": tuned_results[:5],
                "precision_at_5": tuned_p5
            },
            "hybrid": {
                "results": hybrid_results[:5], 
                "precision_at_5": hybrid_p5
            },
            "improvements": {
                "tuned_vs_minimal": tuned_p5 - minimal_p5,
                "hybrid_vs_minimal": hybrid_p5 - minimal_p5,
                "hybrid_vs_tuned": hybrid_p5 - tuned_p5
            }
        }


def test_hybrid_eval():
    """Test the hybrid evaluation client."""
    print("🔥 Testing Hybrid Evaluation Client")
    print("=" * 50)
    
    client = HybridEvalClient()
    
    # Test cases that should show the benefits of hybrid approach
    test_cases = [
        {
            "query": "How do I start new utility service?",
            "expected": ["UTILS:CUST:START_SERVICE:v4#overview", "UTILS:CUST:START_SERVICE:v4#eligibility"],
            "acl": "public"
        },
        {
            "query": "What credit score is needed to start service?", 
            "expected": ["UTILS:CUST:START_SERVICE:v4#eligibility"],
            "acl": "grp_care"
        },
        {
            "query": "Walk me through stopping utility service",
            "expected": ["UTILS:CUST:STOP_SERVICE:v3#overview", "UTILS:CUST:STOP_SERVICE:v3#final_bill"],
            "acl": "public"
        },
        {
            "query": "How do I switch to time-of-use rates?",
            "expected": ["UTILS:RATE:TOU_SWITCH:v3#overview", "UTILS:RATE:TOU_SWITCH:v3#eligibility"],
            "acl": "public"
        }
    ]
    
    total_improvements = {"tuned": 0, "hybrid": 0}
    successful_queries = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test['query']}")
        print(f"   Expected: {len(test['expected'])} docs")
        print(f"   ACL: {test['acl']}")
        
        try:
            result = client.compare_hybrid_modes(
                test['query'], 
                test['expected'],
                test['acl']
            )
            
            minimal_p5 = result['minimal']['precision_at_5']
            tuned_p5 = result['tuned']['precision_at_5']
            hybrid_p5 = result['hybrid']['precision_at_5']
            
            print(f"   📊 Precision@5:")
            print(f"      Minimal: {minimal_p5:.3f}")
            print(f"      Tuned:   {tuned_p5:.3f} ({result['improvements']['tuned_vs_minimal']:+.3f})")
            print(f"      Hybrid:  {hybrid_p5:.3f} ({result['improvements']['hybrid_vs_minimal']:+.3f})")
            
            # Track improvements
            total_improvements["tuned"] += result['improvements']['tuned_vs_minimal']
            total_improvements["hybrid"] += result['improvements']['hybrid_vs_minimal'] 
            successful_queries += 1
            
            # Show top results
            print(f"   🏆 Top 3 Hybrid Results:")
            for j, (doc_id, score) in enumerate(result['hybrid']['results'][:3], 1):
                is_expected = "✅" if doc_id in test['expected'] else "◦"
                print(f"      {j}. {doc_id} ({score:.3f}) {is_expected}")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
    
    if successful_queries > 0:
        avg_tuned_improvement = total_improvements["tuned"] / successful_queries
        avg_hybrid_improvement = total_improvements["hybrid"] / successful_queries
        
        print(f"\n" + "=" * 50)
        print(f"📊 HYBRID EVALUATION SUMMARY:")
        print(f"   Successful queries: {successful_queries}/{len(test_cases)}")
        print(f"   Average tuned improvement:  {avg_tuned_improvement:+.3f}")
        print(f"   Average hybrid improvement: {avg_hybrid_improvement:+.3f}")
        
        if avg_hybrid_improvement > avg_tuned_improvement:
            print(f"   ✅ Hybrid is best: {avg_hybrid_improvement - avg_tuned_improvement:+.3f} better than tuned")
        elif avg_tuned_improvement > avg_hybrid_improvement:
            print(f"   ⚖️ Tuned is better: {avg_tuned_improvement - avg_hybrid_improvement:+.3f} better than hybrid")
        else:
            print(f"   🟰 Tuned and hybrid are tied")


if __name__ == "__main__":
    test_hybrid_eval()