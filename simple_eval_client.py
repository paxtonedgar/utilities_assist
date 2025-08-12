#!/usr/bin/env python3
"""
Simplified evaluation client using proven minimal multi_match approach.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from eval.os_eval_client import OpenSearchEvalClient
from src.infra.opensearch_client import SearchFilters
from typing import Dict, Any, Optional, List, Tuple
import re

class SimpleEvalClient(OpenSearchEvalClient):
    """Simplified evaluation client using basic multi_match queries."""
    
    def extract_key_terms(self, query_text: str) -> str:
        """Extract key terms from natural language queries."""
        # Remove question words and common stopwords
        stopwords = {'what', 'how', 'do', 'i', 'is', 'are', 'the', 'to', 'for', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'by', 'with', 'from'}
        
        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r'[^\w\s]', '', query_text.lower())
        
        # Split into words and filter stopwords
        words = [word for word in cleaned.split() if word not in stopwords and len(word) > 2]
        
        # Return key terms as space-separated string
        return ' '.join(words)
    
    def _build_minimal_query(self, 
                           query: str, 
                           filters: Optional[SearchFilters]) -> Dict[str, Any]:
        """Build minimal multi_match query that's proven to work."""
        
        # Extract key terms from natural language
        key_terms = self.extract_key_terms(query)
        
        # Basic multi_match query with proven field boosts
        base_query = {
            "multi_match": {
                "query": key_terms,
                "fields": ["title^5", "body^1", "section^2"],
                "type": "best_fields"
            }
        }
        
        # Apply filters if provided
        filter_clauses = []
        if filters:
            filter_clauses = self._build_filter_clauses(filters)
        
        # Wrap in bool query if filters exist
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
    
    def rank_docs_minimal(self, 
                         query_text: str, 
                         acl_hash: Optional[str] = None,
                         index_alias: str = "confluence_current") -> List[Tuple[str, float]]:
        """
        Rank documents using minimal proven approach.
        """
        
        # Build search filters
        filters = SearchFilters(acl_hash=acl_hash) if acl_hash else None
        
        # Build minimal query
        query_body = self._build_minimal_query(query_text, filters)
        
        # Execute search
        hits = self.search(index_alias, query_body, size=10)
        
        # Return (doc_id, score) tuples
        return [(hit.doc_id, hit.score) for hit in hits]


def test_simple_eval():
    """Test the simplified evaluation client."""
    print("🔥 Testing Simplified Evaluation Client")
    print("=" * 50)
    
    client = SimpleEvalClient()
    
    # Test cases from golden set that should now work
    test_cases = [
        {
            "id": "Q001",
            "query": "How do I start new utility service?",
            "expected": ["UTILS:CUST:START_SERVICE:v4#overview", "UTILS:CUST:START_SERVICE:v4#eligibility"],
            "acl": "public"
        },
        {
            "id": "Q002", 
            "query": "What credit score is needed to start service?",
            "expected": ["UTILS:CUST:START_SERVICE:v4#eligibility"],
            "acl": "grp_care"
        },
        {
            "id": "Q008",
            "query": "Walk me through stopping utility service",
            "expected": ["UTILS:CUST:STOP_SERVICE:v3#overview", "UTILS:CUST:STOP_SERVICE:v3#final_bill"],
            "acl": "public"
        },
        {
            "id": "Q011",
            "query": "How do I switch to time-of-use rates?",
            "expected": ["UTILS:RATE:TOU_SWITCH:v3#overview", "UTILS:RATE:TOU_SWITCH:v3#eligibility"],
            "acl": "public"
        },
        {
            "id": "Q021",
            "query": "How do I report a power outage?",
            "expected": ["UTILS:EMERGENCY:OUTAGE:v1#reporting"],
            "acl": "public"
        }
    ]
    
    total_precision = 0
    successful_queries = 0
    
    for test in test_cases:
        print(f"\n🔍 {test['id']}: {test['query']}")
        print(f"   Expected: {test['expected']}")
        print(f"   ACL: {test['acl']}")
        
        # Extract key terms to show what we're actually searching
        key_terms = client.extract_key_terms(test['query'])
        print(f"   Key terms: '{key_terms}'")
        
        try:
            results = client.rank_docs_minimal(
                query_text=test['query'],
                acl_hash=test['acl'],
                index_alias="confluence_current"
            )
            
            print(f"   📊 Results ({len(results)} found):")
            if results:
                precision_hits = 0
                for i, (doc_id, score) in enumerate(results[:5]):  # Top 5 for P@5
                    is_expected = "✅" if doc_id in test['expected'] else "◦"
                    if doc_id in test['expected']:
                        precision_hits += 1
                    print(f"      {i+1}. {doc_id} (score: {score:.2f}) {is_expected}")
                
                precision_at_5 = precision_hits / 5.0
                print(f"   📈 Precision@5: {precision_at_5:.3f}")
                total_precision += precision_at_5
                successful_queries += 1
            else:
                print("      No results found")
                
        except Exception as e:
            print(f"   💥 Error: {e}")
    
    if successful_queries > 0:
        avg_precision = total_precision / successful_queries
        print(f"\n" + "=" * 50)
        print(f"📊 SUMMARY:")
        print(f"   Successful queries: {successful_queries}/{len(test_cases)}")
        print(f"   Average Precision@5: {avg_precision:.3f}")
        print(f"   Status: {'✅ PASS' if avg_precision > 0.3 else '⚠️ NEEDS IMPROVEMENT'}")


if __name__ == "__main__":
    test_simple_eval()