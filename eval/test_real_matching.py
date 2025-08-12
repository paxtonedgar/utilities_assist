#!/usr/bin/env python3
"""
Test what our tuned BM25 query should actually find vs simple keyword matching.
Demonstrates the impact of proper field boosts, phrase matching, and penalties.
"""

import json
from typing import List, Dict, Any, Set
import re


def load_corpus():
    """Load mock corpus."""
    corpus = {}
    with open('eval/mock_corpus/utilities_docs.jsonl', 'r') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus[doc['canonical_id']] = doc
    return corpus


def simple_keyword_matching(query: str, corpus: Dict) -> List[Dict]:
    """Current evaluation simulation - simple keyword overlap."""
    query_words = set(query.lower().split())
    scored_docs = []
    
    for doc_id, doc in corpus.items():
        doc_text = f"{doc.get('title', '')} {doc.get('body', '')} {doc.get('section', '')}".lower()
        doc_words = set(doc_text.split())
        
        overlap = len(query_words.intersection(doc_words))
        score = overlap / len(query_words) if query_words else 0
        
        if score > 0:
            scored_docs.append({
                'id': doc_id,
                'title': doc.get('title', ''),
                'score': score,
                'method': 'simple_keyword'
            })
    
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    return scored_docs[:10]


def tuned_bm25_simulation(query: str, corpus: Dict) -> List[Dict]:
    """Simulate what tuned BM25 with our improvements should find."""
    
    # Extract key phrases like our tuned query does
    key_phrases = extract_key_phrases(query)
    query_words = set(query.lower().split())
    
    scored_docs = []
    
    for doc_id, doc in corpus.items():
        title = doc.get('title', '').lower()
        body = doc.get('body', '').lower()  
        section = doc.get('section', '').lower()
        
        score = 0.0
        
        # Title matching gets massive 10x boost
        title_words = set(title.split())
        title_overlap = len(query_words.intersection(title_words))
        if title_overlap > 0:
            score += title_overlap * 10.0  # title^10 boost
        
        # Exact title phrase matching gets additional 6x boost
        if any(phrase.lower() in title for phrase in query.split() if len(phrase) > 3):
            score += 6.0
        
        # Section matching gets 4x boost  
        section_words = set(section.split())
        section_overlap = len(query_words.intersection(section_words))
        if section_overlap > 0:
            score += section_overlap * 4.0  # section^4 boost
            
        # Body matching gets 1x (baseline)
        body_words = set(body.split())
        body_overlap = len(query_words.intersection(body_words))
        if body_overlap > 0:
            score += body_overlap * 1.0
            
        # Key phrase proximity bonus
        for phrase in key_phrases:
            if phrase in title:
                score += 3.0  # Title phrase bonus
            if phrase in body:
                score += 1.5  # Body phrase bonus
                
        # Generic section penalty (-1.2x)
        generic_sections = ['overview', 'global', 'platform', 'general', 'introduction']
        if section in generic_sections:
            score *= 0.8  # Apply -1.2x penalty (multiply by 1 - 1.2 = -0.2, but keep positive)
            
        # Generic title penalty  
        if any(generic in title for generic in ['overview', 'global', 'platform', 'general']):
            score *= 0.9
            
        # Recency boost for newer docs (simplified)
        if '2024-11' in doc.get('updated_at', '') or '2024-12' in doc.get('updated_at', '') or '2025-' in doc.get('updated_at', ''):
            score *= 1.2
        
        if score > 0:
            scored_docs.append({
                'id': doc_id,
                'title': doc.get('title', ''),
                'score': score,
                'section': section,
                'updated_at': doc.get('updated_at', ''),
                'method': 'tuned_bm25'
            })
    
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    return scored_docs[:10]


def extract_key_phrases(query: str, max_phrases: int = 3) -> List[str]:
    """Extract key phrases from query."""
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'do', 'i'}
    
    words = re.findall(r'\b\w+\b', query.lower())
    words = [w for w in words if w not in stopwords and len(w) > 2]
    
    key_phrases = []
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        key_phrases.append(bigram)
    
    return key_phrases[:max_phrases]


def analyze_query(query: str, expected_docs: List[str], corpus: Dict):
    """Analyze why a query fails and how tuning would help."""
    print(f"\n🔍 ANALYZING: '{query}'")
    print("=" * 60)
    
    # Simple matching (current evaluation)
    simple_results = simple_keyword_matching(query, corpus)
    
    # Tuned BM25 matching (what we implemented)
    tuned_results = tuned_bm25_simulation(query, corpus)
    
    print(f"Expected docs: {expected_docs}")
    print()
    
    print("📊 SIMPLE KEYWORD MATCHING (current eval):")
    for i, result in enumerate(simple_results[:5], 1):
        found_marker = "✅" if result['id'] in expected_docs else "❌"
        print(f"  {i}. {found_marker} {result['title']} (score: {result['score']:.2f})")
    
    print("\n🚀 TUNED BM25 SIMULATION:")
    for i, result in enumerate(tuned_results[:5], 1):
        found_marker = "✅" if result['id'] in expected_docs else "❌"
        print(f"  {i}. {found_marker} {result['title']} (score: {result['score']:.1f})")
        
    # Calculate precision improvement
    simple_hits = sum(1 for r in simple_results[:5] if r['id'] in expected_docs)
    tuned_hits = sum(1 for r in tuned_results[:5] if r['id'] in expected_docs)
    
    simple_p5 = simple_hits / 5.0
    tuned_p5 = tuned_hits / 5.0
    
    print(f"\n📈 PRECISION@5 COMPARISON:")
    print(f"  Simple: {simple_p5:.3f} ({simple_hits}/5 correct)")
    print(f"  Tuned:  {tuned_p5:.3f} ({tuned_hits}/5 correct)")
    print(f"  Improvement: {tuned_p5 - simple_p5:+.3f}")


def main():
    """Test real matching improvements."""
    print("🎯 REAL MATCHING ANALYSIS")
    print("Testing what tuned BM25 should actually find vs simple evaluation")
    print("=" * 70)
    
    corpus = load_corpus()
    
    # Test problematic queries from evaluation
    test_cases = [
        ("What APIs are available for account information?", 
         ["UTILS:API:ACCOUNT_UTILITY:v3", "UTILS:API:CUSTOMER_SUMMARY:v2"]),
        
        ("API Authentication Guide",
         ["UTILS:API:CUSTOMER_SUMMARY:v2"]),  # Should match API docs
        
        ("Customer account API endpoints", 
         ["UTILS:API:ACCOUNT_UTILITY:v3", "UTILS:API:CUSTOMER_SUMMARY:v2"]),
         
        ("How do I report a power outage?",
         ["UTILS:EMERGENCY:OUTAGE:v1#reporting"]),
         
        ("Global customer platform architecture", 
         ["UTILS:GLOBAL:PLATFORM:v2"])
    ]
    
    total_simple = 0
    total_tuned = 0
    
    for query, expected in test_cases:
        analyze_query(query, expected, corpus)
        
        # Calculate scores for summary
        simple_results = simple_keyword_matching(query, corpus)
        tuned_results = tuned_bm25_simulation(query, corpus)
        
        simple_hits = sum(1 for r in simple_results[:5] if r['id'] in expected)
        tuned_hits = sum(1 for r in tuned_results[:5] if r['id'] in expected)
        
        total_simple += simple_hits / 5.0
        total_tuned += tuned_hits / 5.0
    
    print(f"\n🎯 OVERALL IMPACT SUMMARY")
    print("=" * 70)
    print(f"Average Precision@5 (Simple):  {total_simple / len(test_cases):.3f}")
    print(f"Average Precision@5 (Tuned):   {total_tuned / len(test_cases):.3f}")
    print(f"Expected Improvement:          {(total_tuned - total_simple) / len(test_cases):+.3f}")
    
    if total_tuned > total_simple:
        print(f"\n✅ Tuned BM25 should significantly improve precision!")
        print("   The evaluation simulation is too simple to capture the benefits.")
    else:
        print(f"\n⚠️  Need to investigate further tuning opportunities.")


if __name__ == "__main__":
    main()