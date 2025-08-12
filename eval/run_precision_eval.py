#!/usr/bin/env python3
"""
Quick precision evaluation runner to test RRF+MMR+rules vs BM25-only baseline.
Target: Precision@5 >= 0.6 for enhanced pipeline, BM25-only as baseline floor.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.services.rerank import LightweightReranker


def load_test_data():
    """Load golden set and mock corpus."""
    eval_dir = Path(__file__).parent
    
    # Load golden set
    with open(eval_dir / "golden_set.yaml", 'r') as f:
        golden_set = yaml.safe_load(f)
    
    # Load mock corpus
    corpus = {}
    with open(eval_dir / "mock_corpus" / "utilities_docs.jsonl", 'r') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                corpus[doc['canonical_id']] = doc
    
    return golden_set, corpus


def mock_bm25_retrieval(query: str, corpus: Dict, k: int = 10) -> List[Dict]:
    """Simple BM25-like retrieval simulation."""
    query_words = set(query.lower().split())
    
    scored_docs = []
    for doc_id, doc in corpus.items():
        # Simple keyword scoring
        doc_text = f"{doc.get('title', '')} {doc.get('body', '')} {doc.get('section', '')}".lower()
        doc_words = set(doc_text.split())
        
        # BM25-like scoring (simplified)
        overlap = len(query_words.intersection(doc_words))
        score = overlap / max(len(query_words), 1)
        
        if score > 0:
            scored_docs.append({
                'id': doc_id,
                'title': doc.get('title', ''),
                'section': doc.get('section', ''),
                'content': doc.get('body', ''),
                'score': score,
                'updated_at': doc.get('updated_at', ''),
                'url': doc.get('url', '')
            })
    
    # Sort by score and return top k
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    return scored_docs[:k]


def calculate_precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Precision@K."""
    if not retrieved or k == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    correct = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    
    return correct / min(k, len(retrieved_at_k))


def evaluate_pipeline(queries: List[Dict], corpus: Dict, use_reranker: bool = False) -> Dict:
    """Evaluate retrieval pipeline."""
    reranker = LightweightReranker() if use_reranker else None
    
    precision_scores = []
    query_results = []
    
    for query_data in queries:
        query_text = query_data['query']
        expected_docs = query_data.get('expected_doc_ids', [])
        
        # Get BM25 results
        bm25_results = mock_bm25_retrieval(query_text, corpus, k=10)
        
        # Apply reranking if enabled
        if use_reranker and bm25_results:
            rerank_result = reranker.rerank(bm25_results, query_text, max_results=10)
            final_results = rerank_result.documents
        else:
            final_results = bm25_results
        
        # Calculate precision@5
        retrieved_ids = [doc['id'] for doc in final_results]
        precision_5 = calculate_precision_at_k(retrieved_ids, expected_docs, 5)
        precision_scores.append(precision_5)
        
        query_results.append({
            'query': query_text,
            'expected': expected_docs,
            'retrieved': retrieved_ids[:5],
            'precision_5': precision_5
        })
    
    return {
        'avg_precision_5': sum(precision_scores) / len(precision_scores),
        'min_precision_5': min(precision_scores),
        'max_precision_5': max(precision_scores),
        'queries_above_0_6': sum(1 for p in precision_scores if p >= 0.6),
        'total_queries': len(precision_scores),
        'detailed_results': query_results
    }


def main():
    """Run precision evaluation comparison."""
    print("🎯 PRECISION@5 EVALUATION")
    print("=" * 50)
    
    # Load test data
    golden_set, corpus = load_test_data()
    queries = golden_set['queries']
    
    print(f"📊 Evaluating {len(queries)} queries against {len(corpus)} documents")
    print()
    
    # Evaluate BM25-only baseline
    print("🔍 Evaluating BM25-only baseline...")
    baseline_results = evaluate_pipeline(queries, corpus, use_reranker=False)
    
    # Evaluate RRF+MMR+rules pipeline
    print("🚀 Evaluating RRF+MMR+rules pipeline...")
    enhanced_results = evaluate_pipeline(queries, corpus, use_reranker=True)
    
    # Print comparison
    print("\n📈 RESULTS COMPARISON")
    print("-" * 50)
    print(f"{'Metric':<25} {'BM25 Only':<12} {'Enhanced':<12} {'Δ':<8}")
    print("-" * 50)
    
    metrics = [
        ('Avg Precision@5', 'avg_precision_5'),
        ('Min Precision@5', 'min_precision_5'),
        ('Max Precision@5', 'max_precision_5'),
        ('Queries >= 0.6', 'queries_above_0_6')
    ]
    
    for metric_name, metric_key in metrics:
        baseline_val = baseline_results[metric_key]
        enhanced_val = enhanced_results[metric_key]
        
        if metric_key == 'queries_above_0_6':
            baseline_pct = baseline_val / baseline_results['total_queries'] * 100
            enhanced_pct = enhanced_val / enhanced_results['total_queries'] * 100
            delta = enhanced_pct - baseline_pct
            print(f"{metric_name:<25} {baseline_pct:>8.1f}%    {enhanced_pct:>8.1f}%    {delta:>+6.1f}%")
        else:
            delta = enhanced_val - baseline_val
            print(f"{metric_name:<25} {baseline_val:>8.3f}    {enhanced_val:>8.3f}    {delta:>+6.3f}")
    
    # Target assessment
    print("\n🎯 TARGET ASSESSMENT")
    print("-" * 50)
    target_precision = 0.6
    baseline_meets_target = baseline_results['avg_precision_5'] >= target_precision
    enhanced_meets_target = enhanced_results['avg_precision_5'] >= target_precision
    
    print(f"Target: Precision@5 >= {target_precision}")
    print(f"BM25 Baseline:  {'✅' if baseline_meets_target else '❌'} {baseline_results['avg_precision_5']:.3f}")
    print(f"Enhanced Pipeline: {'✅' if enhanced_meets_target else '❌'} {enhanced_results['avg_precision_5']:.3f}")
    
    if enhanced_meets_target:
        improvement = enhanced_results['avg_precision_5'] - baseline_results['avg_precision_5']
        print(f"✨ Success! Enhanced pipeline beats baseline by {improvement:.3f}")
    else:
        print("⚠️  Enhanced pipeline needs tuning to reach target")
    
    # Show worst performing queries
    print("\n🔍 WORST PERFORMING QUERIES (Enhanced Pipeline)")
    print("-" * 50)
    worst_queries = sorted(enhanced_results['detailed_results'], key=lambda x: x['precision_5'])[:5]
    
    for i, result in enumerate(worst_queries, 1):
        print(f"{i}. P@5={result['precision_5']:.3f}: {result['query']}")
        print(f"   Expected: {len(result['expected'])} docs")
        print(f"   Retrieved top 5: {result['retrieved']}")
        print()
    
    return 0


if __name__ == "__main__":
    exit(main())