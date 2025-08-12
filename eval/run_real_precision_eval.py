#!/usr/bin/env python3
"""
Real precision evaluation using actual OpenSearch with tuned vs simple BM25 queries.
Measures true impact of our retrieval tuning against live OpenSearch index.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add paths for imports  
sys.path.append(str(Path(__file__).parent.parent))

from eval.os_eval_client import OpenSearchEvalClient


def load_golden_set() -> Dict[str, Any]:
    """Load golden set queries and expected results."""
    with open(Path(__file__).parent / "golden_set.yaml", 'r') as f:
        return yaml.safe_load(f)


def evaluate_with_opensearch(client: OpenSearchEvalClient, 
                           queries: List[Dict[str, Any]],
                           index_alias: str = "confluence_current") -> Dict[str, Any]:
    """
    Evaluate precision using real OpenSearch with simple vs tuned queries.
    
    Args:
        client: OpenSearch evaluation client
        queries: List of query objects with expected docs
        index_alias: OpenSearch index name
        
    Returns:
        Evaluation results with precision metrics
    """
    
    simple_precisions = []
    tuned_precisions = []
    detailed_results = []
    
    successful_queries = 0
    
    for query_obj in queries:
        query_text = query_obj['query']
        expected_docs = query_obj.get('expected_doc_ids', [])
        acl_hash = query_obj.get('acl')  # Use ACL from golden set
        
        print(f"📊 Evaluating: {query_text[:50]}...")
        
        try:
            # Compare simple vs tuned for this query
            result = client.compare_modes(
                query_text=query_text,
                expected_doc_ids=expected_docs, 
                acl_hash=acl_hash if acl_hash != "public" else None,
                index_alias=index_alias
            )
            
            simple_precisions.append(result['simple']['precision_at_5'])
            tuned_precisions.append(result['tuned']['precision_at_5'])
            detailed_results.append(result)
            successful_queries += 1
            
            # Show progress
            simple_p5 = result['simple']['precision_at_5'] 
            tuned_p5 = result['tuned']['precision_at_5']
            improvement = result['improvement']
            
            print(f"   Simple: {simple_p5:.3f} | Tuned: {tuned_p5:.3f} | Δ: {improvement:+.3f}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            continue
    
    # Calculate overall metrics
    if successful_queries == 0:
        return {
            "error": "No successful queries - OpenSearch may not be accessible",
            "successful_queries": 0
        }
    
    avg_simple = sum(simple_precisions) / len(simple_precisions)
    avg_tuned = sum(tuned_precisions) / len(tuned_precisions)
    
    simple_above_0_6 = sum(1 for p in simple_precisions if p >= 0.6)
    tuned_above_0_6 = sum(1 for p in tuned_precisions if p >= 0.6)
    
    return {
        "successful_queries": successful_queries,
        "total_queries": len(queries),
        "simple": {
            "avg_precision_5": avg_simple,
            "min_precision_5": min(simple_precisions),
            "max_precision_5": max(simple_precisions),
            "queries_above_0_6": simple_above_0_6
        },
        "tuned": {
            "avg_precision_5": avg_tuned,
            "min_precision_5": min(tuned_precisions),
            "max_precision_5": max(tuned_precisions), 
            "queries_above_0_6": tuned_above_0_6
        },
        "improvement": avg_tuned - avg_simple,
        "detailed_results": detailed_results
    }


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a nice format."""
    
    if "error" in results:
        print("❌ EVALUATION FAILED")
        print("=" * 50)
        print(results["error"])
        print("\nTo run real evaluation, you need:")
        print("1. OpenSearch running (scripts/start_opensearch_local.sh)")
        print("2. Mock corpus indexed (scripts/index_mock_docs.py)")
        print("3. Index alias 'confluence_current' pointing to the data")
        return
    
    print("🎯 REAL OPENSEARCH PRECISION@5 EVALUATION")
    print("=" * 60)
    print(f"📊 Successfully evaluated {results['successful_queries']}/{results['total_queries']} queries")
    print()
    
    # Results comparison table
    print("📈 RESULTS COMPARISON")
    print("-" * 60)
    print(f"{'Metric':<25} {'Simple':<12} {'Tuned':<12} {'Δ':<8}")
    print("-" * 60)
    
    simple = results['simple']
    tuned = results['tuned']
    
    metrics = [
        ('Avg Precision@5', 'avg_precision_5'),
        ('Min Precision@5', 'min_precision_5'), 
        ('Max Precision@5', 'max_precision_5'),
        ('Queries >= 0.6', 'queries_above_0_6')
    ]
    
    for metric_name, metric_key in metrics:
        simple_val = simple[metric_key]
        tuned_val = tuned[metric_key]
        
        if metric_key == 'queries_above_0_6':
            simple_pct = simple_val / results['successful_queries'] * 100
            tuned_pct = tuned_val / results['successful_queries'] * 100
            delta = tuned_pct - simple_pct
            print(f"{metric_name:<25} {simple_pct:>8.1f}%    {tuned_pct:>8.1f}%    {delta:>+6.1f}%")
        else:
            delta = tuned_val - simple_val
            print(f"{metric_name:<25} {simple_val:>8.3f}    {tuned_val:>8.3f}    {delta:>+6.3f}")
    
    # Target assessment
    print(f"\n🎯 TARGET ASSESSMENT")
    print("-" * 60)
    target_precision = 0.6
    simple_meets_target = simple['avg_precision_5'] >= target_precision
    tuned_meets_target = tuned['avg_precision_5'] >= target_precision
    
    print(f"Target: Precision@5 >= {target_precision}")
    print(f"Simple BM25:       {'✅' if simple_meets_target else '❌'} {simple['avg_precision_5']:.3f}")
    print(f"Tuned BM25:        {'✅' if tuned_meets_target else '❌'} {tuned['avg_precision_5']:.3f}")
    
    improvement = results['improvement']
    if improvement > 0.05:
        print(f"✨ Success! Tuned BM25 beats simple by {improvement:.3f}")
        if tuned_meets_target:
            print(f"🎯 Target achieved! Tuned BM25 reaches {target_precision} precision threshold")
        else:
            print(f"📈 On track - need {target_precision - tuned['avg_precision_5']:.3f} more to hit target")
    elif improvement > 0:
        print(f"📊 Modest improvement: +{improvement:.3f}")
    else:
        print(f"⚠️  No improvement measured - may need more tuning")
    
    # Show biggest improvements
    if results['detailed_results']:
        print(f"\n🚀 BIGGEST IMPROVEMENTS")
        print("-" * 60)
        
        # Sort by improvement
        sorted_results = sorted(results['detailed_results'], 
                              key=lambda x: x['improvement'], reverse=True)
        
        for result in sorted_results[:5]:
            improvement = result['improvement'] 
            if improvement > 0:
                print(f"• {result['query'][:50]}...")
                print(f"  Simple: {result['simple']['precision_at_5']:.3f} → Tuned: {result['tuned']['precision_at_5']:.3f} ({improvement:+.3f})")


def main():
    """Run real OpenSearch precision evaluation."""
    print("🔍 REAL OPENSEARCH PRECISION EVALUATION")
    print("Testing tuned BM25 against actual OpenSearch index")
    print("=" * 70)
    
    # Load test queries
    try:
        golden_set = load_golden_set()
        queries = golden_set['queries']
        print(f"✅ Loaded {len(queries)} test queries from golden set")
    except Exception as e:
        print(f"❌ Failed to load golden set: {e}")
        return 1
    
    # Initialize OpenSearch client
    try:
        client = OpenSearchEvalClient()
        print(f"✅ OpenSearch client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize OpenSearch client: {e}")
        return 1
    
    # Run evaluation  
    print(f"\n🎯 Running precision evaluation...")
    results = evaluate_with_opensearch(client, queries)
    
    # Print results
    print_results(results)
    
    return 0


if __name__ == "__main__":
    exit(main())