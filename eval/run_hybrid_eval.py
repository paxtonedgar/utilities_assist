#!/usr/bin/env python3
"""
Run comprehensive evaluation comparing minimal, tuned, and hybrid modes.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "src"))

import yaml
import json
import logging
from datetime import datetime
from eval.hybrid_eval_client import HybridEvalClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_golden_set(golden_set_file: str):
    """Load golden set queries from YAML file."""
    with open(golden_set_file, 'r') as f:
        data = yaml.safe_load(f)
    return data['queries']

def calculate_metrics(results, expected_docs):
    """Calculate precision@5, recall@10, and NDCG@10."""
    if not results:
        return {"precision@5": 0.0, "recall@10": 0.0, "ndcg@10": 0.0}
    
    # Get top 5 and top 10
    top_5_ids = [doc_id for doc_id, _ in results[:5]]
    top_10_ids = [doc_id for doc_id, _ in results[:10]]
    
    # Precision@5
    p5_hits = sum(1 for doc_id in top_5_ids if doc_id in expected_docs)
    precision_at_5 = p5_hits / 5.0
    
    # Recall@10  
    r10_hits = sum(1 for doc_id in top_10_ids if doc_id in expected_docs)
    recall_at_10 = r10_hits / len(expected_docs) if expected_docs else 0.0
    
    # Simple NDCG@10 (relevance = 1 for expected docs, 0 otherwise)
    dcg = 0.0
    for i, (doc_id, _) in enumerate(results[:10]):
        if doc_id in expected_docs:
            dcg += 1.0 / (1.0 + i)  # log2(i+2) ≈ 1+i for small i
    
    # IDCG (ideal DCG) - best possible ranking
    idcg = sum(1.0 / (1.0 + i) for i in range(min(len(expected_docs), 10)))
    ndcg_at_10 = dcg / idcg if idcg > 0 else 0.0
    
    return {
        "precision@5": precision_at_5,
        "recall@10": recall_at_10,
        "ndcg@10": ndcg_at_10
    }

def run_hybrid_evaluation(golden_set_file: str = "golden_set.yaml"):
    """Run comprehensive evaluation of all three modes."""
    
    print("🔥 COMPREHENSIVE HYBRID EVALUATION")
    print("=" * 60)
    print()
    
    # Load golden set
    queries = load_golden_set(golden_set_file)
    print(f"📋 Loaded {len(queries)} evaluation queries")
    print()
    
    # Initialize client
    client = HybridEvalClient()
    
    # Results storage
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "modes": {
            "minimal": {"queries": [], "metrics": {}},
            "tuned": {"queries": [], "metrics": {}}, 
            "hybrid": {"queries": [], "metrics": {}}
        }
    }
    
    # Process each query
    for i, query in enumerate(queries, 1):
        query_id = query["id"]
        query_text = query["query"]
        expected_docs = query["expected_doc_ids"]
        acl_hash = query.get("acl", "public")
        
        print(f"[{i:2d}/{len(queries)}] {query_id}: {query_text[:50]}...")
        
        try:
            # Get results from all modes using the compare method
            comparison = client.compare_hybrid_modes(
                query_text=query_text,
                expected_doc_ids=expected_docs,
                acl_hash=acl_hash,
                index_alias="confluence_current"
            )
            
            # Extract results for each mode
            for mode in ["minimal", "tuned", "hybrid"]:
                mode_results = comparison[mode]["results"]
                metrics = calculate_metrics(mode_results, expected_docs)
                
                query_result = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "expected_docs": expected_docs,
                    "retrieved_results": mode_results,
                    "metrics": metrics
                }
                
                all_results["modes"][mode]["queries"].append(query_result)
                
                # Print metrics
                if mode == "minimal":
                    print(f"         Minimal: P@5={metrics['precision@5']:.3f}, R@10={metrics['recall@10']:.3f}")
                elif mode == "tuned":  
                    p5_diff = metrics['precision@5'] - comparison['minimal']['precision_at_5']
                    print(f"         Tuned:   P@5={metrics['precision@5']:.3f}, R@10={metrics['recall@10']:.3f} ({p5_diff:+.3f})")
                else:  # hybrid
                    p5_diff = metrics['precision@5'] - comparison['minimal']['precision_at_5']
                    print(f"         Hybrid:  P@5={metrics['precision@5']:.3f}, R@10={metrics['recall@10']:.3f} ({p5_diff:+.3f})")
                    
        except Exception as e:
            logger.error(f"Failed to evaluate query {query_id}: {e}")
            
            # Add failed result
            for mode in ["minimal", "tuned", "hybrid"]:
                query_result = {
                    "query_id": query_id,
                    "query_text": query_text,
                    "expected_docs": expected_docs,
                    "retrieved_results": [],
                    "metrics": {"precision@5": 0.0, "recall@10": 0.0, "ndcg@10": 0.0},
                    "error": str(e)
                }
                all_results["modes"][mode]["queries"].append(query_result)
            
            print(f"         ❌ Failed: {e}")
    
    print()
    print("📊 CALCULATING OVERALL METRICS...")
    
    # Calculate overall metrics for each mode
    for mode_name, mode_data in all_results["modes"].items():
        queries_data = mode_data["queries"]
        
        if queries_data:
            # Calculate averages
            avg_p5 = sum(q["metrics"]["precision@5"] for q in queries_data) / len(queries_data)
            avg_r10 = sum(q["metrics"]["recall@10"] for q in queries_data) / len(queries_data)
            avg_ndcg = sum(q["metrics"]["ndcg@10"] for q in queries_data) / len(queries_data)
            
            mode_data["metrics"] = {
                "avg_precision@5": avg_p5,
                "avg_recall@10": avg_r10,
                "avg_ndcg@10": avg_ndcg,
                "total_queries": len(queries_data),
                "successful_queries": len([q for q in queries_data if "error" not in q])
            }
    
    # Print final comparison
    print()
    print("🏆 FINAL COMPARISON")
    print("=" * 60)
    
    minimal_metrics = all_results["modes"]["minimal"]["metrics"]
    tuned_metrics = all_results["modes"]["tuned"]["metrics"]
    hybrid_metrics = all_results["modes"]["hybrid"]["metrics"]
    
    print(f"{'Mode':<10} {'P@5':<8} {'R@10':<8} {'NDCG@10':<8} {'vs Minimal':<12}")
    print("-" * 60)
    print(f"{'Minimal':<10} {minimal_metrics['avg_precision@5']:.3f}    {minimal_metrics['avg_recall@10']:.3f}    {minimal_metrics['avg_ndcg@10']:.3f}    baseline")
    
    tuned_p5_diff = tuned_metrics['avg_precision@5'] - minimal_metrics['avg_precision@5']
    tuned_r10_diff = tuned_metrics['avg_recall@10'] - minimal_metrics['avg_recall@10']
    print(f"{'Tuned':<10} {tuned_metrics['avg_precision@5']:.3f}    {tuned_metrics['avg_recall@10']:.3f}    {tuned_metrics['avg_ndcg@10']:.3f}    P@5: {tuned_p5_diff:+.3f}")
    
    hybrid_p5_diff = hybrid_metrics['avg_precision@5'] - minimal_metrics['avg_precision@5']
    hybrid_r10_diff = hybrid_metrics['avg_recall@10'] - minimal_metrics['avg_recall@10']
    print(f"{'Hybrid':<10} {hybrid_metrics['avg_precision@5']:.3f}    {hybrid_metrics['avg_recall@10']:.3f}    {hybrid_metrics['avg_ndcg@10']:.3f}    P@5: {hybrid_p5_diff:+.3f}")
    
    print()
    
    # Determine winner
    best_mode = "minimal"
    best_p5 = minimal_metrics['avg_precision@5']
    
    if tuned_metrics['avg_precision@5'] > best_p5:
        best_mode = "tuned"
        best_p5 = tuned_metrics['avg_precision@5']
    
    if hybrid_metrics['avg_precision@5'] > best_p5:
        best_mode = "hybrid"
        best_p5 = hybrid_metrics['avg_precision@5']
    
    print(f"🎯 WINNER: {best_mode.upper()} mode with {best_p5:.3f} P@5")
    
    # Check recall preservation requirement
    recall_drop = minimal_metrics['avg_recall@10'] - hybrid_metrics['avg_recall@10']
    if recall_drop <= 0.05:  # Within 5%
        print(f"✅ Recall preservation: hybrid drops only {recall_drop:.3f} R@10 (≤5% threshold)")
    else:
        print(f"⚠️  Recall preservation: hybrid drops {recall_drop:.3f} R@10 (>5% threshold)")
    
    # Save results
    output_file = "hybrid_evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hybrid evaluation")
    parser.add_argument("--golden-set", default="golden_set.yaml", help="Golden set file")
    
    args = parser.parse_args()
    
    run_hybrid_evaluation(args.golden_set)