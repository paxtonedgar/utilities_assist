#!/usr/bin/env python3
"""
Compare A/B results between simple and tuned modes.
"""

import json
from pathlib import Path

def compare_ab_results():
    """Compare the latest simple and tuned evaluation results."""
    
    eval_dir = Path("eval")
    simple_file = eval_dir / "latest_simple.json"
    tuned_file = eval_dir / "latest_tuned.json"
    
    # Load results
    if not simple_file.exists() or not tuned_file.exists():
        print("❌ Results files not found. Please run both evaluations first.")
        return
        
    with open(simple_file) as f:
        simple_data = json.load(f)
        
    with open(tuned_file) as f:
        tuned_data = json.load(f)
    
    print("🆚 A/B EVALUATION COMPARISON")
    print("=" * 60)
    print()
    
    # Overall metrics comparison
    simple_metrics = simple_data["overall_metrics"]
    tuned_metrics = tuned_data["overall_metrics"]
    
    print("📊 OVERALL METRICS COMPARISON")
    print("-" * 40)
    print(f"{'Metric':<15} {'Simple':<10} {'Tuned':<10} {'Diff':<10}")
    print("-" * 40)
    
    # Map metric names
    metric_mapping = {
        "precision@5": "avg_precision@5",
        "recall@10": "avg_recall@10", 
        "ndcg@10": "avg_ndcg@10"
    }
    
    for display_name, key in metric_mapping.items():
        simple_val = simple_metrics[key]
        tuned_val = tuned_metrics[key] 
        diff = tuned_val - simple_val
        diff_str = f"{diff:+.3f}"
        
        print(f"{display_name:<15} {simple_val:.3f}      {tuned_val:.3f}      {diff_str}")
    
    print()
    
    # Category comparison
    print("📂 CATEGORY BREAKDOWN COMPARISON")
    print("-" * 60)
    print(f"{'Category':<18} {'Simple P@5':<12} {'Tuned P@5':<12} {'Improvement':<12}")
    print("-" * 60)
    
    simple_categories = simple_data["category_breakdown"]
    tuned_categories = tuned_data["category_breakdown"]
    
    total_improvement = 0
    categories_with_data = 0
    
    for category in simple_categories:
        if category in tuned_categories:
            simple_p5 = simple_categories[category]["precision_at_5"]
            tuned_p5 = tuned_categories[category]["precision_at_5"]
            improvement = tuned_p5 - simple_p5
            
            # Only count categories that have some results
            if simple_p5 > 0 or tuned_p5 > 0:
                total_improvement += improvement
                categories_with_data += 1
            
            improvement_str = f"{improvement:+.3f}"
            if improvement > 0:
                improvement_str = f"✅ {improvement_str}"
            elif improvement < 0:
                improvement_str = f"❌ {improvement_str}" 
            else:
                improvement_str = f"   {improvement_str}"
                
            print(f"{category:<18} {simple_p5:.3f}        {tuned_p5:.3f}        {improvement_str}")
    
    print("-" * 60)
    if categories_with_data > 0:
        avg_improvement = total_improvement / categories_with_data
        print(f"Average improvement: {avg_improvement:+.3f}")
    
    print()
    
    # Query-level wins/losses
    simple_queries = {q["query_id"]: q for q in simple_data["queries"]}
    tuned_queries = {q["query_id"]: q for q in tuned_data["queries"]}
    
    wins = 0
    losses = 0
    ties = 0
    
    print("🏆 QUERY-LEVEL COMPARISON (Top 10 Improvements)")
    print("-" * 60)
    
    improvements = []
    
    for query_id in simple_queries:
        if query_id in tuned_queries:
            simple_p5 = simple_queries[query_id]["precision_at_5"]
            tuned_p5 = tuned_queries[query_id]["precision_at_5"]
            improvement = tuned_p5 - simple_p5
            
            if improvement > 0:
                wins += 1
            elif improvement < 0:
                losses += 1
            else:
                ties += 1
                
            improvements.append({
                "query_id": query_id,
                "query": simple_queries[query_id]["query_text"][:50] + "...",
                "simple_p5": simple_p5,
                "tuned_p5": tuned_p5,
                "improvement": improvement
            })
    
    # Sort by improvement and show top 10
    improvements.sort(key=lambda x: x["improvement"], reverse=True)
    
    for i, result in enumerate(improvements[:10]):
        status = "✅" if result["improvement"] > 0 else "❌" if result["improvement"] < 0 else "="
        print(f"{result['query_id']}: {status} {result['improvement']:+.3f} ({result['simple_p5']:.3f} → {result['tuned_p5']:.3f})")
        print(f"     {result['query']}")
    
    print()
    print("🎯 SUMMARY")
    print("-" * 30)
    print(f"Tuned wins: {wins}")
    print(f"Tuned losses: {losses}")  
    print(f"Ties: {ties}")
    
    overall_improvement = tuned_metrics["avg_precision@5"] - simple_metrics["avg_precision@5"]
    if overall_improvement > 0:
        print(f"✅ Tuned mode is better by {overall_improvement:+.3f} P@5")
    elif overall_improvement < 0:
        print(f"❌ Simple mode is better by {abs(overall_improvement):.3f} P@5")
    else:
        print("🟰 Modes are tied")


if __name__ == "__main__":
    compare_ab_results()