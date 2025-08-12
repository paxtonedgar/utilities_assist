#!/usr/bin/env python3
"""
Compare Simple vs Tuned BM25 evaluation results.
Analyzes A/B test results and shows improvement metrics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any


def load_results(file_path: Path) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def compare_modes(simple_file: Path, tuned_file: Path):
    """Compare simple vs tuned BM25 evaluation results."""
    
    try:
        simple_results = load_results(simple_file)
        tuned_results = load_results(tuned_file)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("\nRun both modes first:")
        print("  python eval/run_eval.py --mode simple")
        print("  python eval/run_eval.py --mode tuned")
        return 1
    
    print("🔄 A/B COMPARISON - SIMPLE vs TUNED BM25")
    print("=" * 60)
    
    simple_metrics = simple_results["overall_metrics"]
    tuned_metrics = tuned_results["overall_metrics"]
    
    # Overall comparison
    print("\n📊 OVERALL METRICS COMPARISON")
    print("-" * 60)
    print(f"{'Metric':<15} {'Simple':<10} {'Tuned':<10} {'Δ':<10} {'% Change'}")
    print("-" * 60)
    
    metrics = [
        ("Precision@5", "avg_precision@5"),
        ("Recall@10", "avg_recall@10"), 
        ("NDCG@10", "avg_ndcg@10"),
        ("Success Rate", "success_rate")
    ]
    
    for metric_name, metric_key in metrics:
        simple_val = simple_metrics[metric_key]
        tuned_val = tuned_metrics[metric_key]
        delta = tuned_val - simple_val
        
        if simple_val > 0:
            pct_change = (delta / simple_val) * 100
            pct_str = f"{pct_change:+.1f}%"
        else:
            pct_str = "N/A" if delta == 0 else "∞"
        
        print(f"{metric_name:<15} {simple_val:<10.3f} {tuned_val:<10.3f} {delta:<+10.3f} {pct_str}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS")
    print("-" * 60)
    
    precision_improvement = tuned_metrics["avg_precision@5"] - simple_metrics["avg_precision@5"]
    recall_improvement = tuned_metrics["avg_recall@10"] - simple_metrics["avg_recall@10"]
    
    if precision_improvement > 0.05:
        print(f"✅ Significant precision improvement: +{precision_improvement:.3f}")
    elif precision_improvement > 0:
        print(f"📈 Modest precision improvement: +{precision_improvement:.3f}")
    elif precision_improvement == 0:
        print(f"➖ No precision change")
    else:
        print(f"⚠️  Precision regression: {precision_improvement:.3f}")
    
    # Threshold analysis
    simple_thresholds = simple_results["threshold_evaluation"]
    tuned_thresholds = tuned_results["threshold_evaluation"]
    
    print(f"\n🎯 THRESHOLD ANALYSIS")
    print("-" * 60)
    
    print(f"Target Precision@5 ≥ 0.6:")
    print(f"  Simple BM25: {'✅' if simple_thresholds['results']['meets_precision'] else '❌'} {simple_metrics['avg_precision@5']:.3f}")
    print(f"  Tuned BM25:  {'✅' if tuned_thresholds['results']['meets_precision'] else '❌'} {tuned_metrics['avg_precision@5']:.3f}")
    
    # Category comparison
    if simple_results["category_breakdown"] and tuned_results["category_breakdown"]:
        print(f"\n📂 CATEGORY COMPARISON")
        print("-" * 60)
        
        categories = set(simple_results["category_breakdown"].keys()) | set(tuned_results["category_breakdown"].keys())
        
        for category in sorted(categories):
            simple_cat = simple_results["category_breakdown"].get(category, {"avg_precision@5": 0.0})
            tuned_cat = tuned_results["category_breakdown"].get(category, {"avg_precision@5": 0.0})
            
            simple_p5 = simple_cat["avg_precision@5"]
            tuned_p5 = tuned_cat["avg_precision@5"]
            delta_p5 = tuned_p5 - simple_p5
            
            print(f"  {category:15} Simple: {simple_p5:.3f} | Tuned: {tuned_p5:.3f} | Δ: {delta_p5:+.3f}")
    
    # Overall recommendation
    print(f"\n🏁 RECOMMENDATION")
    print("-" * 60)
    
    simple_pass = simple_thresholds["results"]["overall_pass"]
    tuned_pass = tuned_thresholds["results"]["overall_pass"]
    
    if tuned_pass and not simple_pass:
        print("🎉 Use TUNED BM25 - meets target thresholds")
    elif tuned_pass and simple_pass:
        if precision_improvement > 0.01:
            print("🎉 Use TUNED BM25 - better performance")
        else:
            print("🤔 Either mode works - minimal difference")
    elif not tuned_pass and not simple_pass:
        if precision_improvement > 0:
            print("📈 Use TUNED BM25 - shows improvement trend")
        else:
            print("🔄 Both modes need more tuning")
    else:
        print("⚠️  Investigate tuning - simple mode performing better")
    
    return 0


def main():
    """Main comparison function."""
    
    parser = argparse.ArgumentParser(description="Compare Simple vs Tuned BM25 evaluation results")
    parser.add_argument("--simple", default="eval/latest_simple.json",
                       help="Path to simple mode results")
    parser.add_argument("--tuned", default="eval/latest_tuned.json", 
                       help="Path to tuned mode results")
    
    args = parser.parse_args()
    
    simple_file = Path(args.simple)
    tuned_file = Path(args.tuned)
    
    return compare_modes(simple_file, tuned_file)


if __name__ == "__main__":
    exit(main())