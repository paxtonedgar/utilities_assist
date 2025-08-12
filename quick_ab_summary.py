#!/usr/bin/env python3
"""
Quick A/B summary based on the evaluation outputs.
"""

def quick_ab_summary():
    """Display the A/B comparison based on evaluation outputs."""
    
    print("🆚 A/B EVALUATION SUMMARY")
    print("=" * 50)
    print()
    
    print("📊 OVERALL METRICS COMPARISON")
    print("-" * 40)
    print(f"{'Metric':<15} {'Simple':<10} {'Tuned':<10} {'Diff':<10}")
    print("-" * 40)
    
    # Results from the evaluation outputs
    simple_p5 = 0.353
    simple_r10 = 0.372
    simple_ndcg = 0.386
    
    tuned_p5 = 0.217
    tuned_r10 = 0.139
    tuned_ndcg = 0.167
    
    print(f"{'precision@5':<15} {simple_p5:.3f}      {tuned_p5:.3f}      {tuned_p5-simple_p5:+.3f}")
    print(f"{'recall@10':<15} {simple_r10:.3f}      {tuned_r10:.3f}      {tuned_r10-simple_r10:+.3f}")
    print(f"{'ndcg@10':<15} {simple_ndcg:.3f}      {tuned_ndcg:.3f}      {tuned_ndcg-simple_ndcg:+.3f}")
    
    print()
    print("🎯 KEY INSIGHTS")
    print("-" * 30)
    
    if simple_p5 > tuned_p5:
        diff = simple_p5 - tuned_p5
        print(f"❌ Simple mode WINS by {diff:.3f} P@5")
        print(f"   Tuned boosts are hurting precision")
    else:
        diff = tuned_p5 - simple_p5
        print(f"✅ Tuned mode WINS by {diff:.3f} P@5")
        print(f"   Tuned boosts are helping!")
    
    print()
    print("📈 PROGRESS vs BASELINE")
    print("-" * 30)
    baseline_p5 = 0.300  # Original simple mode before fixes
    
    simple_improvement = simple_p5 - baseline_p5
    tuned_improvement = tuned_p5 - baseline_p5
    
    print(f"Baseline (before fixes): {baseline_p5:.3f} P@5")
    print(f"Simple (with key terms): {simple_p5:.3f} P@5 ({simple_improvement:+.3f})")
    print(f"Tuned (with key terms):  {tuned_p5:.3f} P@5 ({tuned_improvement:+.3f})")
    
    print()
    print("🔍 NEXT STEPS")
    print("-" * 30)
    if simple_p5 > tuned_p5:
        print("• Tuned query complexity is reducing recall")
        print("• Consider simpler boosts or tune parameters")
        print("• Focus on recall first, precision second")
    else:
        print("• Tuned mode is working - optimize further!")
        print("• Try different boost values")
        print("• Add more sophisticated features")


if __name__ == "__main__":
    quick_ab_summary()