#!/usr/bin/env python3
"""
A/B Testing evaluation system for BM25 retrieval modes.

Features:
- Simple vs Tuned BM25 query evaluation
- Golden set evaluation with Recall@10, NDCG@10, Precision@5  
- Direct OpenSearch querying for real precision measurement
- Per-query diagnostics and comprehensive reporting
- Mode-specific output files (eval/latest_simple.json, eval/latest_tuned.json)
"""

import argparse
import json
import logging
import math
import time
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import numpy as np

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.os_eval_client import OpenSearchEvalClient

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation encounters critical errors."""
    pass


class ABTestEvaluator:
    """A/B Testing evaluator for simple vs tuned BM25 retrieval modes."""
    
    def __init__(self, mode: str = "tuned"):
        """Initialize evaluator with specified mode.
        
        Args:
            mode: 'simple' or 'tuned' BM25 query mode
        """
        if mode not in ["simple", "tuned"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'simple' or 'tuned'")
        
        self.mode = mode
        self.eval_client = OpenSearchEvalClient()
        self.results = []
    
    def calculate_precision_at_k(self, retrieved_ids: List[str], expected_ids: List[str], k: int = 5) -> float:
        """Calculate Precision@K."""
        if not retrieved_ids:
            return 0.0
        
        retrieved_k = set(retrieved_ids[:k])
        expected_set = set(expected_ids)
        
        intersection = retrieved_k & expected_set
        return len(intersection) / min(len(retrieved_ids), k)
    
    def calculate_recall_at_k(self, retrieved_ids: List[str], expected_ids: List[str], k: int = 10) -> float:
        """Calculate Recall@K."""
        if not expected_ids:
            return 1.0  # Perfect recall if no expected results
        
        retrieved_k = set(retrieved_ids[:k])
        expected_set = set(expected_ids)
        
        intersection = retrieved_k & expected_set
        return len(intersection) / len(expected_set)
    
    def calculate_ndcg_at_k(self, retrieved_ids: List[str], expected_ids: List[str], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K."""
        if not expected_ids:
            return 1.0
        
        # Create relevance scores (1 for expected docs, 0 for others)
        relevance_scores = []
        for doc_id in retrieved_ids[:k]:
            relevance_scores.append(1.0 if doc_id in expected_ids else 0.0)
        
        # Calculate DCG
        dcg = 0.0
        for i, rel in enumerate(relevance_scores):
            dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
        
        # Calculate IDCG (ideal DCG)
        ideal_relevance = [1.0] * min(len(expected_ids), k)
        idcg = 0.0
        for i, rel in enumerate(ideal_relevance):
            idcg += rel / math.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_query(self, query_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single query using the specified mode."""
        query_id = query_spec["id"]
        query_text = query_spec["query"]
        expected_ids = query_spec["expected_doc_ids"]
        acl = query_spec.get("acl", "public")
        
        start_time = time.time()
        result = {
            "query_id": query_id,
            "query_text": query_text,
            "expected_ids": expected_ids,
            "acl": acl,
            "mode": self.mode,
            "category": query_spec.get("category", "unknown")
        }
        
        try:
            # Use evaluation client to rank documents
            # Always pass ACL hash for proper filtering, including "public"
            acl_hash = acl
            
            ranked_docs = self.eval_client.rank_docs(
                query_text=query_text,
                mode=self.mode,
                acl_hash=acl_hash,
                index_alias="confluence_current"
            )
            
            # Extract doc IDs from results
            retrieved_ids = [doc_id for doc_id, score in ranked_docs]
            
            # Calculate metrics
            precision_5 = self.calculate_precision_at_k(retrieved_ids, expected_ids, 5)
            recall_10 = self.calculate_recall_at_k(retrieved_ids, expected_ids, 10)
            ndcg_10 = self.calculate_ndcg_at_k(retrieved_ids, expected_ids, 10)
            
            result.update({
                "retrieved_ids": retrieved_ids[:10],
                "retrieved_scores": [score for doc_id, score in ranked_docs[:10]],
                "metrics": {
                    "precision@5": precision_5,
                    "recall@10": recall_10,
                    "ndcg@10": ndcg_10
                },
                "retrieval_time_ms": (time.time() - start_time) * 1000,
                "success": True
            })
            
            # Analysis
            found_expected = [doc_id for doc_id in retrieved_ids[:10] if doc_id in expected_ids]
            missed_expected = [doc_id for doc_id in expected_ids if doc_id not in retrieved_ids[:10]]
            
            result["analysis"] = {
                "found_expected": found_expected,
                "missed_expected": missed_expected,
                "unexpected_results": [doc_id for doc_id in retrieved_ids[:5] if doc_id not in expected_ids]
            }
            
        except Exception as e:
            logger.error(f"Query {query_id} failed: {e}")
            result.update({
                "success": False,
                "error": str(e),
                "metrics": {"precision@5": 0.0, "recall@10": 0.0, "ndcg@10": 0.0}
            })
        
        return result


def run_evaluation(golden_set_file: Path, mode: str = "tuned") -> Dict[str, Any]:
    """Run A/B evaluation for specified BM25 mode."""
    
    print(f"🚀 Starting A/B Evaluation - {mode.upper()} BM25 Mode")
    print("="*60)
    
    # Load golden set
    print(f"📋 Loading golden set from {golden_set_file.name}...")
    with open(golden_set_file, 'r') as f:
        golden_set = yaml.safe_load(f)
    
    queries = golden_set["queries"]
    print(f"✅ Loaded {len(queries)} evaluation queries")
    
    # Initialize evaluator
    evaluator = ABTestEvaluator(mode=mode)
    
    # Run evaluation
    print(f"\n🧪 Running {mode} evaluation on {len(queries)} queries...")
    
    all_results = []
    category_stats = defaultdict(list)
    
    for i, query_spec in enumerate(queries, 1):
        print(f"   [{i:2d}/{len(queries)}] {query_spec['id']}: {query_spec['query'][:50]}...")
        
        result = evaluator.evaluate_query(query_spec)
        all_results.append(result)
        
        category = result["category"]
        if result["success"]:
            category_stats[category].append(result["metrics"])
        
        # Show quick results
        if result["success"]:
            p5 = result["metrics"]["precision@5"]
            r10 = result["metrics"]["recall@10"]
            print(f"      P@5: {p5:.3f} | R@10: {r10:.3f}")
    
    # Calculate aggregate metrics
    print(f"\n📊 Calculating metrics...")
    
    successful_results = [r for r in all_results if r["success"]]
    total_queries = len(all_results)
    successful_queries = len(successful_results)
    
    if successful_results:
        avg_precision_5 = np.mean([r["metrics"]["precision@5"] for r in successful_results])
        avg_recall_10 = np.mean([r["metrics"]["recall@10"] for r in successful_results])
        avg_ndcg_10 = np.mean([r["metrics"]["ndcg@10"] for r in successful_results])
    else:
        avg_precision_5 = avg_recall_10 = avg_ndcg_10 = 0.0
    
    # Category breakdowns
    category_metrics = {}
    for category, metrics_list in category_stats.items():
        if metrics_list:
            category_metrics[category] = {
                "count": len(metrics_list),
                "avg_precision@5": np.mean([m["precision@5"] for m in metrics_list]),
                "avg_recall@10": np.mean([m["recall@10"] for m in metrics_list]),
                "avg_ndcg@10": np.mean([m["ndcg@10"] for m in metrics_list])
            }
    
    # Thresholds from golden set
    thresholds = golden_set.get("evaluation", {}).get("thresholds", {})
    precision_threshold = thresholds.get("precision@5_min", 0.6)
    recall_threshold = thresholds.get("recall@10_min", 0.8)
    ndcg_threshold = thresholds.get("ndcg@10_min", 0.7)
    
    # Evaluate against thresholds
    meets_precision = avg_precision_5 >= precision_threshold
    meets_recall = avg_recall_10 >= recall_threshold
    meets_ndcg = avg_ndcg_10 >= ndcg_threshold
    
    overall_pass = meets_precision and meets_recall and meets_ndcg
    
    # Build final results
    evaluation_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "mode": mode,
            "golden_set_file": str(golden_set_file),
            "total_queries": total_queries,
            "index_alias": "confluence_current"
        },
        "overall_metrics": {
            "successful_queries": successful_queries,
            "failed_queries": total_queries - successful_queries,
            "success_rate": successful_queries / total_queries if total_queries > 0 else 0,
            "avg_precision@5": avg_precision_5,
            "avg_recall@10": avg_recall_10,
            "avg_ndcg@10": avg_ndcg_10
        },
        "threshold_evaluation": {
            "thresholds": {
                "precision@5": precision_threshold,
                "recall@10": recall_threshold,
                "ndcg@10": ndcg_threshold
            },
            "results": {
                "meets_precision": meets_precision,
                "meets_recall": meets_recall,
                "meets_ndcg": meets_ndcg,
                "overall_pass": overall_pass
            }
        },
        "category_breakdown": category_metrics,
        "query_results": all_results
    }
    
    return evaluation_results


def print_evaluation_summary(results: Dict[str, Any]):
    """Print formatted A/B evaluation summary."""
    
    mode = results["config"]["mode"].upper()
    print(f"\n" + "="*60)
    print(f"📊 A/B EVALUATION SUMMARY - {mode} BM25")
    print("="*60)
    
    overall = results["overall_metrics"]
    thresholds = results["threshold_evaluation"]
    
    print(f"🔢 Queries: {overall['successful_queries']}/{results['config']['total_queries']} successful")
    print(f"📈 Success Rate: {overall['success_rate']:.1%}")
    
    print(f"\n📋 Overall Metrics ({mode} BM25):")
    print(f"   Precision@5:  {overall['avg_precision@5']:.3f} (threshold: {thresholds['thresholds']['precision@5']:.2f}) {'✅' if thresholds['results']['meets_precision'] else '❌'}")
    print(f"   Recall@10:    {overall['avg_recall@10']:.3f} (threshold: {thresholds['thresholds']['recall@10']:.2f}) {'✅' if thresholds['results']['meets_recall'] else '❌'}")
    print(f"   NDCG@10:      {overall['avg_ndcg@10']:.3f} (threshold: {thresholds['thresholds']['ndcg@10']:.2f}) {'✅' if thresholds['results']['meets_ndcg'] else '❌'}")
    
    print(f"\n🎯 {mode} Result: {'✅ PASS' if thresholds['results']['overall_pass'] else '❌ FAIL'}")
    
    # Category breakdown
    if results["category_breakdown"]:
        print(f"\n📂 Category Breakdown:")
        for category, metrics in results["category_breakdown"].items():
            print(f"   {category:15} ({metrics['count']:2d} queries): P@5={metrics['avg_precision@5']:.3f}, R@10={metrics['avg_recall@10']:.3f}, NDCG@10={metrics['avg_ndcg@10']:.3f}")
    
    # Failed queries
    failed_queries = [q for q in results["query_results"] if not q["success"]]
    if failed_queries:
        print(f"\n❌ Failed Queries:")
        for q in failed_queries:
            print(f"   {q['query_id']}: {q.get('error', 'Unknown error')}")
    
    # Low-performing queries
    low_precision = [q for q in results["query_results"] 
                    if q["success"] and q["metrics"]["precision@5"] < 0.4]
    if low_precision:
        print(f"\n⚠️  Low Precision Queries (P@5 < 0.4):")
        for q in low_precision[:5]:  # Show first 5
            print(f"   {q['query_id']} (P@5={q['metrics']['precision@5']:.3f}): {q['query_text'][:50]}...")
    
    # Best performing queries
    high_precision = [q for q in results["query_results"] 
                     if q["success"] and q["metrics"]["precision@5"] >= 0.8]
    if high_precision:
        print(f"\n🎯 High Precision Queries (P@5 ≥ 0.8):")
        for q in sorted(high_precision, key=lambda x: x["metrics"]["precision@5"], reverse=True)[:3]:
            print(f"   {q['query_id']} (P@5={q['metrics']['precision@5']:.3f}): {q['query_text'][:50]}...")
    
    print("\n" + "="*60)


def main():
    """Main A/B evaluation function."""
    
    parser = argparse.ArgumentParser(description="Run A/B evaluation for BM25 retrieval modes")
    parser.add_argument("--mode", choices=["simple", "tuned"], required=True,
                       help="BM25 query mode: 'simple' or 'tuned'")
    parser.add_argument("--golden-set", default="eval/golden_set.yaml",
                       help="Path to golden set YAML file")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Resolve file paths
    golden_set_file = Path(args.golden_set)
    output_file = Path(f"eval/latest_{args.mode}.json")
    
    if not golden_set_file.exists():
        raise FileNotFoundError(f"Golden set file not found: {golden_set_file}")
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run evaluation
        results = run_evaluation(
            golden_set_file=golden_set_file,
            mode=args.mode
        )
        
        # Print summary
        print_evaluation_summary(results)
        
        # Save results with custom JSON encoder
        def json_serializer(obj):
            """Custom JSON serializer for numpy types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=json_serializer)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Print summary table for easy comparison
        overall = results["overall_metrics"]
        print(f"\n📋 SUMMARY TABLE ({args.mode.upper()})")
        print("-" * 50)
        print(f"Mode:         {args.mode}")
        print(f"Precision@5:  {overall['avg_precision@5']:.3f}")
        print(f"Recall@10:    {overall['avg_recall@10']:.3f}")
        print(f"NDCG@10:      {overall['avg_ndcg@10']:.3f}")
        print(f"Success Rate: {overall['success_rate']:.1%}")
        print("-" * 50)
        
        # Exit with appropriate code
        overall_pass = results["threshold_evaluation"]["results"]["overall_pass"]
        exit_code = 0 if overall_pass else 1
        
        print(f"\n🏁 {args.mode.upper()} Evaluation {'PASSED' if overall_pass else 'FAILED'}")
        return exit_code
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ Evaluation failed: {e}")
        if "404" in str(e) or "Connection" in str(e):
            print("\n💡 Make sure OpenSearch is running and the index 'confluence_current' exists")
            print("   You may need to run: scripts/start_opensearch_local.sh")
            print("   And index data: scripts/index_mock_docs.py")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)