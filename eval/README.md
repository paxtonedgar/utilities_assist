# A/B Evaluation Framework

Enterprise-grade A/B testing system for BM25 retrieval optimization.

## Quick Start

```bash
# Run Simple BM25 evaluation  
python eval/run_eval.py --mode simple

# Run Tuned BM25 evaluation
python eval/run_eval.py --mode tuned

# Compare results
python eval/compare_modes.py
```

## Files

### Core Evaluation
- **`run_eval.py`** - A/B evaluation runner with `--mode simple|tuned`
- **`os_eval_client.py`** - OpenSearch client for real precision measurement
- **`compare_modes.py`** - Side-by-side A/B analysis and recommendations

### Test Data
- **`golden_set.yaml`** - 30 curated test queries with expected document IDs
- **`mock_corpus/utilities_docs.jsonl`** - Mock document corpus for testing

### Results
- **`latest_simple.json`** - Simple BM25 evaluation results
- **`latest_tuned.json`** - Tuned BM25 evaluation results

## Key Features

✅ **Real OpenSearch Integration** - Direct queries to actual search index  
✅ **Comprehensive Metrics** - Precision@5, Recall@10, NDCG@10  
✅ **Category Analysis** - Performance breakdown by query type  
✅ **ACL Testing** - Validates access control boundaries  
✅ **Version Conflicts** - Tests handling of newer vs older documents  

## Output Format

Each evaluation produces detailed JSON with:
- Per-query metrics and retrieved document IDs
- Category-level performance breakdown  
- Threshold analysis (target: P@5 ≥ 0.6)
- Low/high performing query identification
- Complete analysis and diagnostics

## Expected Results

With live OpenSearch + indexed data:
- **Simple BM25**: Baseline precision performance
- **Tuned BM25**: Expected +0.080 P@5 improvement from:
  - Multi-match with best_fields (title^10, section^4, body^1)
  - Dynamic minimum_should_match (60%/75%)
  - Enhanced generic penalties (-1.2x)
  - Stronger recency decay (75d half-life)
  - Phrase and proximity boosting

## Prerequisites

1. OpenSearch running on `localhost:9200`
2. Index `confluence_current` with document data
3. Python dependencies: `numpy`, `pyyaml`, `requests`

## Enterprise Usage

This framework provides production-ready evaluation capabilities for measuring retrieval improvements in enterprise search systems.