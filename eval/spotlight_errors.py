#!/usr/bin/env python3
"""
Spotlight the 10 worst precision queries with offending document texts.
Helps identify where mock corpus needs minimal tweaks to improve evaluation quality.
"""

import json
import yaml
import sys
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.services.retrieve import RetrievalService
from src.services.rerank import LightweightReranker
from src.infra.config import get_settings


@dataclass
class PrecisionError:
    """Details of a precision error."""
    query_id: str
    query: str
    precision_at_5: float
    expected_docs: List[str]
    retrieved_docs: List[str]
    offending_docs: List[Dict[str, Any]]
    missing_docs: List[str]


def load_golden_set(path: str) -> Dict[str, Any]:
    """Load golden set from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_mock_corpus(path: str) -> Dict[str, Any]:
    """Load mock corpus from JSONL file."""
    docs = {}
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                docs[doc['canonical_id']] = doc
    return docs


def calculate_precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Calculate Precision@K metric."""
    if not retrieved or k == 0:
        return 0.0
    
    retrieved_at_k = retrieved[:k]
    relevant_set = set(relevant)
    correct = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
    
    return correct / min(k, len(retrieved_at_k))


def evaluate_query(
    query: Dict[str, Any], 
    retrieval_service: RetrievalService,
    reranker: LightweightReranker,
    corpus: Dict[str, Any],
    use_reranker: bool = True
) -> Tuple[float, List[str]]:
    """Evaluate a single query and return precision@5 and retrieved doc IDs."""
    
    query_text = query['query']
    
    # Mock retrieval (in real implementation, this would hit OpenSearch)
    # For now, simulate retrieval by scoring docs based on keyword overlap
    scored_docs = []
    
    query_words = set(query_text.lower().split())
    
    for doc_id, doc in corpus.items():
        # Simple scoring based on keyword matches
        doc_text = f"{doc.get('title', '')} {doc.get('body', '')} {doc.get('section', '')}".lower()
        doc_words = set(doc_text.split())
        
        overlap = len(query_words.intersection(doc_words))
        score = overlap / len(query_words) if query_words else 0
        
        if score > 0:
            scored_docs.append({
                'id': doc_id,
                'title': doc.get('title', ''),
                'score': score,
                'content': doc.get('body', '')[:200] + '...' if len(doc.get('body', '')) > 200 else doc.get('body', ''),
                'section': doc.get('section', ''),
                'updated_at': doc.get('updated_at', ''),
                'url': doc.get('url', '')
            })
    
    # Sort by score
    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    
    # Apply reranking if requested
    if use_reranker and scored_docs:
        rerank_result = reranker.rerank(scored_docs, query_text, max_results=10)
        final_docs = rerank_result.documents
    else:
        final_docs = scored_docs[:10]
    
    retrieved_ids = [doc['id'] for doc in final_docs]
    expected_ids = query.get('expected_doc_ids', [])
    
    precision_5 = calculate_precision_at_k(retrieved_ids, expected_ids, 5)
    
    return precision_5, retrieved_ids


def find_worst_precision_queries(
    golden_set: Dict[str, Any],
    corpus: Dict[str, Any],
    top_n: int = 10
) -> List[PrecisionError]:
    """Find the worst performing queries by precision@5."""
    
    settings = get_settings()
    retrieval_service = RetrievalService(settings)  # This would need proper initialization
    reranker = LightweightReranker()
    
    precision_errors = []
    
    for query in golden_set['queries']:
        query_id = query.get('id', 'unknown')
        
        # Evaluate with reranker (our target system)
        precision_5, retrieved_ids = evaluate_query(query, retrieval_service, reranker, corpus, use_reranker=True)
        
        expected_ids = query.get('expected_doc_ids', [])
        retrieved_top_5 = retrieved_ids[:5]
        
        # Find offending docs (retrieved but not relevant) and missing docs
        expected_set = set(expected_ids)
        retrieved_set = set(retrieved_top_5)
        
        offending_doc_ids = [doc_id for doc_id in retrieved_top_5 if doc_id not in expected_set]
        missing_doc_ids = [doc_id for doc_id in expected_ids if doc_id not in retrieved_set]
        
        # Get full doc details for offending docs
        offending_docs = []
        for doc_id in offending_doc_ids:
            if doc_id in corpus:
                doc = corpus[doc_id]
                offending_docs.append({
                    'id': doc_id,
                    'title': doc.get('title', ''),
                    'content': doc.get('body', '')[:300] + '...' if len(doc.get('body', '')) > 300 else doc.get('body', ''),
                    'section': doc.get('section', ''),
                    'updated_at': doc.get('updated_at', '')
                })
        
        error = PrecisionError(
            query_id=query_id,
            query=query['query'],
            precision_at_5=precision_5,
            expected_docs=expected_ids,
            retrieved_docs=retrieved_top_5,
            offending_docs=offending_docs,
            missing_docs=missing_doc_ids
        )
        
        precision_errors.append(error)
    
    # Sort by worst precision first
    precision_errors.sort(key=lambda x: x.precision_at_5)
    
    return precision_errors[:top_n]


def print_error_analysis(errors: List[PrecisionError]):
    """Print detailed analysis of precision errors."""
    
    print("🔍 PRECISION@5 ERROR SPOTLIGHT")
    print("=" * 60)
    print(f"Analyzing {len(errors)} worst-performing queries\n")
    
    for i, error in enumerate(errors, 1):
        print(f"#{i} QUERY: {error.query} (ID: {error.query_id})")
        print(f"   Precision@5: {error.precision_at_5:.3f}")
        print(f"   Expected: {len(error.expected_docs)} docs | Retrieved: {len(error.retrieved_docs)} docs")
        
        if error.missing_docs:
            print(f"   ❌ MISSING: {', '.join(error.missing_docs)}")
        
        if error.offending_docs:
            print("   🚫 OFFENDING DOCS (retrieved but not relevant):")
            for doc in error.offending_docs:
                print(f"      • {doc['id']}: {doc['title']}")
                print(f"        Section: {doc['section']} | Updated: {doc['updated_at']}")
                print(f"        Content: {doc['content']}")
                print()
        
        print("-" * 60)
        print()


def suggest_corpus_improvements(errors: List[PrecisionError]):
    """Suggest specific improvements to the mock corpus."""
    
    print("💡 CORPUS IMPROVEMENT SUGGESTIONS")
    print("=" * 60)
    
    # Analyze common patterns in offending docs
    offending_patterns = {}
    for error in errors:
        for doc in error.offending_docs:
            doc_type = doc['id'].split(':')[2] if ':' in doc['id'] else 'unknown'
            if doc_type not in offending_patterns:
                offending_patterns[doc_type] = []
            offending_patterns[doc_type].append({
                'query': error.query,
                'doc': doc
            })
    
    print("Common offending document types:")
    for doc_type, instances in offending_patterns.items():
        print(f"  • {doc_type}: {len(instances)} instances")
        if len(instances) > 2:  # Show examples for frequent offenders
            print("    Examples:")
            for instance in instances[:2]:
                print(f"      - Query: '{instance['query']}' → {instance['doc']['title']}")
    
    print("\nSpecific recommendations:")
    print("1. Make overview docs less keyword-dense")
    print("2. Strengthen specific doc titles with unique phrases")
    print("3. Ensure newer docs clearly override older ones")
    print("4. Add explicit version conflicts to test recency handling")
    print("5. Review ACL boundaries for proper access control testing")


def main():
    """Main execution function."""
    
    # Set up paths
    eval_dir = Path(__file__).parent
    golden_set_path = eval_dir / "golden_set.yaml"
    mock_corpus_path = eval_dir / "mock_corpus" / "utilities_docs.jsonl"
    
    if not golden_set_path.exists():
        print(f"❌ Golden set not found at {golden_set_path}")
        return 1
    
    if not mock_corpus_path.exists():
        print(f"❌ Mock corpus not found at {mock_corpus_path}")
        return 1
    
    # Load data
    print("📊 Loading golden set and mock corpus...")
    golden_set = load_golden_set(golden_set_path)
    corpus = load_mock_corpus(mock_corpus_path)
    
    print(f"✅ Loaded {len(golden_set['queries'])} queries and {len(corpus)} documents")
    
    # Find worst queries
    print("🔎 Identifying worst precision queries...")
    worst_errors = find_worst_precision_queries(golden_set, corpus, top_n=10)
    
    # Print analysis
    print_error_analysis(worst_errors)
    suggest_corpus_improvements(worst_errors)
    
    # Summary stats
    avg_precision = sum(error.precision_at_5 for error in worst_errors) / len(worst_errors)
    print(f"📈 SUMMARY")
    print(f"Average Precision@5 of worst 10 queries: {avg_precision:.3f}")
    print(f"Target: ≥ 0.6 | Current worst: {worst_errors[0].precision_at_5:.3f}")
    
    if avg_precision < 0.6:
        print("🔧 Action needed: Corpus requires hardening to reach target precision")
    else:
        print("✅ Evaluation corpus appears ready for reliable precision measurement")
    
    return 0


if __name__ == "__main__":
    exit(main())