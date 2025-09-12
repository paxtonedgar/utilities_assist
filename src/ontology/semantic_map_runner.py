from __future__ import annotations

"""
CLI runner: builds the Phase 1 Document Semantic Map for a given diagnostics folder.

Steps:
 1) Build fingerprints from diagnostics (keywords + patterns)
 2) Embed and cluster (BERTopic if available else TF‑IDF + KMeans)
 3) Label topics via shared chat client (few‑shot)
 4) Fetch docs and parse structure; classify segments
 5) Write doc_map.jsonl and topic/taxonomy artifacts
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

from src.infra.resource_manager import initialize_resources
from src.infra.settings import get_settings
from .semantic_fingerprints import build_fingerprints
from .semantic_cluster import embed_tokens, cluster_with_bertopic, cluster_with_tfidf, build_topic_cards, llm_label_topics
from src.infra.opensearch_client import OpenSearchClient
from .structure_parser import parse_hit_to_segments
from .index_profiles import detect_profile, get_profile_config

logger = logging.getLogger(__name__)


def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run(diag_dir: str, out_dir: str, k: int = 12, max_docs: int = 500):
    dpath = Path(diag_dir)
    opath = Path(out_dir)

    settings = get_settings()
    resources = initialize_resources(settings)

    # 1) fingerprints
    fps = build_fingerprints(dpath)
    if not fps:
        logger.warning("No fingerprints found; ensure diagnostics were generated")
        return
    docs = fps[:max_docs]

    # 2) clustering
    vectors = embed_tokens(resources, docs)
    labels = None
    topic_model = None
    if vectors:
        res = cluster_with_bertopic(vectors, docs)
        if res:
            topic_model, topics, probs = res
            labels = topics
    if labels is None:
        km = cluster_with_tfidf(docs, k=k) or {}
        labels = km.get("labels")
    if labels is None:
        logger.error("Clustering failed")
        return

    # 3) topic cards + LLM labeling
    # Normalize labels to plain ints (avoid numpy types)
    labels = [int(x) for x in labels]
    cards = build_topic_cards(docs, labels)
    topic_labels = llm_label_topics(resources, cards)
    topics_payload = {str(tid): {**cards.get(tid, {}), **topic_labels.get(tid, {})} for tid in cards}
    _write_json(opath / "topics.json", topics_payload)

    # 4) structure parse + semantic map
    client = OpenSearchClient()
    doc_map: List[Dict[str, Any]] = []
    doc_graph: List[Dict[str, Any]] = []
    for d, lab in zip(docs, labels):
        doc_id = d.get("doc_id")
        index = d.get("index")
        profile = detect_profile(index or "")
        try:
            hit = client.get_doc_by_id(index=index, doc_id=doc_id)
            segments, links = parse_hit_to_segments(hit, resources=resources, index_profile=profile)
        except Exception:
            segments, links = [], []
        tl = topic_labels.get(int(lab), {"label": "misc", "confidence": 0.5})
        hints = get_profile_config(profile)
        doc_map.append({
            "doc_id": doc_id,
            "index": index,
            "topic_id": int(lab),
            "doc_type": tl.get("label"),
            "label_confidence": tl.get("confidence", 0.5),
            "segments": segments,
            "extraction_hints": hints,
        })
        for url in links:
            doc_graph.append({"doc_id": doc_id, "href": url})

    _write_jsonl(opath / "doc_map.jsonl", doc_map)
    if doc_graph:
        _write_jsonl(opath / "doc_graph.jsonl", doc_graph)


def main():
    ap = argparse.ArgumentParser(description="Build Document Semantic Map (Phase 1)")
    ap.add_argument("--diag-dir", type=str, required=True, help="Diagnostics folder for an index")
    ap.add_argument("--out", type=str, default="outputs/semantic_map", help="Output dir")
    ap.add_argument("--k", type=int, default=12, help="Num clusters for TF-IDF fallback")
    ap.add_argument("--max-docs", type=int, default=500, help="Max docs to process")
    args = ap.parse_args()
    run(args.diag_dir, args.out, k=args.k, max_docs=args.max_docs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
