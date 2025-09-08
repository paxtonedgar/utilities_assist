"""
ID Queue workflow: build a queue of document ids via match_all (non-PIT
search_after) and then process docs by id until the queue is exhausted.

Queue entries are NDJSON lines with {"_id": str, "_index": str}.
Processed ids are tracked in a ledger to support resume and deduplication.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable

from src.infra.resource_manager import initialize_resources
from src.infra.settings import get_settings
from src.ontology.doc_by_doc import process_doc


# Note: _resolve_alias_or_index removed - not needed when using search_client.iterate_ids


def build_queue(index: str | None, out_file: str, batch: int, limit: int | None) -> int:
    # Initialize resources like Streamlit app does - this provides working authentication
    settings = get_settings()
    resources = initialize_resources(settings)
    search_client = resources.search_client
    
    idx = index or settings.search_index_alias
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        # Use the same search client that works in Streamlit - handles authentication properly
        for hit in search_client.iterate_ids(index=idx, batch_size=batch, max_docs=limit):
            _id = hit.get("_id")
            _index = hit.get("_index", idx)
            if _id:
                f.write(json.dumps({"_id": _id, "_index": _index}) + "\n")
                n += 1
    print(f"Queue built: {n} ids -> {path}")
    return n


# Note: Raw request functions removed - now using search_client.iterate_ids from resource manager


def _load_ids(queue_file: Path) -> Iterable[Dict[str, str]]:
    with queue_file.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                if "_id" in obj:
                    yield obj
            except Exception:
                # Allow plain IDs too
                yield {"_id": ln}


def _load_ledger(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def _append_ledger(path: Path, doc_id: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(doc_id + "\n")


def process_queue(queue_file: str, out_dir: str, resume: bool, ledger_file: str | None) -> int:
    # Initialize resources like Streamlit app does - this provides working authentication
    settings = get_settings()
    resources = initialize_resources(settings)
    search_client = resources.search_client
    
    qpath = Path(queue_file)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ledger = Path(ledger_file) if ledger_file else (out_path / "processed.ids")
    seen = _load_ledger(ledger) if resume else set()

    steps_f = (out_path / "steps.ndjson").open("a", encoding="utf-8")
    edges_f = (out_path / "edges.ndjson").open("a", encoding="utf-8")
    meta_f = (out_path / "docs.ndjson").open("a", encoding="utf-8")
    processed = 0
    skipped = 0
    try:
        for item in _load_ids(qpath):
            did = item.get("_id")
            if not did or did in seen:
                skipped += 1
                continue
            idx = item.get("_index") or settings.search_index_alias
            try:
                hit = search_client.get_doc_by_id(index=idx, doc_id=did)
            except Exception as e:
                print(f"WARN: failed to fetch {idx}/{did}: {e}")
                continue

            result = process_doc(hit)
            meta_f.write(json.dumps({"doc_id": result["doc_id"], "index": result["index"], "steps": len(result["steps"]), "edges": len(result["edges"])}) + "\n")
            for s in result["steps"]:
                steps_f.write(json.dumps({"doc_id": result["doc_id"], **s}) + "\n")
            for e in result["edges"]:
                edges_f.write(json.dumps({"doc_id": result["doc_id"], **e, "score": None, "accepted": False, "signals": None}) + "\n")

            _append_ledger(ledger, did)
            seen.add(did)
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} docs (skipped {skipped})...")
    finally:
        steps_f.close(); edges_f.close(); meta_f.close()
    print(f"Done. Processed {processed} docs (skipped {skipped}). Outputs in {out_path}")
    return processed


def main():
    ap = argparse.ArgumentParser(description="ID queue build/process")
    ap.add_argument("--index", type=str, default=None, help="Index or alias for queue build")
    ap.add_argument("--queue-file", type=str, default="outputs/ontology_queue/ids.ndjson", help="Queue file path")
    ap.add_argument("--batch", type=int, default=500, help="Batch size for queue build")
    ap.add_argument("--limit", type=int, default=0, help="Max ids to add (0 = all)")
    ap.add_argument("--out-dir", type=str, default="outputs/ontology_queue", help="Output dir for processing")
    ap.add_argument("--resume", action="store_true", help="Resume processing using ledger to skip already processed ids")
    ap.add_argument("--ledger", type=str, default=None, help="Custom ledger path (defaults to <out-dir>/processed.ids)")
    ap.add_argument("--mode", type=str, choices=["build", "process", "both"], default="both", help="Run mode")
    args = ap.parse_args()

    limit = None if args.limit == 0 else args.limit
    if args.mode in ("build", "both"):
        build_queue(args.index, args.queue_file, args.batch, limit)
    if args.mode in ("process", "both"):
        process_queue(args.queue_file, args.out_dir, args.resume, args.ledger)


if __name__ == "__main__":
    main()
