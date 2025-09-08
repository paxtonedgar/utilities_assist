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

from src.infra.opensearch_client import OpenSearchClient
from src.infra.settings import get_settings
from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy
import requests
from src.ontology.doc_by_doc import process_doc


def _resolve_alias_or_index(name: str) -> list[str]:
    """Return list of concrete indices for an alias or a single concrete index.

    Does not modify infra client; uses same auth/proxy utilities.
    """
    s = get_settings()
    base = s.opensearch_host.rstrip("/")
    _setup_jpmc_proxy()
    auth = _get_aws_auth()
    try:
        r = requests.get(f"{base}/_alias/{name}", auth=auth, timeout=30)
        if r.ok and isinstance(r.json(), dict):
            keys = list(r.json().keys())
            return keys if keys else [name]
    except Exception:
        pass
    return [name]


def build_queue(index: str | None, out_file: str, batch: int, limit: int | None) -> int:
    client = OpenSearchClient()
    idx = index or client.settings.search_index_alias
    path = Path(out_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        # Resolve alias to concrete indices to avoid {alias}/_search 404s on managed domains
        targets = _resolve_alias_or_index(idx)
        for target in targets:
            # Prefer scroll API for robust full iteration (commonly allowed on managed clusters)
            for _id, _index in _iter_ids_via_scroll(target, batch, limit):
                f.write(json.dumps({"_id": _id, "_index": _index}) + "\n")
                n += 1
    print(f"Queue built: {n} ids -> {path}")
    return n


def _iter_ids_via_scroll(index: str, batch: int, limit: int | None):
    """Yield (_id, _index) using the OpenSearch Scroll API.

    Uses the same SigV4/proxy plumbing as the app.
    """
    s = get_settings()
    base = s.opensearch_host.rstrip("/")
    _setup_jpmc_proxy()
    auth = _get_aws_auth()

    # 1) initial search to obtain scroll_id (prefer scroll)
    params = {"scroll": "2m"}
    body = {"size": batch, "sort": ["_doc"], "query": {"match_all": {}}}
    url = f"{base}/{index}/_search"
    try:
        r = requests.post(url, params=params, json=body, auth=auth, timeout=60)
        r.raise_for_status()
    except requests.HTTPError as e:
        # On clusters where scroll is blocked (404 on alias/_search?scroll), fallback to from/size pagination
        status = getattr(e.response, "status_code", None)
        if status == 404:
            yield from _iter_ids_via_fromsize(base, auth, index, batch, limit)
            return
        raise
    data = r.json()
    scroll_id = data.get("_scroll_id")
    hits = data.get("hits", {}).get("hits", [])
    yielded = 0

    def _yield_hits(harr):
        nonlocal yielded
        for h in harr or []:
            _id = h.get("_id")
            _index = h.get("_index", index)
            if _id:
                yield (_id, _index)
                yielded += 1
                if limit and yielded >= limit:
                    return True
        return False

    # first page
    stop = _yield_hits(hits)
    if stop:
        return

    # 2) scroll until empty or limit reached
    scroll_url = f"{base}/_search/scroll"
    while scroll_id:
        sr = requests.post(scroll_url, json={"scroll": "2m", "scroll_id": scroll_id}, auth=auth, timeout=60)
        sr.raise_for_status()
        sdata = sr.json()
        scroll_id = sdata.get("_scroll_id")
        shits = sdata.get("hits", {}).get("hits", [])
        if not shits:
            break
        if _yield_hits(shits):
            break


def _iter_ids_via_fromsize(base: str, auth, index: str, batch: int, limit: int | None):
    """Yield (_id, _index) using from/size pagination with a simple match_all query.

    This mimics a "normal" search request body similar to what the app uses, avoiding
    scroll and special sort bodies that gateways may reject on aliases.
    """
    offset = 0
    yielded = 0
    url = f"{base}/{index}/_search"
    while True:
        body = {"from": offset, "size": batch, "query": {"match_all": {}}}
        r = requests.post(url, json=body, auth=auth, timeout=60)
        r.raise_for_status()
        data = r.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            break
        for h in hits:
            _id = h.get("_id")
            _index = h.get("_index", index)
            if _id:
                yield (_id, _index)
                yielded += 1
                if limit and yielded >= limit:
                    return
        offset += batch


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
    client = OpenSearchClient()
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
            idx = item.get("_index") or client.settings.search_index_alias
            try:
                hit = client.get_doc_by_id(index=idx, doc_id=did)
            except Exception as e:
                print(f"WARN: failed to fetch {idx}/{did}: {e}")
                continue

            result = process_doc(hit)
            meta_f.write(json.dumps({"doc_id": result["doc_id"], "index": result["index"], "steps": len(result["steps"]), "edges": len(result["edges"])}) + "\n")
            for s in result["steps"]:
                steps_f.write(json.dumps({"doc_id": result["doc_id"], **s}) + "\n")
            for e, sc in zip(result["edges"], result["scored"]):
                edges_f.write(json.dumps({"doc_id": result["doc_id"], **e, "score": sc.get("score"), "accepted": sc.get("accepted"), "signals": sc.get("signals")}) + "\n")

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
