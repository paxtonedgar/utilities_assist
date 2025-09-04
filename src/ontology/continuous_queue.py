"""
Continuous scan runner: repeatedly builds an ID queue for each index,
filters out already-processed IDs (from a ledger), processes remaining IDs,
then repeats until no unprocessed IDs remain.

Defaults to scanning the main and swagger indices.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Set

from src.infra.settings import get_settings
from src.infra.search_config import OpenSearchConfig
from src.ontology.id_queue import build_queue, process_queue


def _load_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def _filter_queue(queue_path: Path, ledger_ids: Set[str], out_path: Path) -> int:
    """Write only unprocessed ids to out_path; return count written."""
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with queue_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for ln in src:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                did = obj.get("_id")
                if did and did not in ledger_ids:
                    dst.write(json.dumps(obj) + "\n")
                    count += 1
            except Exception:
                # plain id line
                if ln not in ledger_ids:
                    dst.write(ln + "\n")
                    count += 1
    return count


def run(indices: List[str], out_dir: str, batch: int):
    for index in indices:
        print(f"=== Scanning index: {index} ===")
        base = Path(out_dir) / index
        base.mkdir(parents=True, exist_ok=True)
        queue_path = base / "ids.ndjson"
        filtered_queue = base / "ids.unprocessed.ndjson"
        ledger = base / "processed.ids"

        while True:
            # 1) Build queue of all ids
            build_queue(index=index, out_file=str(queue_path), batch=batch, limit=None)

            # 2) Filter out processed ids
            ledger_ids = _load_ids(ledger)
            remaining = _filter_queue(queue_path, ledger_ids, filtered_queue)
            print(f"Remaining to process in {index}: {remaining}")
            if remaining == 0:
                print(f"Index {index}: complete.")
                break

            # 3) Process filtered queue (no resume; queue already filtered)
            processed = process_queue(queue_file=str(filtered_queue), out_dir=str(base), resume=False, ledger_file=str(ledger))
            if processed == 0:
                print(f"Index {index}: no progress; stopping to avoid loop.")
                break


def main():
    ap = argparse.ArgumentParser(description="Continuous queue scan across indices")
    ap.add_argument("--indices", type=str, default="", help="Comma-separated list of indices; empty=main+swagger")
    ap.add_argument("--out-dir", type=str, default="outputs/continuous", help="Base output directory")
    ap.add_argument("--batch", type=int, default=500, help="Batch size for queue build")
    args = ap.parse_args()

    if args.indices:
        idxs = [s.strip() for s in args.indices.split(",") if s.strip()]
    else:
        s = get_settings()
        idxs = [s.search_index_alias, OpenSearchConfig.get_swagger_index()]

    run(indices=idxs, out_dir=args.out_dir, batch=args.batch)


if __name__ == "__main__":
    main()

