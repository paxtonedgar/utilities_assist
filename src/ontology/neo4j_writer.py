from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

from neo4j import GraphDatabase

from src.utils import load_config


def _load_neo4j_config() -> Dict[str, str]:
    cfg = load_config()
    # Case-insensitive section/key lookup
    sec = None
    for s in cfg.sections():
        if s.lower() == "neo4j":
            sec = s
            break
    data = {}
    if sec:
        for k, v in cfg[sec].items():
            data[k.lower()] = v
    # Allow env override
    import os
    data.setdefault("uri", os.getenv("NEO4J_URI", ""))
    data.setdefault("user", os.getenv("NEO4J_USER", ""))
    data.setdefault("password", os.getenv("NEO4J_PASSWORD", ""))
    data.setdefault("database", os.getenv("NEO4J_DATABASE", "neo4j"))
    return data


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _step_id(row: Dict[str, Any]) -> str:
    parts = [
        row.get("doc_id", ""),
        str(row.get("section_title", "")),
        str(row.get("order", "")),
        str(row.get("label", "")),
    ]
    return _sha1("|".join(parts))


def _read_ndjson(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    return rows


def _chunked(seq: List[Any], size: int = 1000):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _ensure_constraints(tx):
    tx.run("CREATE CONSTRAINT step_id IF NOT EXISTS ON (s:Step) ASSERT s.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT entity_key IF NOT EXISTS ON (e:Entity) ASSERT (e.type, e.name) IS UNIQUE")


def _merge_steps(session, steps: List[Dict[str, Any]]):
    rows = []
    for s in steps:
        sid = _step_id(s)
        props = {
            "id": sid,
            "label": s.get("label"),
            "verb": s.get("verb"),
            "obj": s.get("obj"),
            "doc_id": s.get("doc_id"),
            "section": s.get("section_title"),
            "page_url": s.get("page_url"),
            "order": s.get("order"),
        }
        rows.append({"id": sid, "props": props})

    if not rows:
        return

    query = (
        "UNWIND $rows AS row "
        "MERGE (s:Step {id: row.id}) "
        "SET s += row.props"
    )

    for chunk in _chunked(rows):
        session.execute_write(lambda tx: tx.run(query, rows=chunk))


def _merge_next_edges(session, edges: List[Dict[str, Any]]):
    rows = []
    for e in edges:
        a, b = e.get("a", {}), e.get("b", {})
        if not a or not b:
            continue
        src = _step_id(a)
        dst = _step_id(b)
        rows.append(
            {
                "src": src,
                "dst": dst,
                "score": e.get("score"),
                "accepted": False,  # default to false per YAGNI spec
            }
        )
    if not rows:
        return
    query = (
        "UNWIND $rows AS row "
        "MERGE (a:Step {id: row.src}) "
        "MERGE (b:Step {id: row.dst}) "
        "MERGE (a)-[r:NEXT]->(b) "
        "SET r.score = row.score, r.accepted = row.accepted"
    )
    for chunk in _chunked(rows):
        session.execute_write(lambda tx: tx.run(query, rows=chunk))


def _merge_entities(session, entities: List[Dict[str, Any]]):
    rows = []
    for e in entities:
        t = e.get("type")
        n = e.get("name")
        if t and n:
            rows.append({"type": t, "name": n})
    if not rows:
        return
    query = "UNWIND $rows AS row MERGE (e:Entity {type: row.type, name: row.name})"
    for chunk in _chunked(rows):
        session.execute_write(lambda tx: tx.run(query, rows=chunk))


def push_to_neo4j(out_dirs: List[Path], database: str):
    cfg = _load_neo4j_config()
    uri = cfg.get("uri")
    user = cfg.get("user")
    pwd = cfg.get("password")
    db = cfg.get("database", database or "neo4j")
    if not (uri and user and pwd):
        raise RuntimeError("Missing Neo4j config: set [NEO4J] uri,user,password in config.ini or NEO4J_* env vars")

    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    with driver.session(database=db) as session:
        session.execute_write(_ensure_constraints)

        for base in out_dirs:
            steps_path = base / "steps.ndjson"
            edges_path = base / "edges.ndjson"
            entities_path = base / "entities.ndjson"
            steps = _read_ndjson(steps_path)
            edges = _read_ndjson(edges_path)
            entities = _read_ndjson(entities_path)
            if not steps and not edges:
                print(f"No data found in {base}")
                continue
            print(f"Pushing from {base}: steps={len(steps)} edges={len(edges)} entities={len(entities)}")
            _merge_steps(session, steps)
            next_edges = [e for e in edges if e.get("type") == "NEXT"]
            _merge_next_edges(session, next_edges)
            _merge_entities(session, entities)


def main():
    ap = argparse.ArgumentParser(description="Push ontology outputs to Neo4j")
    ap.add_argument("--inputs", type=str, nargs="+", required=True, help="One or more output directories containing steps.ndjson/edges.ndjson")
    ap.add_argument("--database", type=str, default="neo4j", help="Neo4j database name")
    args = ap.parse_args()

    dirs = [Path(p) for p in args.inputs]
    push_to_neo4j(dirs, database=args.database)


if __name__ == "__main__":
    main()
