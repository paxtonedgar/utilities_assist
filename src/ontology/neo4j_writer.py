from __future__ import annotations
import argparse
import json
import hashlib
import configparser
from pathlib import Path
from typing import Dict, Any, List, Tuple

from neo4j import GraphDatabase
from src.infra.settings import _load_shared_config


def _load_neo4j_config() -> Dict[str, str]:
    """Load [neo4j] credentials using the shared resource-config loader."""
    cfg = _load_shared_config()  # Centralized, cached config.ini loader
    data: Dict[str, str] = {}
    # Case-insensitive [neo4j] section
    sec_name = None
    for s in cfg.sections():
        if s.lower() == "neo4j":
            sec_name = s
            break
    if sec_name:
        for k, v in cfg[sec_name].items():
            data[k.lower()] = v
    # Accept both url and uri (be forgiving with config keys)
    if "uri" not in data and "url" in data:
        data["uri"] = data.get("url", "")
    # Allow env override
    import os
    if not data.get("uri"):
        data["uri"] = os.getenv("NEO4J_URI", "")
    if not data.get("user"):
        data["user"] = os.getenv("NEO4J_USER", "")
    if not data.get("password"):
        data["password"] = os.getenv("NEO4J_PASSWORD", "")
    data.setdefault("database", os.getenv("NEO4J_DATABASE", "neo4j"))
    return data


def _composite_doc_id(index_name: str | None, doc_id: str | None) -> str:
    idx = (index_name or "").strip()
    did = (doc_id or "").strip()
    return f"{idx}/{did}" if idx else did


def _step_id(row: Dict[str, Any]) -> str:
    """Composite Step id: <index>/<doc_id>_<order>.

    Falls back to legacy hashed id if required fields are missing.
    """
    idx = row.get("index")
    did = row.get("doc_id")
    order = row.get("order")
    if did is not None and order is not None:
        return f"{_composite_doc_id(idx, did)}_{order}"
    # Fallback: legacy stable hash
    parts = [
        str(row.get("doc_id", "")),
        str(row.get("section_title", "")),
        str(row.get("order", "")),
        str(row.get("label", "")),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()


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
    tx.run("CREATE CONSTRAINT doc_id IF NOT EXISTS ON (d:Doc) ASSERT d.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT entity_key IF NOT EXISTS ON (e:Entity) ASSERT (e.type, e.name) IS UNIQUE")


def _merge_steps(session, steps: List[Dict[str, Any]]):
    rows = []
    for s in steps:
        sid = _step_id(s)
        comp_doc = _composite_doc_id(s.get("index"), s.get("doc_id"))
        props = {
            "id": sid,
            "label": s.get("label") or s.get("canonical_label"),
            "verb": s.get("verb"),
            "obj": s.get("obj") or s.get("object"),
            "doc_id": s.get("doc_id"),
            "section": s.get("section_title") or s.get("section"),
            "page_url": s.get("page_url"),
            "order": s.get("order"),
        }
        rows.append({"id": sid, "props": props, "doc_comp": comp_doc})

    if not rows:
        return

    query = (
        "UNWIND $rows AS row "
        "MERGE (s:Step {id: row.id}) "
        "SET s += row.props "
        "WITH s, row "
        "MERGE (d:Doc {id: row.doc_comp}) "
        "MERGE (s)-[:OF_DOC]->(d)"
    )

    for chunk in _chunked(rows):
        session.execute_write(lambda tx: tx.run(query, rows=chunk))


def _merge_next_edges(session, edges: List[Dict[str, Any]]):
    rows = []
    for e in edges:
        a, b = e.get("a", {}), e.get("b", {})
        if a and b:
            src = _step_id(a)
            dst = _step_id(b)
            accepted = bool(e.get("accepted", False))
        else:
            # Fallback: top-level fields (doc_id, index, src_order, dst_order)
            idx = e.get("index")
            did = e.get("doc_id")
            src_order = e.get("src_order") or e.get("order_a")
            dst_order = e.get("dst_order") or e.get("order_b")
            comp = _composite_doc_id(idx, did)
            if comp and src_order is not None and dst_order is not None:
                src = f"{comp}_{src_order}"
                dst = f"{comp}_{dst_order}"
                accepted = bool(e.get("accepted", False))
            else:
                continue
        rows.append(
            {
                "src": src,
                "dst": dst,
                "score": e.get("score"),
                "accepted": accepted,
            }
        )
    if not rows:
        return
    query = (
        "UNWIND $rows AS row "
        "MERGE (a:Step {id: row.src}) "
        "MERGE (b:Step {id: row.dst}) "
        "MERGE (a)-[r:NEXT]->(b) "
        "SET r.score = row.score, r.accepted = coalesce(r.accepted, row.accepted)"
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
            docs_path = base / "docs.ndjson"
            steps = _read_ndjson(steps_path)
            edges = _read_ndjson(edges_path)
            entities = _read_ndjson(entities_path)
            docs = _read_ndjson(docs_path)
            if not steps and not edges:
                print(f"No data found in {base}")
                continue
            print(f"Pushing from {base}: docs={len(list(docs)) if docs_path.exists() else 0} steps={len(steps)} edges={len(edges)} entities={len(entities)}")
            # Reset docs iterable (we consumed it for len) and push docs if present
            docs = _read_ndjson(docs_path)
            if docs_path.exists():
                drows = []
                for d in docs:
                    comp = _composite_doc_id(d.get("index"), d.get("doc_id"))
                    drows.append({
                        "id": comp,
                        "index_name": d.get("index"),
                        "step_cnt": d.get("steps"),
                        "edge_cnt": d.get("edges"),
                    })
                if drows:
                    dquery = (
                        "UNWIND $rows AS row "
                        "MERGE (d:Doc {id: row.id}) "
                        "SET d.index_name = row.index_name, d.step_cnt = row.step_cnt, d.edge_cnt = row.edge_cnt"
                    )
                    for chunk in _chunked(drows):
                        session.execute_write(lambda tx: tx.run(dquery, rows=chunk))
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
