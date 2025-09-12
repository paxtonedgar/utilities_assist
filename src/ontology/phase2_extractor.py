from __future__ import annotations

"""
Phase 2: Entity extraction from Phase 1 segments.

Reads a semantic_map/<index>/doc_map.jsonl and emits entities.jsonl with
normalized entities derived from segment types:

  - Table(spec_table): Parameters (name, type, required, location, description)
  - Table(ownership_matrix): Team/Person ownership over a system/process
  - Table(flow_table): Step entities with stage/order (best-effort)
  - StepBlock: Action/Tool mentions (lightweight regex-based extraction)

All entities include provenance (doc_id, anchors, segment_type) and a
confidence score.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path
import argparse
import json
import re


@dataclass
class Entity:
    entity_id: str
    type: str
    name: str
    aliases: List[str]
    source_doc: str
    source_segment: Dict[str, Any]
    attrs: Dict[str, Any]
    confidence: float


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except Exception:
                continue
    return rows


def _normalize_headers(headers: List[str]) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    canon = {
        "name": {"name", "field name", "parameter", "param", "field"},
        "type": {"type", "data type", "datatype"},
        "required": {"required", "mandatory", "req?", "required?"},
        "location": {"location", "in", "place"},
        "description": {"description", "desc", "details", "notes"},
        # ownership
        "team": {"team", "squad", "group", "dept", "department", "organization", "org"},
        "owner": {"owner", "steward", "maintainer", "responsible"},
        "contact": {"contact", "manager", "lead", "poc"},
        # flow
        "step": {"step", "stage", "phase", "task", "activity"},
        "action": {"action", "what", "do", "operation"},
        "result": {"result", "output", "outcome"},
    }
    for i, h in enumerate(headers or []):
        h_low = str(h).strip().lower()
        for key, variants in canon.items():
            if any(v in h_low for v in variants):
                if key not in idx:  # first match wins
                    idx[key] = i
    return idx


def _bool_from_cell(cell: str) -> Optional[bool]:
    s = str(cell).strip().lower()
    if s in {"y", "yes", "true", "1"}:
        return True
    if s in {"n", "no", "false", "0"}:
        return False
    return None


def _mk_entity_id(doc_id: str, kind: str, name: str, extra: str = "") -> str:
    base = f"{doc_id}:{kind}:{name}:{extra}".strip(":")
    return base[:256]


def _extract_spec_table(doc_id: str, section_title: str | None, headers: List[str], rows: List[List[str]]) -> List[Entity]:
    ents: List[Entity] = []
    idx = _normalize_headers(headers)
    for r in rows or []:
        name = r[idx.get("name", -1)] if idx.get("name") is not None and idx.get("name") < len(r) else None
        if not name:
            continue
        dtype = r[idx.get("type", -1)] if idx.get("type") is not None and idx.get("type") < len(r) else None
        required_cell = r[idx.get("required", -1)] if idx.get("required") is not None and idx.get("required") < len(r) else None
        location = r[idx.get("location", -1)] if idx.get("location") is not None and idx.get("location") < len(r) else None
        description = r[idx.get("description", -1)] if idx.get("description") is not None and idx.get("description") < len(r) else None
        required = _bool_from_cell(required_cell) if required_cell is not None else None
        eid = _mk_entity_id(doc_id, "Parameter", str(name), str(section_title or ""))
        ents.append(
            Entity(
                entity_id=eid,
                type="Parameter",
                name=str(name),
                aliases=[],
                source_doc=doc_id,
                source_segment={"type": "Table", "table_type": "spec_table", "section_title": section_title},
                attrs={"data_type": dtype, "required": required, "location": location, "description": description},
                confidence=0.9,
            )
        )
    return ents


def _extract_ownership_table(doc_id: str, section_title: str | None, headers: List[str], rows: List[List[str]]) -> List[Entity]:
    ents: List[Entity] = []
    idx = _normalize_headers(headers)
    # Determine owner/team/contact columns
    owner_keys = [k for k in ("owner", "team", "contact") if k in idx]
    for r in rows or []:
        for key in owner_keys:
            pos = idx.get(key)
            if pos is None or pos >= len(r):
                continue
            val = str(r[pos]).strip()
            if not val:
                continue
            etype = "Team" if key == "team" else ("Person" if key == "contact" else "Team")
            eid = _mk_entity_id(doc_id, etype, val, str(section_title or ""))
            ents.append(
                Entity(
                    entity_id=eid,
                    type=etype,
                    name=val,
                    aliases=[],
                    source_doc=doc_id,
                    source_segment={"type": "Table", "table_type": "ownership_matrix", "section_title": section_title},
                    attrs={"role": key, "owns": True},
                    confidence=0.8,
                )
            )
    return ents


def _extract_flow_table(doc_id: str, section_title: str | None, headers: List[str], rows: List[List[str]]) -> List[Entity]:
    ents: List[Entity] = []
    idx = _normalize_headers(headers)
    for order, r in enumerate(rows or [], start=1):
        label = r[idx.get("step", -1)] if idx.get("step") is not None and idx.get("step") < len(r) else None
        action = r[idx.get("action", -1)] if idx.get("action") is not None and idx.get("action") < len(r) else None
        result = r[idx.get("result", -1)] if idx.get("result") is not None and idx.get("result") < len(r) else None
        name = label or action
        if not name:
            continue
        eid = _mk_entity_id(doc_id, "Step", str(name), str(order))
        ents.append(
            Entity(
                entity_id=eid,
                type="Step",
                name=str(name),
                aliases=[],
                source_doc=doc_id,
                source_segment={"type": "Table", "table_type": "flow_table", "section_title": section_title},
                attrs={"order": order, "action": action, "result": result},
                confidence=0.75,
            )
        )
    return ents


_TOOL_RE = re.compile(r"`([^`]+)`|\b([A-Z][A-Za-z0-9_\-]{2,})\b")


def _extract_stepblock(doc_id: str, section_title: str | None, items: List[str]) -> List[Entity]:
    ents: List[Entity] = []
    tool_hits: Dict[str, int] = {}
    for it in items or []:
        for m in _TOOL_RE.findall(it):
            cand = next((x for x in m if x), None)
            if not cand:
                continue
            s = cand.strip()
            if len(s) < 3:
                continue
            tool_hits[s] = tool_hits.get(s, 0) + 1
    for name, cnt in tool_hits.items():
        eid = _mk_entity_id(doc_id, "Tool", name, str(section_title or ""))
        ents.append(
            Entity(
                entity_id=eid,
                type="Tool",
                name=name,
                aliases=[],
                source_doc=doc_id,
                source_segment={"type": "StepBlock", "section_title": section_title},
                attrs={"mentions": cnt},
                confidence=0.6,
            )
        )
    return ents


def extract_entities(sem_dir: Path, out_path: Path) -> int:
    doc_map = _read_jsonl(sem_dir / "doc_map.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as fw:
        for d in doc_map:
            doc_id = d.get("doc_id")
            segments = d.get("segments") or []
            for s in segments:
                stype = s.get("type")
                if stype == "Table":
                    headers = (s.get("features") or {}).get("headers") or []
                    rows = (s.get("features") or {}).get("rows") or []  # rows may not be present in Phase 1; best-effort
                    ttype = s.get("table_type")
                    sec = (s.get("anchors") or {}).get("section_title")
                    if ttype == "spec_table":
                        ents = _extract_spec_table(doc_id, sec, headers, rows)
                    elif ttype == "ownership_matrix":
                        ents = _extract_ownership_table(doc_id, sec, headers, rows)
                    elif ttype == "flow_table":
                        ents = _extract_flow_table(doc_id, sec, headers, rows)
                    else:
                        ents = []
                elif stype == "StepBlock":
                    items = (s.get("features") or {}).get("items") or []
                    sec = (s.get("anchors") or {}).get("section_title")
                    ents = _extract_stepblock(doc_id, sec, items)
                else:
                    ents = []

                for e in ents:
                    fw.write(json.dumps(asdict(e)) + "\n")
                    n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description="Phase 2 entity extractor from segments")
    ap.add_argument("--semantic-dir", type=str, required=True, help="Semantic map directory")
    ap.add_argument("--out", type=str, default=None, help="Output file path (entities.jsonl by default)")
    args = ap.parse_args()
    sem_dir = Path(args.semantic_dir)
    out = Path(args.out) if args.out else sem_dir / "entities.jsonl"
    n = extract_entities(sem_dir, out)
    print(f"Wrote {n} entities -> {out}")


if __name__ == "__main__":
    main()

