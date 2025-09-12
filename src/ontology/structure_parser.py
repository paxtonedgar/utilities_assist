from __future__ import annotations

"""
Structure-preserving parsing for Stage 1B.

Fetches documents from OpenSearch and emits semantic segments (tables, lists,
code/JSON, header sections) with anchors and simple type classifications.
"""

from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _get_sections_source(hit: Dict[str, Any]) -> List[Dict[str, Any]]:
    src = hit.get("_source", {})
    sections = src.get("sections")
    if isinstance(sections, list):
        return [s for s in sections if isinstance(s, dict)]
    return []


def _parse_html(html: str) -> Dict[str, Any]:
    """Parse HTML to extract headers, lists, tables, and pre/code blocks."""
    try:
        from lxml import html as lhtml  # type: ignore
    except Exception:
        return {"headers": [], "lists": [], "tables": [], "codes": []}

    try:
        root = lhtml.fromstring(html)
    except Exception:
        return {"headers": [], "lists": [], "tables": [], "codes": []}

    headers = []
    for i in range(1, 7):
        for el in root.findall(f".//h{i}"):
            headers.append({"level": i, "text": (el.text_content() or "").strip()})

    lists = []
    for tag in ("ul", "ol"):
        for el in root.findall(f".//{tag}"):
            items = [(li.text_content() or "").strip() for li in el.findall(".//li")]
            if items:
                lists.append({"ordered": tag == "ol", "items": items[:50]})

    tables = []
    for el in root.findall(".//table"):
        # Extract header row and first N rows
        headers_row = []
        head = el.find(".//thead")
        if head is not None:
            ths = head.findall(".//th")
            headers_row = [th.text_content().strip() for th in ths]
        if not headers_row:
            # try first row of tbody
            first_tr = el.find(".//tr")
            if first_tr is not None:
                headers_row = [td.text_content().strip() for td in first_tr.findall(".//th|.//td")]
        rows = []
        for r in el.findall(".//tr")[1:6]:
            rows.append([td.text_content().strip() for td in r.findall(".//th|.//td")])
        tables.append({"headers": headers_row[:20], "rows": rows})

    codes = []
    for tag in ("pre", "code"):
        for el in root.findall(f".//{tag}"):
            txt = (el.text_content() or "").strip()
            if txt:
                codes.append({"lang": None, "sample": txt[:1000]})

    return {"headers": headers, "lists": lists, "tables": tables, "codes": codes}


def classify_table(headers: List[str]) -> str:
    hs = [h.lower() for h in headers]
    if any(x in " ".join(hs) for x in ["owner", "team", "contact", "group", "steward"]):
        return "ownership_matrix"
    if any(x in " ".join(hs) for x in ["parameter", "field", "type", "default", "required"]):
        return "spec_table"
    if any(x in " ".join(hs) for x in ["step", "action", "result", "precondition", "postcondition"]):
        return "flow_table"
    return "other"


def parse_hit_to_segments(hit: Dict[str, Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for s in _get_sections_source(hit):
        html = s.get("content") or s.get("text") or s.get("body") or ""
        if not isinstance(html, str) or len(html) < 20:
            continue
        parsed = _parse_html(html)
        # Lists as StepBlocks
        for lst in parsed["lists"]:
            segments.append({
                "type": "StepBlock" if lst.get("ordered") else "ListBlock",
                "anchors": {"section_title": s.get("title") or s.get("heading")},
                "features": {"items": lst.get("items", [])[:10]},
                "confidence": 0.6 if lst.get("ordered") else 0.4,
            })
        # Tables
        for tbl in parsed["tables"]:
            ttype = classify_table(tbl.get("headers") or [])
            segments.append({
                "type": "Table",
                "table_type": ttype,
                "anchors": {"section_title": s.get("title") or s.get("heading")},
                "features": {"headers": (tbl.get("headers") or [])[:20]},
                "confidence": 0.6,
            })
        # Code/JSON (simple capture)
        for code in parsed["codes"]:
            segments.append({
                "type": "CodeBlock",
                "anchors": {"section_title": s.get("title") or s.get("heading")},
                "features": {"lang": code.get("lang"), "sample": code.get("sample")},
                "confidence": 0.5,
            })
    return segments

