from __future__ import annotations

"""
Structure-preserving parsing for Stage 1B.

Fetches documents from OpenSearch and emits semantic segments (tables, lists,
code/JSON, header sections) with anchors and simple type classifications.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def _get_sections_source(hit: Dict[str, Any]) -> List[Dict[str, Any]]:
    src = hit.get("_source", {})
    sections = src.get("sections")
    if isinstance(sections, list):
        return [s for s in sections if isinstance(s, dict)]
    return []


def debug_first_doc(hit: Dict[str, Any]) -> None:
    """Print a quick structural debug of the first document hit.

    Shows top-level keys, section[0] keys, and probes common content fields
    printing length and first 500 chars. Also flags HTML/markdown table cues.
    """
    try:
        print("=== DOCUMENT STRUCTURE DEBUG ===")
        source = hit.get("_source", {})
        print(f"Top fields: {list(source.keys())}")
        sections = source.get("sections", [])
        if isinstance(sections, list) and len(sections) > 0 and isinstance(sections[0], dict):
            print(f"Section[0] keys: {list(sections[0].keys())}")
            for field in ["content", "body", "html", "text", "raw"]:
                if field in sections[0]:
                    content = sections[0][field]
                    cstr = str(content)
                    print(f"Field '{field}': exists, Length: {len(cstr)}")
                    preview = cstr[:500].replace("\n", " ")
                    print(f"First 500 chars: {preview}")
                    lower = cstr.lower()
                    if "<table" in lower:
                        print("  ✓ Contains <table> tags!")
                    if ("|" in cstr) and ("---" in cstr or ":-" in cstr):
                        print("  ✓ Contains markdown table markers!")
        else:
            print("No sections[] or unexpected sections format")
    except Exception as e:
        print(f"DEBUG ERROR: {e}")


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

    # Links
    links = []
    try:
        for a in root.findall('.//a'):
            href = a.get('href') or ''
            if href:
                links.append(href)
    except Exception:
        pass
    return {"headers": headers, "lists": lists, "tables": tables, "codes": codes, "links": links}


OWNERSHIP_SYNS = {"owner", "team", "contact", "steward", "poc", "pm", "tl", "lead", "manager", "responsible", "maintainer", "squad", "group", "dept", "department", "organization"}
TECH_SYNS = {"parameter", "field", "type", "default", "required", "format", "example", "schema", "property", "attribute", "value", "datatype", "constraint"}
PROC_SYNS = {"step", "action", "result", "pre", "post", "condition", "phase", "stage", "task", "activity", "input", "output", "trigger", "outcome"}


def classify_table(headers: List[str]) -> Tuple[str, float]:
    hs = [h.lower() for h in headers]
    joined = " ".join(hs)
    if any(x in joined for x in OWNERSHIP_SYNS):
        return "ownership_matrix", 0.7
    if any(x in joined for x in TECH_SYNS):
        return "spec_table", 0.6
    if any(x in joined for x in PROC_SYNS):
        return "flow_table", 0.6
    return "other", 0.3

def _is_html_like(text: str) -> bool:
    return "<" in text and ">" in text and "</" in text

def _extract_plain_text_segments(text: str, section_title: str | None) -> List[Dict[str, Any]]:
    segs: List[Dict[str, Any]] = []
    import re
    bullets = re.findall(r"(?m)^(?:\s*[-•*]|\s*\d+[.)])\s+(.+)$", text)
    if bullets:
        segs.append({
            "type": "StepBlock",
            "anchors": {"section_title": section_title},
            "features": {"items": bullets[:10]},
            "confidence": 0.5,
            "segment_confidence": {"extraction_confidence": 0.6, "classification_confidence": 0.6, "value_confidence": 0.6},
        })
    imps = re.findall(r"(?mi)^(?:to\s+[a-z]+|step\s*\d+)[^\n]{10,}$", text)
    if imps and not bullets:
        segs.append({
            "type": "StepBlock",
            "anchors": {"section_title": section_title},
            "features": {"items": imps[:10]},
            "confidence": 0.4,
            "segment_confidence": {"extraction_confidence": 0.5, "classification_confidence": 0.5, "value_confidence": 0.5},
        })
    # Markdown tables: header line + separator line ('---') then rows
    tables = _parse_markdown_tables(text)
    for t in tables:
        ttype, base_conf = classify_table(t.get("headers") or [])
        segs.append({
            "type": "Table",
            "table_type": ttype,
            "anchors": {"section_title": section_title},
            "features": {"headers": (t.get("headers") or [])[:20]},
            "confidence": base_conf,
            "segment_confidence": {"extraction_confidence": 0.6, "classification_confidence": base_conf, "value_confidence": 0.6},
        })
    return segs

_table_llm_cache: Dict[str, Tuple[str, float]] = {}

def classify_table_with_llm(headers: List[str], resources) -> Tuple[str, float]:
    import json
    sig = "|".join([h.strip().lower() for h in headers[:10]])
    if sig in _table_llm_cache:
        return _table_llm_cache[sig]
    label, conf = "other", 0.4
    try:
        client = getattr(resources, "chat_client", None)
        if client is None:
            return label, conf
        system = "You classify tables by headers. Return JSON {type, confidence}. Types: ownership_matrix, spec_table, flow_table, other."
        prompt = {"headers": headers[:10]}
        resp = client.chat.completions.create(
            model=resources.settings.chat.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json.dumps(prompt)}],
            temperature=0.0,
        )
        content = resp.choices[0].message.content  # type: ignore
        data = json.loads(content or "{}")
        label = str(data.get("type", "other"))
        conf = float(data.get("confidence", 0.5))
    except Exception:
        pass
    _table_llm_cache[sig] = (label, conf)
    return label, conf


def parse_hit_to_segments(hit: Dict[str, Any], resources=None, index_profile: str | None = None) -> Tuple[List[Dict[str, Any]], List[str]]:
    segments: List[Dict[str, Any]] = []
    links_acc: List[str] = []
    for s in _get_sections_source(hit):
        content = s.get("content") or s.get("text") or s.get("body") or ""
        if not isinstance(content, str) or len(content) < 20:
            continue
        if _is_html_like(content):
            parsed = _parse_html(content)
            links_acc.extend(parsed.get("links", []))
            for lst in parsed["lists"]:
                segments.append({
                    "type": "StepBlock" if lst.get("ordered") else "ListBlock",
                    "anchors": {"section_title": s.get("title") or s.get("heading")},
                    "features": {"items": lst.get("items", [])[:10]},
                    "confidence": 0.6 if lst.get("ordered") else 0.4,
                    "segment_confidence": {"extraction_confidence": 0.7, "classification_confidence": 0.7 if lst.get("ordered") else 0.5, "value_confidence": 0.6},
                })
            for tbl in parsed["tables"]:
                ttype, base_conf = classify_table(tbl.get("headers") or [])
                if ttype == "other" and resources is not None:
                    ttype, llm_conf = classify_table_with_llm(tbl.get("headers") or [], resources)
                    base_conf = max(base_conf, llm_conf)
                segments.append({
                    "type": "Table",
                    "table_type": ttype,
                    "anchors": {"section_title": s.get("title") or s.get("heading")},
                    "features": {"headers": (tbl.get("headers") or [])[:20]},
                    "confidence": base_conf,
                    "segment_confidence": {"extraction_confidence": 0.7, "classification_confidence": base_conf, "value_confidence": 0.7 if ttype != 'other' else 0.4},
                })
            for code in parsed["codes"]:
                segments.append({
                    "type": "CodeBlock",
                    "anchors": {"section_title": s.get("title") or s.get("heading")},
                    "features": {"lang": code.get("lang"), "sample": code.get("sample")},
                    "confidence": 0.5,
                    "segment_confidence": {"extraction_confidence": 0.5, "classification_confidence": 0.4, "value_confidence": 0.5},
                })
        else:
            segments.extend(_extract_plain_text_segments(content, s.get("title") or s.get("heading")))
    try:
        segments.extend(_extract_structured_fields(hit, index_profile=index_profile))
    except Exception:
        pass
    return segments, links_acc

def _extract_structured_fields(hit: Dict[str, Any], index_profile: str | None) -> List[Dict[str, Any]]:
    segs: List[Dict[str, Any]] = []
    src = hit.get("_source", {})
    prof = index_profile or "product"
    if prof == "swagger":
        server_url = None
        for s in (src.get("sections") or []):
            if isinstance(s, dict) and s.get("server_url"):
                server_url = s.get("server_url"); break
        if server_url:
            segs.append({
                "type": "EndpointBlock",
                "anchors": {},
                "features": {"server_url": server_url},
                "confidence": 0.7,
                "segment_confidence": {"extraction_confidence": 0.7, "classification_confidence": 0.7, "value_confidence": 0.8},
            })
        for s in (src.get("sections") or []):
            if not isinstance(s, dict):
                continue
            info = s.get("info")
            if isinstance(info, dict):
                feat = {k: info.get(k) for k in ("description", "version") if info.get(k)}
                contact = info.get("contact")
                if isinstance(contact, dict):
                    feat["contact"] = {k: contact.get(k) for k in ("name", "email") if contact.get(k)}
                if feat:
                    segs.append({
                        "type": "ApiInfo",
                        "anchors": {},
                        "features": feat,
                        "confidence": 0.7,
                        "segment_confidence": {"extraction_confidence": 0.7, "classification_confidence": 0.7, "value_confidence": 0.8},
                    })
    elif prof == "events":
        import json as _json
        meta = src.get("metadata") if isinstance(src.get("metadata"), dict) else {}
        extraction = meta.get("extractionRule")
        if extraction:
            try:
                data = _json.loads(extraction) if isinstance(extraction, str) else (extraction if isinstance(extraction, dict) else None)
            except Exception:
                data = None
            if isinstance(data, dict):
                segs.append({
                    "type": "EventRule",
                    "anchors": {},
                    "features": _summarize_event_rule(data),
                    "confidence": 0.6,
                    "segment_confidence": {"extraction_confidence": 0.6, "classification_confidence": 0.6, "value_confidence": 0.7},
                })
    return segs

def _summarize_event_rule(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(data.get("mappings"), list):
        m = []
        for it in data["mappings"][:10]:
            if isinstance(it, dict):
                src = it.get("source") or it.get("from")
                tgt = it.get("target") or it.get("to")
                if src or tgt:
                    m.append({"source": src, "target": tgt})
        if m:
            out["mappings"] = m
    for k in ("publisher", "publishers", "producer", "consumers", "subscriber"):
        if data.get(k):
            out[k] = data.get(k)
    keys = data.get("keys") or data.get("keyFields")
    if keys:
        out["keys"] = keys
    return out


def _parse_markdown_tables(text: str) -> List[Dict[str, Any]]:
    """Very small markdown table parser.

    Recognizes blocks:
      | h1 | h2 |
      |----|----|
      | v1 | v2 |
    Returns list of {headers: [...], rows: [[...], ...]} (rows limited to 5).
    """
    lines = text.splitlines()
    tables: List[Dict[str, Any]] = []
    i = 0
    import re
    row_re = re.compile(r"^\s*\|.*\|\s*$")
    sep_re = re.compile(r"^\s*\|\s*[:\-]+(\|\s*[:\-]+)+\|\s*$")

    def split_row(s: str) -> List[str]:
        s = s.strip().strip('|')
        cells = [c.strip() for c in s.split('|')]
        return cells

    while i < len(lines) - 1:
        if row_re.match(lines[i]) and sep_re.match(lines[i + 1]):
            headers = split_row(lines[i])
            rows: List[List[str]] = []
            j = i + 2
            while j < len(lines) and row_re.match(lines[j]) and len(rows) < 5:
                rows.append(split_row(lines[j]))
                j += 1
            tables.append({"headers": headers, "rows": rows})
            i = j
        else:
            i += 1
    return tables
