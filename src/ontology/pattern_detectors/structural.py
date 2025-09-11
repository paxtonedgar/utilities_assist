from __future__ import annotations

import re
from typing import Dict

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # optional dependency
    BeautifulSoup = None


_BULLET_RE = re.compile(r"(?m)^\s*(?:[-–•*]|\d+[.)])\s+\S")
_ORDERED_RE = re.compile(r"(?m)^\s*(?:\d+\.|\(\d+\))\s+\S")
_MARKDOWN_TABLE_RE = re.compile(r"(?m)^\|.+\|\s*$")
_BACKTICKS_RE = re.compile(r"`[^`]+`")
_HEADER_RE = re.compile(r"(?m)^(#+)\s+\S|^\s*(?:[A-Z][A-Z0-9 _-]{3,})\s*$")


def _analyze_html_struct(html: str) -> Dict[str, int]:
    if not BeautifulSoup:
        # lightweight tag counters
        return {
            "tables": len(re.findall(r"<table\b", html, flags=re.I)),
            "headers": len(re.findall(r"<h[1-6]\b", html, flags=re.I)),
            "lists": len(re.findall(r"<(ul|ol)\b", html, flags=re.I)),
            "code_blocks": len(re.findall(r"<(pre|code)\b", html, flags=re.I)),
        }
    soup = BeautifulSoup(html, "html.parser")
    return {
        "tables": len(soup.find_all("table")),
        "headers": sum(len(soup.find_all(f"h{i}")) for i in range(1, 7)),
        "lists": len(soup.find_all(["ul", "ol"])),
        "code_blocks": len(soup.find_all(["pre", "code"])),
    }


def analyze(text: str, is_html: bool) -> Dict[str, int]:
    if is_html:
        base = _analyze_html_struct(text)
        # include inline cues too
        base["inline_backticks"] = len(_BACKTICKS_RE.findall(text))
        return base
    # plain text / markdown-like
    return {
        "lists": len(_BULLET_RE.findall(text)),
        "ordered_lists": len(_ORDERED_RE.findall(text)),
        "headers": len(_HEADER_RE.findall(text)),
        "tables": len(_MARKDOWN_TABLE_RE.findall(text)),
        "code_blocks": text.count("```"),
        "inline_backticks": len(_BACKTICKS_RE.findall(text)),
    }

