from __future__ import annotations

import re
from typing import Dict

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None

# Confluence storage format often uses ac:structured-macro, ri: / atlassian-* namespaces
MACRO_RE = re.compile(r"ac:structured-macro|data-macro-name=\"([^\"]+)\"", re.I)
PAGE_PROP_RE = re.compile(r"(page-properties)|ac:parameter|ac:layout", re.I)
INTERNAL_LINK_RE = re.compile(r"/wiki|/display/|/pages/viewpage", re.I)


def analyze_html(html: str) -> Dict[str, int]:
    macros = len(MACRO_RE.findall(html))
    page_props = len(PAGE_PROP_RE.findall(html))
    internal_links = len(INTERNAL_LINK_RE.findall(html))

    # Also detect macro elements if BeautifulSoup is present
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        macros += len(soup.find_all(attrs={"ac:name": True}))
        internal_links += sum(1 for a in soup.find_all("a") if a.get("href", "").startswith("/"))

    return {
        "macros": macros,
        "page_properties": page_props,
        "internal_links": internal_links,
    }

