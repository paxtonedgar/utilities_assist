from __future__ import annotations

"""
Index profile registry and utilities.

Profiles guide Stage 1 parsing/extraction by indicating primary/structured
fields and priorities. Detection uses index name prefixes/patterns.
"""

from typing import Dict, Any
import re


PROFILES: Dict[str, Dict[str, Any]] = {
    "swagger": {
        "match": [r"^khub-.*swagger.*", r"^khub-opensearch-swagger.*"],
        "primary_fields": ["sections[].content", "sections[].info"],
        "structured_fields": ["sections[].components", "sections[].server_url"],
        "extraction_priority": ["endpoints", "api_specs", "ownership_from_contact"],
    },
    "events": {
        "match": [r"^khub-.*events.*", r"^khub-.*ciuevents.*", r"^khub-opensearch-.*events.*"],
        "primary_fields": ["metadata.extractionRule"],
        "structured_fields": ["metadata.publishers", "metadata.description"],
        "extraction_priority": ["json_rules", "data_flows", "event_mappings"],
    },
    "product": {
        "match": [r"^khub-product-.*", r"^khub-test-md.*", r"^khub-opensearch-index.*"],
        "primary_fields": ["sections[].content"],
        "structured_fields": ["page_url"],
        "extraction_priority": ["tables", "workflows", "internal_links"],
        "html_rate": 0.56,
    },
}


def detect_profile(index_name: str) -> str:
    for name, cfg in PROFILES.items():
        for pat in cfg.get("match", []):
            try:
                if re.search(pat, index_name or ""):
                    return name
            except Exception:
                continue
    return "product"  # default generic


def get_profile_config(profile: str) -> Dict[str, Any]:
    return PROFILES.get(profile, PROFILES["product"]).copy()

