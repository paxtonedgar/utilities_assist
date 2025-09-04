"""
Graph schema definitions for steps and entities.

Defines minimal data structures for Step/Entity nodes and typed edges with
provenance and confidence. This module is implementation-agnostic (usable with
Neo4j, NetworkX, or OpenSearch documents).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class Evidence:
    doc_id: str
    index: str
    page_url: Optional[str] = None
    section_title: Optional[str] = None
    text_snippet: Optional[str] = None
    offsets: Optional[Dict[str, int]] = None  # e.g., {"start": 120, "end": 180}


@dataclass
class Entity:
    id: str
    name: str
    type: Optional[str] = None  # e.g., service, pipeline, environment
    synonyms: List[str] = field(default_factory=list)


@dataclass
class Step:
    id: str
    label: str  # human-readable, e.g., "Configure CI pipeline"
    verb: Optional[str] = None
    obj: Optional[str] = None
    entities: List[str] = field(default_factory=list)  # entity ids
    order: Optional[int] = None  # intra-doc sequence
    evidence: Optional[Evidence] = None


@dataclass
class Edge:
    src: str
    dst: str
    type: str  # NEXT | REQUIRES | MENTIONS | SAME_AS
    confidence: float = 0.5
    evidence_refs: List[Evidence] = field(default_factory=list)


NEO4J_NODE_LABELS = {
    "Step": ["id", "label", "verb", "obj", "order"],
    "Entity": ["id", "name", "type"],
}

NEO4J_EDGE_TYPES = {
    "NEXT": ["confidence"],
    "REQUIRES": ["confidence"],
    "MENTIONS": ["confidence"],
    "SAME_AS": ["confidence"],
}

