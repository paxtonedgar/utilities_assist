"""
Graph construction scaffolding.

Builds Step/Entity nodes and typed edges from extracted steps and linked
entities. Provides a minimal interface to export to Neo4j or to OpenSearch.
"""

from typing import List, Dict, Any, Tuple
from .graph_schema import Step, Entity, Edge, Evidence


def build_intra_doc_edges(steps: List[Step]) -> List[Edge]:
    """Create NEXT edges based on step.order when available."""
    edges: List[Edge] = []
    ordered = [s for s in steps if s.order is not None]
    ordered.sort(key=lambda s: s.order)
    for a, b in zip(ordered, ordered[1:]):
        edges.append(Edge(src=a.id, dst=b.id, type="NEXT", confidence=0.9))
    return edges


def infer_requires_edges(steps: List[Step]) -> List[Edge]:
    """Placeholder for `REQUIRES` inference using lexical/LLM cues."""
    # TODO: implement rule-based and optional LLM validation for prerequisites
    return []


def export_to_neo4j(nodes_steps: List[Step], nodes_entities: List[Entity], edges: List[Edge]) -> None:
    """Export nodes and edges to Neo4j (placeholder).

    Use neo4j driver and UNWIND batches to MERGE nodes and edges.
    """
    # TODO: implement driver wiring and write queries
    pass


def index_steps_to_opensearch(steps: List[Step]) -> None:
    """Optional: index steps into an OpenSearch index for search/QA."""
    # TODO: define minimal mapping and bulk index steps for discoverability
    pass

