from __future__ import annotations

"""Shared pydantic models used across the taxonomy term pipeline."""

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conlist


class LabelPrototype(BaseModel):
    """Represents the prototype embedding for a taxonomy label."""

    class_id: str
    embedding: List[float]
    document_ids: List[str] = Field(default_factory=list)
    text_length: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TermCandidate(BaseModel):
    """Raw term candidate with contexts prior to scoring."""

    term_id: str
    surface: str
    class_id: str
    frequency: int
    contexts: conlist(str, min_items=1)
    embedding: List[float]
    doc_ids: List[str] = Field(default_factory=list)
    stats: Dict[str, float] = Field(default_factory=dict)


class ScoredTerm(BaseModel):
    """Term candidate enriched with contrastive scores and selection flag."""

    term_id: str
    surface: str
    class_id: str
    scores: Dict[str, float]
    selected: bool = False
    evidence: conlist(str, min_items=1)
    frequency: int = 0
    doc_ids: List[str] = Field(default_factory=list)


class ContrastiveMetrics(BaseModel):
    """Summary statistics for contrastive scoring."""

    total_terms: int = 0
    selected_terms: int = 0
    margin_median: float = 0.0
    margin_p10: float = 0.0
    specificity_median: float = 0.0
    parent_delta_median: float = 0.0


class GazetteerEntry(BaseModel):
    """Gazetteer entry ready for downstream indexing."""

    entity_id: str
    class_id: str
    preferred_name: str
    aliases: List[str] = Field(default_factory=list)
    evidence_contexts: conlist(str, min_items=1)
    source: str
    status: Literal["active", "candidate"] = "candidate"
    scores: Dict[str, float] = Field(default_factory=dict)


class BuildManifest(BaseModel):
    """Metadata describing a taxonomy term build."""

    taxonomy_version: str = "v1"
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    seed: int = 0
    code_sha: Optional[str] = None
    parameters: Dict[str, float] = Field(default_factory=dict)
    counts: Dict[str, int] = Field(default_factory=dict)

