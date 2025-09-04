"""
Pydantic schemas for Step/Entity/Edge documents for storage/indexing.
Includes versioning and timestamps to support lifecycle and drift detection.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time


class EvidenceRef(BaseModel):
    doc_id: str
    index: Optional[str] = None
    snippet: Optional[str] = None
    offsets: Optional[Dict[str, int]] = None
    content_hash: Optional[str] = None


class StepDoc(BaseModel):
    id: str
    canonical_label: str
    verb: Optional[str] = None
    obj: Optional[str] = None
    entities: List[str] = []
    doc_id: Optional[str] = None
    section: Optional[str] = None
    order: Optional[int] = None
    evidence: Optional[EvidenceRef] = None
    embedding: Optional[List[float]] = None
    version: str = Field(default="v1")
    created_at: float = Field(default_factory=lambda: time.time())
    updated_at: float = Field(default_factory=lambda: time.time())


class EdgeDoc(BaseModel):
    src_step: str
    dst_step: str
    type: str
    confidence: float
    signal_breakdown: Dict[str, float] = {}
    evidence_refs: List[EvidenceRef] = []
    valid_from: float = Field(default_factory=lambda: time.time())
    valid_to: Optional[float] = None
    version: str = Field(default="v1")

