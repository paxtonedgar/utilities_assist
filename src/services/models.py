"""Data models for the services layer."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class IntentResult(BaseModel):
    """Result of intent classification."""
    intent: str  # "confluence", "swagger", "list", "info", "restart"
    confidence: float = 0.0
    reasoning: Optional[str] = None


class SearchResult(BaseModel):
    """Individual search result."""
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any] = {}


class RetrievalResult(BaseModel):
    """Combined retrieval results."""
    results: List[SearchResult]
    total_found: int
    retrieval_time_ms: float
    method: str  # "bm25", "knn", "rrf"
    diagnostics: Optional[Dict[str, Any]] = None


class SourceChip(BaseModel):
    """Source citation chip for UI."""
    title: str
    url: Optional[str] = None
    doc_id: str
    excerpt: Optional[str] = None


class TurnResult(BaseModel):
    """Complete result of a conversation turn."""
    answer: str
    sources: List[SourceChip] = []
    intent: IntentResult
    retrieval: Optional[RetrievalResult] = None
    response_time_ms: float = 0.0
    error: Optional[str] = None