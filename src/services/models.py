"""Data models for the services layer."""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel
import time


class IntentResult(BaseModel):
    """Result of intent classification."""

    intent: str  # "confluence", "swagger", "list", "info", "restart"
    confidence: float = 0.0
    reasoning: Optional[str] = None


class SearchResult(BaseModel):
    """Individual search result with canonical schema."""

    doc_id: str
    title: str
    url: Optional[str] = None
    score: float
    content: str  # Main content/snippet
    metadata: Dict[str, Any] = {}
    rerank_score: Optional[float] = None  # Cross-encoder relevance score

    @property
    def text(self) -> str:
        """Backward compatibility alias for content field."""
        return self.content


class RetrievalResult(BaseModel):
    """Combined retrieval results."""

    results: List["Passage"]
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


# New models for passage extraction refactor

@dataclass
class Passage:
    """Unified passage/result type - replaces SearchResult entirely."""
    doc_id: str
    index: str
    text: str  # Main content (was .content)
    score: float
    title: Optional[str] = None
    section_title: Optional[str] = None
    page_url: Optional[str] = None
    api_name: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)  # Was .metadata
    rerank_score: Optional[float] = None
    
    # Properties for clean access patterns with defensive programming
    @property
    def url(self) -> Optional[str]:
        """URL for display/linking - checks page_url then meta"""
        if hasattr(self, 'page_url') and self.page_url:
            return self.page_url
        if hasattr(self, 'meta') and self.meta:
            return self.meta.get("url") or self.meta.get("page_url")
        return None
    
    @property  
    def content(self) -> str:
        """Alias for text during migration"""
        return getattr(self, 'text', '')
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Alias for meta during migration"""
        return getattr(self, 'meta', {})


@dataclass
class RankedHit:
    """OpenSearch hit with extracted passages and RRF ranking."""
    hit: Dict[str, Any]
    passages: List[Passage]
    rank_rrf: int
    index: str


@dataclass
class RerankResult:
    """Result of cross-encoder reranking with policy decisions."""
    items: List[RankedHit]
    used_ce: bool
    reason: str  # 'ok' | 'timeout' | 'collapse' | 'skipped_definitional'


@dataclass
class ExtractorConfig:
    """Configuration for passage extraction."""
    section_field_order: List[str] = field(default_factory=lambda: ['content', 'text'])
    doc_field_order: List[str] = field(default_factory=lambda: ['body', 'content', 'text', 'description', 'summary'])
    max_sections: int = 5
    min_chars: int = 80
    max_chars: int = 1200
    swagger_suffix: str = "-swagger-index"
    drop_metadata_only_swagger: bool = True


@dataclass
class IndexProfile:
    """Schema learning profile for an index."""
    samples: int = 0
    inner_hits_seen: int = 0
    content_paths: Dict[str, int] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)
