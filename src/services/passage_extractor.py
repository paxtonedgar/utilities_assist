"""
Passage extraction service - A1+A2 logic for clean OpenSearch hit processing.

This module handles all complex content extraction from OpenSearch hits,
supporting multiple data structures (inner_hits, nested sections, flat fields).
"""

import logging
from typing import List, Dict, Any, Optional
from src.services.models import Passage, ExtractorConfig, RankedHit
from src.telemetry.logger import log_event

logger = logging.getLogger(__name__)


def extract_passages(hit: Dict[str, Any], cfg: ExtractorConfig) -> List[Passage]:
    """
    Extract usable passages from OpenSearch hit with fallback strategy.
    
    A1 Behavior: Try inner_hits first, then fallback to source fields.
    
    Args:
        hit: Raw OpenSearch hit dictionary
        cfg: Extractor configuration
        
    Returns:
        List of extracted passages with text content
    """
    # A1: Inner-hits first approach
    passages = _from_inner_hits(hit, cfg)
    
    if not passages:
        # A1: Fallback to source fields
        passages = _from_source_fields(hit, cfg)
    
    # A1: Postprocess and filter
    return _postprocess(passages=passages, cfg=cfg, src=hit.get("_source", {}))


def _from_inner_hits(hit: Dict[str, Any], cfg: ExtractorConfig) -> List[Passage]:
    """Extract passages from inner_hits.matched_sections structure."""
    try:
        inner_hits = (
            hit.get("inner_hits", {})
            .get("matched_sections", {})
            .get("hits", {})
            .get("hits", [])
        )
        
        if not inner_hits:
            return []
            
        passages = []
        source_meta = hit.get("_source", {})
        
        for sec in inner_hits[:cfg.max_sections]:
            sec_source = sec.get("_source", {})
            
            # Try section field order to find text content
            text = None
            for field in cfg.section_field_order:
                if sec_source.get(field):
                    text = str(sec_source[field]).strip()
                    break
                    
            if text:
                passage = Passage(
                    doc_id=hit.get("_id", ""),
                    index=hit.get("_index", ""),
                    text=text,
                    section_title=sec_source.get("title") or sec_source.get("heading"),
                    score=sec.get("_score", hit.get("_score", 0.0)),
                    page_url=source_meta.get("page_url"),
                    api_name=source_meta.get("api_name"),
                    title=source_meta.get("title"),
                )
                passages.append(passage)
                
        log_event(
            stage="extract_inner_hits",
            inner_hits_found=len(inner_hits),
            passages_extracted=len(passages),
            index=hit.get("_index"),
        )
        
        return passages
        
    except Exception as e:
        logger.warning(f"Inner hits extraction failed: {e}")
        return []


def _from_source_fields(hit: Dict[str, Any], cfg: ExtractorConfig) -> List[Passage]:
    """Extract passages from top-level _source fields with multiple fallbacks."""
    try:
        source = hit.get("_source", {})
        passages = []
        
        # Strategy 1: Extract from sections array
        sections = source.get("sections")
        if isinstance(sections, list):
            for section in sections[:3]:  # Limit to first 3 sections
                if isinstance(section, dict):
                    text = None
                    for field in cfg.section_field_order:
                        if section.get(field):
                            text = str(section[field]).strip()
                            break
                            
                    if text:
                        passage = _mk_passage(
                            hit, 
                            text, 
                            section.get("title") or section.get("heading")
                        )
                        passages.append(passage)
                        
            if passages:
                log_event(
                    stage="extract_sections_array", 
                    sections_found=len(sections),
                    passages_extracted=len(passages),
                    index=hit.get("_index"),
                )
                return passages
        
        # Strategy 2: Extract from document-level fields
        for field in cfg.doc_field_order:
            if source.get(field):
                text = str(source[field]).strip()
                if text:
                    passage = _mk_passage(hit, text, None)
                    log_event(
                        stage="extract_doc_field", 
                        field=field,
                        index=hit.get("_index"),
                    )
                    return [passage]
        
        # Strategy 3: No content found
        log_event(
            stage="extract_no_content",
            available_fields=list(source.keys()),
            index=hit.get("_index"),
        )
        return []
        
    except Exception as e:
        logger.warning(f"Source fields extraction failed: {e}")
        return []


def _mk_passage(hit: Dict[str, Any], text: str, section_title: Optional[str]) -> Passage:
    """Create passage from hit and text content."""
    source = hit.get("_source", {})
    return Passage(
        doc_id=hit.get("_id", ""),
        index=hit.get("_index", ""),
        text=text,
        section_title=section_title,
        score=hit.get("_score", 0.0),
        page_url=source.get("page_url"),
        api_name=source.get("api_name"),
        title=source.get("title"),
    )


def _postprocess(passages: List[Passage], cfg: ExtractorConfig, src: Dict[str, Any]) -> List[Passage]:
    """Clean and filter passages according to configuration."""
    cleaned = []
    
    for passage in passages:
        # Normalize whitespace
        text = " ".join(passage.text.split())
        
        # Apply length filters
        if cfg.min_chars <= len(text) <= cfg.max_chars:
            # Create new passage with cleaned text
            cleaned_passage = Passage(
                doc_id=passage.doc_id,
                index=passage.index,
                text=text,
                section_title=passage.section_title,
                score=passage.score,
                page_url=passage.page_url,
                api_name=passage.api_name,
                title=passage.title,
            )
            cleaned.append(cleaned_passage)
        else:
            log_event(
                stage="extract_filtered",
                reason="length",
                text_length=len(text),
                min_chars=cfg.min_chars,
                max_chars=cfg.max_chars,
            )
    
    return cleaned


def is_metadata_only_swagger(rh: RankedHit, cfg: ExtractorConfig) -> bool:
    """
    A2: Check if this is a Swagger hit with only metadata (should be filtered).
    
    Args:
        rh: RankedHit with extracted passages
        cfg: Extractor configuration
        
    Returns:
        True if this is a metadata-only Swagger hit that should be dropped
    """
    # Only apply to Swagger indices
    if not rh.index.endswith(cfg.swagger_suffix):
        return False
        
    # If we extracted passages, it's not metadata-only
    if rh.passages:
        return False
        
    # Check if hit only has metadata fields
    source_fields = set((rh.hit.get("_source", {})).keys())
    metadata_fields = {"api_name", "utility_name", "sections", "title", "method", "endpoint"}
    
    # If all fields are metadata fields, this is metadata-only
    is_metadata_only = source_fields.issubset(metadata_fields)
    
    if is_metadata_only:
        log_event(
            stage="swagger_metadata_only",
            index=rh.index,
            doc_id=rh.hit.get("_id"),
            available_fields=list(source_fields),
        )
    
    return is_metadata_only


def extract_passages_batch(hits: List[Dict[str, Any]], cfg: ExtractorConfig) -> List[RankedHit]:
    """
    Extract passages from multiple hits and create RankedHit objects.
    
    Args:
        hits: List of OpenSearch hits
        cfg: Extractor configuration
        
    Returns:
        List of RankedHit objects with extracted passages
    """
    ranked_hits = []
    
    for rank, hit in enumerate(hits):
        passages = extract_passages(hit, cfg)
        
        ranked_hit = RankedHit(
            hit=hit,
            passages=passages,
            rank_rrf=rank,
            index=hit.get("_index", ""),
        )
        ranked_hits.append(ranked_hit)
    
    # Log batch statistics
    total_passages = sum(len(rh.passages) for rh in ranked_hits)
    log_event(
        stage="extract_batch",
        hits_processed=len(hits),
        total_passages=total_passages,
        avg_passages_per_hit=total_passages / len(hits) if hits else 0,
    )
    
    return ranked_hits