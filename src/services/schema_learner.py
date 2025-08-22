"""
Response-driven schema learner - C1 logic for adaptive extraction.

This module learns from OpenSearch responses to automatically tune extraction
strategies without requiring direct cluster access or manual configuration.
"""

import logging
import time
from typing import Dict, List, Optional
from src.services.models import IndexProfile, Passage
from src.telemetry.logger import log_event

logger = logging.getLogger(__name__)


class SchemaLearner:
    """
    C1: Response-driven schema learning for adaptive content extraction.
    
    Observes OpenSearch response patterns per index to:
    - Learn which content extraction paths work best
    - Detect indices with reliable inner_hits vs those without
    - Guide retry strategies and optimization hints
    """
    
    def __init__(self, ttl_sec: int = 86400, max_indices: int = 100):
        """
        Initialize schema learner with TTL and capacity limits.
        
        Args:
            ttl_sec: Time to live for index profiles (default 1 day)
            max_indices: Maximum number of indices to track
        """
        self.ttl = ttl_sec
        self.max_indices = max_indices
        self.store: Dict[str, IndexProfile] = {}
    
    def observe_hit(self, index: str, hit: Dict, passages: List[Passage]) -> None:
        """
        Observe an OpenSearch hit and update schema knowledge.
        
        Args:
            index: Index name
            hit: Raw OpenSearch hit
            passages: Extracted passages from the hit
        """
        try:
            # Ensure we don't exceed max indices
            if len(self.store) >= self.max_indices and index not in self.store:
                self._evict_oldest()
            
            # Get or create profile for this index
            profile = self.store.setdefault(index, IndexProfile())
            
            # Update sample count
            profile.samples += 1
            profile.last_seen = time.time()
            
            # Track inner_hits presence
            if "inner_hits" in hit:
                profile.inner_hits_seen += 1
            
            # Infer and record content extraction path
            if passages:
                content_path = self._infer_content_path(hit, passages)
                profile.content_paths[content_path] = profile.content_paths.get(content_path, 0) + 1
            else:
                # Record failed extraction
                profile.content_paths["no_content"] = profile.content_paths.get("no_content", 0) + 1
            
            # Log observation (sampled to avoid spam)
            if profile.samples % 10 == 0:  # Log every 10th observation
                log_event(
                    stage="schema_observation",
                    index=index,
                    samples=profile.samples,
                    inner_hits_rate=profile.inner_hits_seen / profile.samples,
                    top_content_path=self._get_top_content_path(profile),
                )
                
        except Exception as e:
            logger.warning(f"Schema observation failed for {index}: {e}")
    
    def preferred_paths(self, index: str) -> List[str]:
        """
        Get preferred content extraction paths for an index.
        
        Returns paths ordered by observed success rate, with fallbacks.
        
        Args:
            index: Index name
            
        Returns:
            Ordered list of content extraction paths to try
        """
        profile = self.store.get(index)
        
        if not profile or profile.samples < 5:
            # Not enough data, return default ordering
            return [
                "inner_hits", 
                "sections.content", 
                "sections.text",
                "body", 
                "content", 
                "text", 
                "description", 
                "summary"
            ]
        
        # Sort by success count (descending)
        ranked_paths = sorted(
            profile.content_paths.items(), 
            key=lambda kv: kv[1], 
            reverse=True
        )
        
        # Extract path names, excluding failures
        learned_paths = [path for path, count in ranked_paths if path != "no_content"]
        
        # Always include fallbacks not yet observed
        fallback_paths = [
            "inner_hits", "sections.content", "sections.text", 
            "body", "content", "text", "description", "summary"
        ]
        
        # Combine learned + fallbacks (avoiding duplicates)
        final_paths = learned_paths.copy()
        for fallback in fallback_paths:
            if fallback not in final_paths:
                final_paths.append(fallback)
        
        return final_paths
    
    def has_reliable_inner_hits(self, index: str, min_rate: float = 0.6, min_samples: int = 5) -> bool:
        """
        Check if an index reliably provides inner_hits.
        
        Args:
            index: Index name
            min_rate: Minimum rate of inner_hits presence (0.0-1.0)
            min_samples: Minimum samples needed for reliable assessment
            
        Returns:
            True if index reliably has inner_hits
        """
        profile = self.store.get(index)
        
        if not profile or profile.samples < min_samples:
            return False  # Not enough data, assume unreliable
        
        inner_hits_rate = profile.inner_hits_seen / profile.samples
        return inner_hits_rate >= min_rate
    
    def should_retry_with_nested(self, index: str) -> bool:
        """
        Determine if a nested query retry is recommended for this index.
        
        Useful for Swagger indices that sometimes need different query structures.
        
        Args:
            index: Index name
            
        Returns:
            True if nested retry is recommended
        """
        # Swagger indices with low inner_hits reliability
        if index.endswith("-swagger-index"):
            return not self.has_reliable_inner_hits(index, min_rate=0.4)
        
        return False
    
    def get_extraction_stats(self, index: str) -> Dict:
        """
        Get extraction statistics for an index.
        
        Args:
            index: Index name
            
        Returns:
            Dictionary with extraction statistics
        """
        profile = self.store.get(index)
        
        if not profile:
            return {
                "samples": 0,
                "inner_hits_rate": 0.0,
                "top_content_path": "unknown",
                "reliability": "insufficient_data"
            }
        
        inner_hits_rate = profile.inner_hits_seen / profile.samples if profile.samples > 0 else 0.0
        top_path = self._get_top_content_path(profile)
        
        # Assess reliability
        if profile.samples < 5:
            reliability = "insufficient_data"
        elif inner_hits_rate >= 0.8:
            reliability = "high"
        elif inner_hits_rate >= 0.5:
            reliability = "medium"
        else:
            reliability = "low"
        
        return {
            "samples": profile.samples,
            "inner_hits_rate": inner_hits_rate,
            "top_content_path": top_path,
            "reliability": reliability,
            "content_paths": dict(profile.content_paths),
            "last_seen": profile.last_seen,
        }
    
    def cleanup_expired(self) -> int:
        """
        Remove expired index profiles.
        
        Returns:
            Number of profiles removed
        """
        current_time = time.time()
        expired_indices = []
        
        for index, profile in self.store.items():
            if current_time - profile.last_seen > self.ttl:
                expired_indices.append(index)
        
        for index in expired_indices:
            del self.store[index]
        
        if expired_indices:
            log_event(
                stage="schema_cleanup",
                expired_count=len(expired_indices),
                remaining_count=len(self.store),
            )
        
        return len(expired_indices)
    
    def _infer_content_path(self, hit: Dict, passages: List[Passage]) -> str:
        """Infer which path was used to extract content."""
        if not passages:
            return "no_content"
        
        # Check for inner_hits structure
        if "inner_hits" in hit:
            return "inner_hits"
        
        # Check for sections array in source
        source = hit.get("_source", {})
        if isinstance(source.get("sections"), list):
            return "sections.content"
        
        # Try to infer from source fields
        content_fields = ["body", "content", "text", "description", "summary"]
        for field in content_fields:
            if source.get(field):
                return field
        
        return "unknown"
    
    def _get_top_content_path(self, profile: IndexProfile) -> str:
        """Get the most successful content path for a profile."""
        if not profile.content_paths:
            return "unknown"
        
        # Filter out failures and get top path
        valid_paths = {k: v for k, v in profile.content_paths.items() if k != "no_content"}
        
        if not valid_paths:
            return "no_content"
        
        return max(valid_paths.items(), key=lambda kv: kv[1])[0]
    
    def _evict_oldest(self) -> None:
        """Evict the oldest index profile to make room."""
        if not self.store:
            return
        
        oldest_index = min(self.store.items(), key=lambda kv: kv[1].last_seen)[0]
        del self.store[oldest_index]
        
        log_event(
            stage="schema_eviction",
            evicted_index=oldest_index,
            remaining_count=len(self.store),
        )


# Global schema learner instance
_global_learner: Optional[SchemaLearner] = None


def get_schema_learner() -> SchemaLearner:
    """Get the global schema learner instance."""
    global _global_learner
    if _global_learner is None:
        _global_learner = SchemaLearner()
    return _global_learner


def observe_extraction(index: str, hit: Dict, passages: List[Passage]) -> None:
    """Convenience function to observe extraction results."""
    learner = get_schema_learner()
    learner.observe_hit(index, hit, passages)


def get_preferred_extraction_paths(index: str) -> List[str]:
    """Convenience function to get preferred extraction paths."""
    learner = get_schema_learner()
    return learner.preferred_paths(index)