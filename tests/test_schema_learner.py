"""
Tests for schema learner module.

Covers C1 response-driven schema learning functionality.
"""

import time
from unittest.mock import patch
from src.services.schema_learner import (
    SchemaLearner,
    get_schema_learner,
    observe_extraction,
    get_preferred_extraction_paths
)
from src.services.models import Passage


class TestSchemaLearner:
    """Test core schema learning functionality."""
    
    def test_initial_state(self):
        """Test learner starts with empty state."""
        learner = SchemaLearner()
        
        assert len(learner.store) == 0
        assert learner.preferred_paths("unknown-index") == [
            "inner_hits", "sections.content", "sections.text",
            "body", "content", "text", "description", "summary"
        ]
        assert learner.has_reliable_inner_hits("unknown-index") is False
    
    def test_observe_inner_hits_pattern(self):
        """Test learning from hits with inner_hits structure."""
        learner = SchemaLearner()
        
        # Simulate 5 hits with inner_hits
        for i in range(5):
            hit = {
                "_id": f"doc{i}",
                "_index": "test-index",
                "inner_hits": {"matched_sections": {"hits": {"hits": []}}}
            }
            passages = [Passage(
                doc_id=f"doc{i}", index="test-index", text=f"content {i}",
                section_title=None, score=0.5, page_url=None, api_name=None, title=f"Doc {i}"
            )]
            
            learner.observe_hit("test-index", hit, passages)
        
        profile = learner.store["test-index"]
        assert profile.samples == 5
        assert profile.inner_hits_seen == 5
        assert profile.content_paths["inner_hits"] == 5
        
        # Should now consider inner_hits reliable
        assert learner.has_reliable_inner_hits("test-index") is True
        
        # Preferred paths should prioritize inner_hits
        paths = learner.preferred_paths("test-index")
        assert paths[0] == "inner_hits"
    
    def test_observe_mixed_patterns(self):
        """Test learning from mixed hit patterns."""
        learner = SchemaLearner()
        
        # 3 hits with inner_hits
        for i in range(3):
            hit = {"_id": f"inner{i}", "inner_hits": {}}
            passages = [Passage(
                doc_id=f"inner{i}", index="mixed-index", text=f"inner content {i}",
                section_title=None, score=0.5, page_url=None, api_name=None, title=f"Inner {i}"
            )]
            learner.observe_hit("mixed-index", hit, passages)
        
        # 7 hits without inner_hits (sections.content)
        for i in range(7):
            hit = {"_id": f"sections{i}", "_source": {"sections": [{"content": "text"}]}}
            passages = [Passage(
                doc_id=f"sections{i}", index="mixed-index", text=f"sections content {i}",
                section_title=None, score=0.5, page_url=None, api_name=None, title=f"Sections {i}"
            )]
            learner.observe_hit("mixed-index", hit, passages)
        
        profile = learner.store["mixed-index"]
        assert profile.samples == 10
        assert profile.inner_hits_seen == 3
        assert profile.content_paths["inner_hits"] == 3
        assert profile.content_paths["sections.content"] == 7
        
        # inner_hits rate is 30% - not reliable
        assert learner.has_reliable_inner_hits("mixed-index") is False
        
        # Should prefer sections.content due to higher success rate
        paths = learner.preferred_paths("mixed-index")
        assert paths[0] == "sections.content"
        assert "inner_hits" in paths  # Still included as fallback
    
    def test_observe_no_content_extraction(self):
        """Test handling when no content is extracted."""
        learner = SchemaLearner()
        
        hit = {"_id": "empty", "_source": {"title": "No Content"}}
        passages = []  # No passages extracted
        
        learner.observe_hit("empty-index", hit, passages)
        
        profile = learner.store["empty-index"]
        assert profile.samples == 1
        assert profile.inner_hits_seen == 0
        assert profile.content_paths["no_content"] == 1
    
    def test_swagger_nested_retry_recommendation(self):
        """Test recommendation for Swagger nested retries."""
        learner = SchemaLearner()
        
        # Swagger index with low inner_hits reliability
        for i in range(5):
            hit = {"_id": f"swagger{i}", "_source": {"api_name": "Test"}}
            passages = []  # Simulate extraction failures
            learner.observe_hit("api-swagger-index", hit, passages)
        
        # Should recommend nested retry for unreliable Swagger index
        assert learner.should_retry_with_nested("api-swagger-index") is True
        
        # Regular index should not trigger retry
        assert learner.should_retry_with_nested("main-index") is False
        
        # Reliable Swagger index should not trigger retry
        reliable_learner = SchemaLearner()
        for i in range(5):
            hit = {"_id": f"reliable{i}", "inner_hits": {}}
            passages = [Passage(
                doc_id=f"reliable{i}", index="reliable-swagger-index", text="content",
                section_title=None, score=0.5, page_url=None, api_name=None, title="Reliable"
            )]
            reliable_learner.observe_hit("reliable-swagger-index", hit, passages)
        
        assert reliable_learner.should_retry_with_nested("reliable-swagger-index") is False
    
    def test_extraction_stats(self):
        """Test extraction statistics reporting."""
        learner = SchemaLearner()
        
        # Add some observations
        for i in range(10):
            has_inner_hits = i < 7  # 70% have inner_hits
            hit = {"_id": f"doc{i}"}
            if has_inner_hits:
                hit["inner_hits"] = {}
            
            passages = [Passage(
                doc_id=f"doc{i}", index="stats-index", text="content",
                section_title=None, score=0.5, page_url=None, api_name=None, title="Doc"
            )] if has_inner_hits else []
            
            learner.observe_hit("stats-index", hit, passages)
        
        stats = learner.get_extraction_stats("stats-index")
        
        assert stats["samples"] == 10
        assert stats["inner_hits_rate"] == 0.7
        assert stats["reliability"] == "medium"  # 70% is medium reliability
        assert "inner_hits" in stats["content_paths"]
        assert stats["content_paths"]["inner_hits"] == 7
    
    def test_ttl_and_cleanup(self):
        """Test TTL-based cleanup of old profiles."""
        learner = SchemaLearner(ttl_sec=1)  # Very short TTL for testing
        
        # Add an observation
        hit = {"_id": "old_doc"}
        passages = []
        learner.observe_hit("old-index", hit, passages)
        
        assert "old-index" in learner.store
        
        # Wait for TTL to expire
        with patch('time.time') as mock_time:
            # Simulate time progression
            mock_time.side_effect = [
                time.time(),  # Initial observation
                time.time() + 2  # 2 seconds later (past TTL)
            ]
            
            # Trigger cleanup
            removed = learner.cleanup_expired()
            
            assert removed == 1
            assert "old-index" not in learner.store
    
    def test_capacity_limits(self):
        """Test that learner respects max_indices capacity."""
        learner = SchemaLearner(max_indices=2)
        
        # Add 3 indices (exceeds capacity)
        for i in range(3):
            hit = {"_id": f"doc{i}"}
            passages = []
            learner.observe_hit(f"index-{i}", hit, passages)
        
        # Should only keep 2 indices (oldest evicted)
        assert len(learner.store) == 2
        assert "index-0" not in learner.store  # Oldest should be evicted
        assert "index-1" in learner.store
        assert "index-2" in learner.store
    
    def test_path_inference(self):
        """Test content path inference from hit structure."""
        learner = SchemaLearner()
        
        # Test inner_hits path
        hit_inner = {"inner_hits": {}}
        passages = [Passage(
            doc_id="test", index="test", text="content", section_title=None,
            score=0.5, page_url=None, api_name=None, title="Test"
        )]
        learner.observe_hit("test-index", hit_inner, passages)
        assert learner.store["test-index"].content_paths["inner_hits"] == 1
        
        # Test sections path
        hit_sections = {"_source": {"sections": [{"content": "text"}]}}
        learner.observe_hit("test-index", hit_sections, passages)
        assert learner.store["test-index"].content_paths["sections.content"] == 1
        
        # Test body field path
        hit_body = {"_source": {"body": "text"}}
        learner.observe_hit("test-index", hit_body, passages)
        assert learner.store["test-index"].content_paths["body"] == 1


class TestGlobalLearnerInstance:
    """Test global learner instance management."""
    
    def test_global_instance_singleton(self):
        """Test that global learner is a singleton."""
        learner1 = get_schema_learner()
        learner2 = get_schema_learner()
        
        assert learner1 is learner2
    
    def test_convenience_functions(self):
        """Test convenience functions work with global instance."""
        # Clear any existing state
        learner = get_schema_learner()
        learner.store.clear()
        
        # Use convenience function
        hit = {"_id": "test", "inner_hits": {}}
        passages = [Passage(
            doc_id="test", index="conv-index", text="content", section_title=None,
            score=0.5, page_url=None, api_name=None, title="Test"
        )]
        
        observe_extraction("conv-index", hit, passages)
        
        # Should be recorded in global learner
        assert "conv-index" in learner.store
        
        # Test preferred paths function
        paths = get_preferred_extraction_paths("conv-index") 
        assert "inner_hits" in paths


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_malformed_hit_structure(self):
        """Test handling of malformed hit structures."""
        learner = SchemaLearner()
        
        # Hit with None values
        hit = {"_id": None, "_source": None}
        passages = []
        
        # Should not crash
        learner.observe_hit("malformed-index", hit, passages)
        
        assert "malformed-index" in learner.store
        assert learner.store["malformed-index"].samples == 1
    
    def test_empty_index_name(self):
        """Test handling of empty or None index names."""
        learner = SchemaLearner()
        
        hit = {"_id": "test"}
        passages = []
        
        # Should handle gracefully
        learner.observe_hit("", hit, passages)
        learner.observe_hit(None, hit, passages)
        
        # Should create entries (even if weird)
        assert len(learner.store) >= 1
    
    def test_very_large_content_paths(self):
        """Test handling when content_paths dict gets large."""
        learner = SchemaLearner()
        
        # Add many different content paths
        for i in range(100):
            hit = {"_id": f"doc{i}", "_source": {f"field_{i}": "content"}}
            passages = [Passage(
                doc_id=f"doc{i}", index="large-index", text="content",
                section_title=None, score=0.5, page_url=None, api_name=None, title="Doc"
            )]
            learner.observe_hit("large-index", hit, passages)
        
        # Should still function correctly
        profile = learner.store["large-index"]
        assert profile.samples == 100
        assert len(profile.content_paths) == 100  # Each hit gets its own path