"""
Tests for rerank policy module.

Covers A3 cross-encoder reranking policies and decisions.
"""

import time
from unittest.mock import Mock, patch
from src.services.rerank_policy import (
    maybe_rerank,
    _is_definitional,
    filter_metadata_only_swagger,
    should_trigger_text_retry
)
from src.services.models import RankedHit


class TestRerankPolicy:
    """Test cross-encoder reranking policy decisions."""
    
    def test_skip_definitional_queries(self):
        """Test that definitional queries skip CE reranking."""
        # Mock cross-encoder
        ce_model = Mock()
        
        rrf_hits = [
            RankedHit(
                hit={"_id": "test1", "_source": {"title": "Test Doc"}},
                passages=[],
                rank_rrf=0,
                index="main-index"
            )
        ]
        
        # Definitional query should be skipped
        result = maybe_rerank(
            rrf_hits=rrf_hits,
            query="what is CIU",
            ce_model=ce_model,
            skip_definitional=True
        )
        
        assert result.used_ce is False
        assert result.reason == 'skipped_definitional'
        assert result.items == rrf_hits
        
        # CE model should not have been called
        ce_model.predict.assert_not_called()
    
    def test_successful_reranking(self):
        """Test successful cross-encoder reranking."""
        # Mock cross-encoder with scores
        ce_model = Mock()
        ce_model.predict.return_value = [0.8, 0.6, 0.9]  # Scores for 3 candidates
        
        # Create test passages for extraction
        from src.services.models import Passage
        
        passages1 = [Passage(
            doc_id="doc1", index="main", text="First document content", 
            section_title=None, score=0.7, page_url=None, api_name=None, title="Doc 1"
        )]
        passages2 = [Passage(
            doc_id="doc2", index="main", text="Second document content",
            section_title=None, score=0.6, page_url=None, api_name=None, title="Doc 2"
        )]
        passages3 = [Passage(
            doc_id="doc3", index="main", text="Third document content",
            section_title=None, score=0.8, page_url=None, api_name=None, title="Doc 3"
        )]
        
        rrf_hits = [
            RankedHit(
                hit={"_id": "doc1", "_source": {"title": "Doc 1"}},
                passages=passages1,
                rank_rrf=0,
                index="main-index"
            ),
            RankedHit(
                hit={"_id": "doc2", "_source": {"title": "Doc 2"}}, 
                passages=passages2,
                rank_rrf=1,
                index="main-index"
            ),
            RankedHit(
                hit={"_id": "doc3", "_source": {"title": "Doc 3"}},
                passages=passages3,
                rank_rrf=2,
                index="main-index"
            )
        ]
        
        result = maybe_rerank(
            rrf_hits=rrf_hits,
            query="complex technical query requiring reranking",
            ce_model=ce_model,
            skip_definitional=True
        )
        
        assert result.used_ce is True
        assert result.reason == 'ok'
        
        # Results should be reordered by CE scores: doc3 (0.9), doc1 (0.8), doc2 (0.6)
        assert len(result.items) == 3
        assert result.items[0].hit["_id"] == "doc3"
        assert result.items[1].hit["_id"] == "doc1" 
        assert result.items[2].hit["_id"] == "doc2"
        
        # CE model should have been called with correct inputs
        ce_model.predict.assert_called_once()
        call_args = ce_model.predict.call_args[0][0]
        assert len(call_args) == 3  # 3 query-text pairs
        assert call_args[0][1] == "First document content"  # First passage text
    
    def test_rerank_timeout(self):
        """Test timeout handling in cross-encoder reranking."""
        # Mock CE model that takes too long
        ce_model = Mock()
        ce_model.predict.side_effect = lambda x: time.sleep(2)  # Simulate slow CE
        
        rrf_hits = [
            RankedHit(
                hit={"_id": "doc1", "_source": {"title": "Doc 1"}},
                passages=[],
                rank_rrf=0,
                index="main-index"
            )
        ]
        
        with patch('time.sleep'):  # Don't actually sleep in tests
            result = maybe_rerank(
                rrf_hits=rrf_hits,
                query="test query",
                ce_model=ce_model,
                timeout_ms=100  # Very short timeout
            )
        
        assert result.used_ce is False
        assert result.reason == 'timeout'
        assert result.items == rrf_hits
    
    def test_insufficient_candidates(self):
        """Test handling when too few candidates for reranking."""
        ce_model = Mock()
        
        # Only one hit - insufficient for reranking
        rrf_hits = [
            RankedHit(
                hit={"_id": "doc1", "_source": {"title": "Doc 1"}},
                passages=[],
                rank_rrf=0,
                index="main-index"
            )
        ]
        
        result = maybe_rerank(
            rrf_hits=rrf_hits,
            query="test query", 
            ce_model=ce_model
        )
        
        assert result.used_ce is False
        assert result.reason == 'insufficient_candidates'
        assert result.items == rrf_hits


class TestDefinitionalDetection:
    """Test definitional query detection logic."""
    
    def test_what_is_queries(self):
        """Test detection of 'what is X' queries."""
        assert _is_definitional("what is CIU") is True
        assert _is_definitional("What is the API") is True
        assert _is_definitional("what are utilities") is True
    
    def test_definition_queries(self):
        """Test detection of definition queries."""
        assert _is_definitional("define CIU") is True
        assert _is_definitional("definition of API") is True
        assert _is_definitional("CIU definition") is True
    
    def test_short_queries(self):
        """Test that very short queries are considered definitional."""
        assert _is_definitional("CIU") is True
        assert _is_definitional("API") is True
        assert _is_definitional("short") is True
        assert _is_definitional("a b c") is True  # 3 tokens
    
    def test_non_definitional_queries(self):
        """Test that complex queries are not definitional."""
        assert _is_definitional("how to configure CIU for production") is False
        assert _is_definitional("troubleshoot API connection issues") is False
        assert _is_definitional("step by step guide") is False
        assert _is_definitional("best practices for implementation") is False


class TestSwaggerFiltering:
    """Test A2 Swagger metadata filtering logic."""
    
    def test_filter_metadata_only_swagger(self):
        """Test filtering of metadata-only Swagger hits."""
        hits = [
            # Metadata-only Swagger hit (should be filtered)
            RankedHit(
                hit={
                    "_id": "swagger_meta",
                    "_source": {
                        "api_name": "TestAPI",
                        "utility_name": "TestUtil",
                        "sections": []
                    }
                },
                passages=[],  # No passages
                rank_rrf=0,
                index="test-swagger-index"
            ),
            # Swagger hit with content (should be kept)
            RankedHit(
                hit={
                    "_id": "swagger_content", 
                    "_source": {
                        "api_name": "TestAPI",
                        "body": "Real documentation"
                    }
                },
                passages=[],  # Would have passages if we ran extraction
                rank_rrf=1,
                index="test-swagger-index"
            ),
            # Non-Swagger hit (should be kept)
            RankedHit(
                hit={
                    "_id": "regular_doc",
                    "_source": {"api_name": "TestAPI"}
                },
                passages=[],
                rank_rrf=2,
                index="main-index"
            )
        ]
        
        filtered = filter_metadata_only_swagger(hits, "-swagger-index")
        
        # Should remove the metadata-only Swagger hit
        assert len(filtered) == 2
        assert filtered[0].hit["_id"] == "swagger_content"
        assert filtered[1].hit["_id"] == "regular_doc"
    
    def test_text_retry_trigger(self):
        """Test when text retry should be triggered."""
        # Case 1: Many hits but few passages
        hits_few_passages = [
            RankedHit(hit={"_id": f"doc{i}"}, passages=[], rank_rrf=i, index="main")
            for i in range(5)
        ]
        hits_few_passages[0].passages = [Mock()]  # Only one hit has passages
        
        assert should_trigger_text_retry(hits_few_passages) is True
        
        # Case 2: Top hits have no passages
        hits_no_top_passages = [
            RankedHit(hit={"_id": f"doc{i}"}, passages=[], rank_rrf=i, index="main")
            for i in range(3)
        ]
        
        assert should_trigger_text_retry(hits_no_top_passages) is True
        
        # Case 3: Good passage coverage (should not trigger)
        hits_good_coverage = [
            RankedHit(hit={"_id": f"doc{i}"}, passages=[Mock()], rank_rrf=i, index="main")
            for i in range(3)
        ]
        
        assert should_trigger_text_retry(hits_good_coverage) is False


class TestErrorHandling:
    """Test error handling in rerank policies."""
    
    def test_ce_model_error(self):
        """Test handling when CE model throws an error."""
        ce_model = Mock()
        ce_model.predict.side_effect = Exception("CE model failed")
        
        rrf_hits = [
            RankedHit(
                hit={"_id": "doc1", "_source": {"title": "Doc 1"}},
                passages=[],
                rank_rrf=0,
                index="main-index"
            )
        ]
        
        result = maybe_rerank(
            rrf_hits=rrf_hits,
            query="test query",
            ce_model=ce_model
        )
        
        assert result.used_ce is False
        assert result.reason == 'error'
        assert result.items == rrf_hits  # Should fallback to original
    
    def test_empty_hits_handling(self):
        """Test handling of empty hit lists."""
        ce_model = Mock()
        
        result = maybe_rerank(
            rrf_hits=[],
            query="test query",
            ce_model=ce_model
        )
        
        assert result.used_ce is False
        assert result.reason == 'insufficient_candidates'
        assert result.items == []