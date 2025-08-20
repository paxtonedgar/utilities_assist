# tests/test_coverage.py
"""
Unit tests for the coverage evaluation system.

Tests cross-encoder answerability scoring, aspect recall, and alpha-nDCG metrics.
"""

import pytest
import numpy as np
from src.quality.coverage import CoverageGate, Passage
from src.quality.subquery import decompose


class TestCoverageGate:
    """Test the CoverageGate class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a lightweight model for testing
        self.gate = CoverageGate(
            model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",  # Smaller model for tests
            tau=0.45,
            alpha=0.5,
            gate_ar=0.60,
            gate_andcg=0.40,
            min_actionable_spans=3
        )
    
    def test_aspect_recall_basic(self):
        """Test basic aspect recall calculation."""
        q = "enable Payment API"
        subqs = decompose(q, max_subqs=4)
        passages = [
            Passage("Steps:\n1. Create Jira\n2. Get API key\nContact: dl-payments@x", {"rank": 1}),
            Passage("GET /v1/payments\nAuth: Bearer ...", {"rank": 2}),
            Passage("Owner: Payments Core Team\nSlack #payments", {"rank": 3}),
        ]
        
        ev = self.gate.evaluate(q, subqs, passages)
        
        # Should have reasonable aspect recall
        assert ev["aspect_recall"] >= 0.0
        assert ev["aspect_recall"] <= 1.0
        assert ev["actionable_spans"] >= 2  # Should detect steps and contact info
        assert "picks" in ev
        assert len(ev["picks"]) <= len(subqs)
    
    def test_score_matrix_shape(self):
        """Test that score matrix has correct dimensions."""
        subqs = ["Prerequisites for: test", "Steps for: test", "Owner for: test"]
        passages = [
            Passage("Some content here", {"rank": 1}),
            Passage("More content here", {"rank": 2})
        ]
        
        mat = self.gate.score_matrix(subqs, passages)
        
        assert mat.shape == (3, 2)  # 3 subqueries, 2 passages
        assert np.all(mat >= 0.0)   # All scores should be non-negative
        assert np.all(mat <= 1.0)   # All scores should be <= 1.0
    
    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        # Empty subqueries
        mat1 = self.gate.score_matrix([], [Passage("content", {"rank": 1})])
        assert mat1.shape == (0, 1)
        
        # Empty passages
        mat2 = self.gate.score_matrix(["query"], [])
        assert mat2.shape == (1, 0)
        
        # Both empty
        mat3 = self.gate.score_matrix([], [])
        assert mat3.shape == (0, 0)
    
    def test_alpha_ndcg_calculation(self):
        """Test alpha-nDCG calculation."""
        # Create a simple test case
        mat = np.array([
            [0.8, 0.6, 0.3],  # subquery 1 scores
            [0.2, 0.9, 0.4],  # subquery 2 scores
        ], dtype="float32")
        
        ranks = [1, 2, 3]  # passage ranks
        
        alpha_ndcg = self.gate.alpha_ndcg(mat, ranks)
        
        assert 0.0 <= alpha_ndcg <= 1.0
        
        # Test with empty matrix
        empty_mat = np.array([], dtype="float32").reshape(0, 0)
        assert self.gate.alpha_ndcg(empty_mat, []) == 0.0
    
    def test_select_passages(self):
        """Test passage selection based on threshold."""
        mat = np.array([
            [0.8, 0.3, 0.6],  # subquery 1: passage 0 and 2 above tau=0.45
            [0.2, 0.9, 0.4],  # subquery 2: only passage 1 above tau
        ], dtype="float32")
        
        subqs = ["query1", "query2"]
        passages = [
            Passage("content1", {"rank": 1}),
            Passage("content2", {"rank": 2}),
            Passage("content3", {"rank": 3}),
        ]
        
        picks = self.gate.select_passages(mat, subqs, passages, top_n_per_aspect=2)
        
        # Should pick passages that meet threshold
        assert 0 in picks  # subquery index 0
        assert 1 in picks  # subquery index 1
        
        # Check that selected passages meet threshold
        for sq_idx, passage_idxs in picks.items():
            for p_idx in passage_idxs:
                assert mat[sq_idx, p_idx] >= self.gate.tau
    
    def test_actionable_spans_counting(self):
        """Test detection of actionable content spans."""
        passages = [
            Passage("Steps:\n1. Do this\n2. Do that", {"rank": 1}),  # Has steps
            Passage("GET /api/v1/endpoint", {"rank": 2}),            # Has endpoint
            Passage("Contact team: dl-support@company.com", {"rank": 3}),  # Has contact
            Passage("Just some random text content", {"rank": 4}),   # No actionable spans
        ]
        
        count = self.gate.count_actionable_spans(passages)
        assert count >= 3  # Should detect at least 3 actionable passages
    
    def test_feature_detection(self):
        """Test individual feature pattern detection."""
        # Test steps pattern
        steps_text = "Steps:\n1. First step\n2. Second step"
        features1 = self.gate._features(steps_text)
        assert features1["steps"] == 1
        
        # Test endpoint pattern  
        endpoint_text = "GET /api/v1/users endpoint for user data"
        features2 = self.gate._features(endpoint_text)
        assert features2["endpoint"] == 1
        
        # Test jira pattern
        jira_text = "Create a Jira ticket for access request"
        features3 = self.gate._features(jira_text)
        assert features3["jira"] == 1
        
        # Test owner pattern
        owner_text = "Owner: Platform Team, contact dl-platform@company.com"
        features4 = self.gate._features(owner_text)
        assert features4["owner"] == 1
        
        # Test table pattern
        table_text = "\n| Column1 | Column2 |\n|---------|----------|\n"
        features5 = self.gate._features(table_text)
        assert features5["table"] == 1


class TestSubqueryDecomposition:
    """Test subquery decomposition functionality."""
    
    def test_basic_decomposition(self):
        """Test basic query decomposition."""
        query = "enable Payment API"
        subqs = decompose(query, max_subqs=6)
        
        assert len(subqs) <= 6
        assert len(subqs) > 0
        
        # Should generate aspect-based subqueries
        subq_text = " ".join(subqs).lower()
        assert "prerequisites" in subq_text or "prerequisite" in subq_text
        assert "jira" in subq_text or "steps" in subq_text
    
    def test_action_verb_detection(self):
        """Test detection of action verbs."""
        action_query = "configure API gateway"
        subqs = decompose(action_query)
        
        # Should generate a direct steps subquery for action verbs
        steps_subq = next((sq for sq in subqs if "exact steps" in sq.lower()), None)
        assert steps_subq is not None
    
    def test_deduplication(self):
        """Test that duplicate subqueries are removed."""
        query = "setup test"
        subqs = decompose(query, max_subqs=10)
        
        # Should not have duplicates
        assert len(subqs) == len(set(subqs))
    
    def test_fallback_behavior(self):
        """Test fallback to original query when no decomposition possible."""
        # Empty query should return original
        empty_result = decompose("")
        assert len(empty_result) == 1
        assert empty_result[0] == ""
        
        # Very short query should still generate subqueries
        short_result = decompose("API")
        assert len(short_result) > 1


class TestCoverageIntegration:
    """Test integration between components."""
    
    def test_end_to_end_evaluation(self):
        """Test complete coverage evaluation workflow."""
        gate = CoverageGate(
            model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
            tau=0.3,  # Lower threshold for test
            gate_ar=0.5,
            gate_andcg=0.3,
            min_actionable_spans=1
        )
        
        user_query = "how to onboard Payment API"
        subqs = decompose(user_query)
        
        passages = [
            Passage(
                "Payment API Onboarding Steps:\n1. Create Jira ticket\n2. Get client ID\n3. Configure endpoints", 
                {"rank": 1, "url": "http://example.com/payment-api", "title": "Payment API Guide"}
            ),
            Passage(
                "POST /api/v1/payment\nAuthorization: Bearer token\nOwner: Payments Team",
                {"rank": 2, "url": "http://example.com/payment-endpoints", "title": "Payment Endpoints"}
            ),
            Passage(
                "Contact dl-payments@company.com for API access. Team Slack: #payments-support",
                {"rank": 3, "url": "http://example.com/payment-support", "title": "Payment Support"}
            )
        ]
        
        result = gate.evaluate(user_query, subqs, passages)
        
        # Verify result structure
        assert "aspect_recall" in result
        assert "alpha_ndcg" in result
        assert "actionable_spans" in result
        assert "gate_pass" in result
        assert "picks" in result
        
        # Should pass with good content
        assert result["actionable_spans"] >= 2  # Should detect multiple actionable passages
        assert result["aspect_recall"] > 0.0
        
        # Should have selected some passages
        total_selected = sum(len(idxs) for idxs in result["picks"].values())
        assert total_selected > 0


# Integration test with mocked SearchResult objects
class MockSearchResult:
    """Mock SearchResult for testing."""
    
    def __init__(self, content: str, url: str = "", title: str = "", metadata: dict = None):
        self.content = content
        self.url = url
        self.title = title
        self.metadata = metadata or {}


def test_search_result_conversion():
    """Test conversion from SearchResult to Passage objects."""
    from src.agent.nodes.search_to_compose import convert_search_results_to_passages
    
    search_results = [
        MockSearchResult(
            content="Payment API setup steps: 1. Get token 2. Configure endpoint",
            url="http://example.com/payment",
            title="Payment Setup",
            metadata={"heading": "Setup Guide"}
        ),
        MockSearchResult(
            content="Contact: payments-support@company.com for help",
            url="http://example.com/support",
            title="Support Info"
        )
    ]
    
    passages = convert_search_results_to_passages(search_results)
    
    assert len(passages) == 2
    assert passages[0]["text"] == search_results[0].content
    assert passages[0]["url"] == search_results[0].url
    assert passages[0]["title"] == search_results[0].title
    assert passages[0]["heading"] == "Setup Guide"
    assert passages[0]["rank"] == 1
    
    assert passages[1]["rank"] == 2
    assert passages[1]["heading"] == ""  # No heading in metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])