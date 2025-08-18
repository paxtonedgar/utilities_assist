"""
Unit tests for LangGraph components - A2 requirement validation.

Tests that each node/tool returns same outputs as current pipeline.
Uses mocks to avoid LLM/DB dependencies.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent.tools.search import search_index_tool, adaptive_search_tool, multi_index_search_tool
from agent.nodes.summarize import summarize_node
from agent.nodes.intent import intent_node
from agent.nodes.combine import combine_node
from agent.graph import create_graph, NODE_REGISTRY
from services.models import SearchResult, RetrievalResult, IntentResult


class TestSearchTools:
    """Test search tools match existing pipeline outputs."""
    
    @pytest.fixture
    def mock_search_client(self):
        """Mock OpenSearch client."""
        client = Mock()
        return client
    
    @pytest.fixture
    def mock_embed_client(self):
        """Mock embedding client.""" 
        client = AsyncMock()
        return client
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample search results for testing."""
        return [
            SearchResult(
                doc_id="doc1",
                content="Customer Summary Utility provides account information",
                score=0.9,
                metadata={"title": "Customer Summary Docs", "url": "http://example.com/1"}
            ),
            SearchResult(
                doc_id="doc2",
                content="API rate limit is 100 requests per second",
                score=0.8,
                metadata={"title": "Rate Limiting", "url": "http://example.com/2"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_search_index_tool_enhanced_rrf(self, mock_search_client, mock_embed_client, sample_search_results):
        """Test search_index_tool with enhanced RRF strategy."""
        
        # Mock the search functions
        mock_result = RetrievalResult(
            results=sample_search_results,
            total_found=2,
            retrieval_time_ms=100,
            method="enhanced_rrf"
        )
        
        with patch('agent.tools.search.enhanced_rrf_search') as mock_search:
            with patch('agent.tools.search.create_single_embedding') as mock_embed:
                mock_embed.return_value = [0.1] * 1024  # Mock embedding
                mock_search.return_value = (mock_result, {})  # Mock search result
                
                result = await search_index_tool(
                    index="confluence_current",
                    query="customer summary utility",
                    search_client=mock_search_client,
                    embed_client=mock_embed_client,
                    strategy="enhanced_rrf"
                )
                
                # Validate output matches RetrievalResult structure
                assert isinstance(result, RetrievalResult)
                assert len(result.results) == 2
                assert result.method == "enhanced_rrf"
                assert result.results[0].content == "Customer Summary Utility provides account information"
                
                # Ensure search was called correctly
                mock_search.assert_called_once()
                mock_embed.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_index_tool_bm25_fallback(self, mock_search_client, mock_embed_client, sample_search_results):
        """Test search_index_tool falls back to BM25 when embedding fails."""
        
        mock_result = RetrievalResult(
            results=sample_search_results,
            total_found=2,
            retrieval_time_ms=50,
            method="bm25"
        )
        
        with patch('agent.tools.search.bm25_search') as mock_bm25:
            with patch('agent.tools.search.create_single_embedding') as mock_embed:
                from embedding_creation import EmbeddingError
                mock_embed.side_effect = EmbeddingError("Embedding failed")
                mock_bm25.return_value = mock_result
                
                result = await search_index_tool(
                    index="confluence_current",
                    query="test query",
                    search_client=mock_search_client,
                    embed_client=mock_embed_client,
                    strategy="enhanced_rrf"
                )
                
                # Should fallback to BM25
                assert result.method == "bm25"
                assert len(result.results) == 2
                mock_bm25.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_adaptive_search_tool_intent_routing(self, mock_search_client, mock_embed_client, sample_search_results):
        """Test adaptive_search_tool routes based on intent like original _perform_retrieval."""
        
        mock_result = RetrievalResult(
            results=sample_search_results,
            total_found=2,
            retrieval_time_ms=75,
            method="bm25"
        )
        
        with patch('agent.tools.search.search_index_tool') as mock_search_tool:
            mock_search_tool.return_value = mock_result
            
            # Test list intent routing (should use BM25)
            result = await adaptive_search_tool(
                query="list all utilities",
                intent_confidence=0.8,
                intent_type="list",
                search_client=mock_search_client,
                embed_client=mock_embed_client
            )
            
            # Verify it called search_index_tool with BM25 strategy
            mock_search_tool.assert_called_once()
            call_args = mock_search_tool.call_args
            assert call_args[1]["strategy"] == "bm25"  # List queries use BM25
            
            # Test high confidence non-list query (should use enhanced_rrf)
            mock_search_tool.reset_mock()
            
            result = await adaptive_search_tool(
                query="what is customer summary utility",
                intent_confidence=0.8,
                intent_type="confluence",
                search_client=mock_search_client,
                embed_client=mock_embed_client
            )
            
            call_args = mock_search_tool.call_args
            assert call_args[1]["strategy"] == "enhanced_rrf"  # High confidence uses enhanced RRF
    
    @pytest.mark.asyncio 
    async def test_multi_index_search_tool(self, mock_search_client, mock_embed_client, sample_search_results):
        """Test multi_index_search_tool searches multiple indices."""
        
        mock_result = RetrievalResult(
            results=sample_search_results,
            total_found=2,
            retrieval_time_ms=100,
            method="enhanced_rrf"
        )
        
        with patch('agent.tools.search.search_index_tool') as mock_search_tool:
            mock_search_tool.return_value = mock_result
            
            results = await multi_index_search_tool(
                indices=["confluence_current", "khub-opensearch-swagger-index"],
                query="test query",
                search_client=mock_search_client,
                embed_client=mock_embed_client
            )
            
            # Should return list of results (one per index)
            assert isinstance(results, list)
            assert len(results) == 2  # Two indices
            
            # Should have called search_index_tool twice
            assert mock_search_tool.call_count == 2
            
            # Each result should be RetrievalResult
            for result in results:
                assert isinstance(result, RetrievalResult)


class TestGraphNodes:
    """Test graph nodes produce same outputs as existing pipeline functions."""
    
    @pytest.fixture
    def mock_resources(self):
        """Mock RAG resources."""
        resources = Mock()
        resources.chat_client = AsyncMock()
        resources.embed_client = AsyncMock()
        resources.search_client = Mock()
        resources.settings = Mock()
        resources.settings.chat.model = "test-model"
        resources.settings.embed.model = "test-embed-model"
        resources.get_config_param = Mock(side_effect=lambda key, default: default)
        return resources
    
    @pytest.fixture
    def sample_state(self):
        """Sample graph state."""
        return {
            "original_query": "What is the Customer Summary Utility?",
            "normalized_query": None,
            "intent": None,
            "search_results": [],
            "workflow_path": [],
            "error_messages": []
        }
    
    @pytest.mark.asyncio
    async def test_summarize_node_output_format(self, mock_resources, sample_state):
        """Test summarize node produces same format as normalize_query."""
        
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = "what is customer summary utility"
        mock_resources.chat_client.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await summarize_node(sample_state, mock_resources)
        
        # Should return dict with normalized_query and workflow_path
        assert "normalized_query" in result
        assert "workflow_path" in result
        assert isinstance(result["normalized_query"], str)
        assert len(result["normalized_query"]) > 0
        assert "summarize" in result["workflow_path"]
    
    @pytest.mark.asyncio
    async def test_summarize_node_fallback_behavior(self, mock_resources, sample_state):
        """Test summarize node falls back to original normalize_query on LLM failure."""
        
        # Mock LLM failure
        mock_resources.chat_client.ainvoke = AsyncMock(side_effect=Exception("LLM failed"))
        
        with patch('agent.nodes.summarize.normalize_query') as mock_normalize:
            mock_normalize.return_value = "fallback normalized query"
            
            result = await summarize_node(sample_state, mock_resources)
            
            # Should use fallback
            assert result["normalized_query"] == "fallback normalized query"
            mock_normalize.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_intent_node_output_format(self, mock_resources, sample_state):
        """Test intent node produces same IntentResult as determine_intent."""
        
        sample_state["normalized_query"] = "what is customer summary utility"
        
        # Mock structured output
        mock_analysis = Mock()
        mock_analysis.intent = "confluence"
        mock_analysis.confidence = 0.8
        mock_analysis.reasoning = "Documentary query about utility"
        
        mock_structured = AsyncMock(return_value=mock_analysis)
        mock_resources.chat_client.with_structured_output = Mock(return_value=mock_structured)
        
        result = await intent_node(sample_state, mock_resources)
        
        # Should return dict with intent and workflow_path
        assert "intent" in result
        assert "workflow_path" in result
        assert isinstance(result["intent"], IntentResult)
        assert result["intent"].intent == "confluence"
        assert result["intent"].confidence == 0.8
        assert "intent" in result["workflow_path"]
    
    @pytest.mark.asyncio
    async def test_intent_node_fallback_behavior(self, mock_resources, sample_state):
        """Test intent node falls back to original determine_intent on LLM failure."""
        
        sample_state["normalized_query"] = "test query"
        
        # Mock LLM failure
        mock_resources.chat_client.with_structured_output = Mock(
            return_value=AsyncMock(side_effect=Exception("Structured output failed"))
        )
        
        with patch('agent.nodes.intent.determine_intent') as mock_determine:
            mock_determine.return_value = IntentResult(intent="fallback", confidence=0.5)
            
            result = await intent_node(sample_state, mock_resources)
            
            # Should use fallback
            assert result["intent"].intent == "fallback"
            assert result["intent"].confidence == 0.5
            mock_determine.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_combine_node_deduplication(self, mock_resources):
        """Test combine node deduplicates results like existing pipeline."""
        
        # Create duplicate results
        results = [
            SearchResult(doc_id="doc1", content="Content 1", score=0.9, metadata={"title": "Doc 1"}),
            SearchResult(doc_id="doc2", content="Content 2", score=0.8, metadata={"title": "Doc 2"}),
            SearchResult(doc_id="doc1", content="Content 1", score=0.7, metadata={"title": "Doc 1"})  # Duplicate
        ]
        
        state = {
            "search_results": results,
            "normalized_query": "test query",
            "intent": IntentResult(intent="confluence", confidence=0.8),
            "workflow_path": [],
            "performance_metrics": {}
        }
        
        result = await combine_node(state, mock_resources)
        
        # Should deduplicate and return combined results
        assert "combined_results" in result
        assert "final_context" in result
        assert "workflow_path" in result
        
        # Should keep only 2 results (deduplicated)
        combined = result["combined_results"]
        assert len(combined) == 2
        
        # Should keep the higher scoring duplicate
        doc1_results = [r for r in combined if r.doc_id == "doc1"]
        assert len(doc1_results) == 1
        assert doc1_results[0].score == 0.9  # Higher score kept


class TestGraphIntegration:
    """Test complete graph integration and behavior."""
    
    @pytest.mark.asyncio
    async def test_graph_creation(self):
        """Test graph can be created without errors."""
        
        graph = create_graph(enable_loops=True)
        assert graph is not None
        
        # Test with loops disabled
        graph_no_loops = create_graph(enable_loops=False)
        assert graph_no_loops is not None
    
    def test_node_registry_completeness(self):
        """Test all required nodes are in registry."""
        required_nodes = [
            "summarize", "intent", "search_confluence", 
            "search_swagger", "search_multi", "rewrite_query",
            "combine", "answer"
        ]
        
        for node_name in required_nodes:
            assert node_name in NODE_REGISTRY, f"Node {node_name} missing from registry"
            assert callable(NODE_REGISTRY[node_name]), f"Node {node_name} is not callable"
    
    @pytest.mark.asyncio
    async def test_compound_question_routing(self):
        """Test A3 requirement: compound questions make ≥2 search calls."""
        
        # This would need to be tested with actual graph execution
        # For now, test the routing logic directly
        from agent.graph import _route_after_intent, GraphState
        
        # Test comparative query routing
        comparative_state = GraphState({
            "normalized_query": "compare customer summary utility and account utility",
            "intent": IntentResult(intent="comparative", confidence=0.8)
        })
        
        route = _route_after_intent(comparative_state)
        assert route == "search_multi", "Comparative queries should route to multi-search"
        
        # Test single entity query routing
        single_state = GraphState({
            "normalized_query": "what is customer summary utility", 
            "intent": IntentResult(intent="confluence", confidence=0.8)
        })
        
        route = _route_after_intent(single_state)
        assert route == "search_confluence", "Single queries should route to single search"


if __name__ == "__main__":
    # Run specific test for debugging
    pytest.main([__file__ + "::TestSearchTools::test_search_index_tool_enhanced_rrf", "-v"])