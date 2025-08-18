"""
Unit tests for LangGraph workflow components.

These tests validate the workflow logic without requiring LLMs or databases,
using mocks and fixtures to test the core functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

from workflows.state import (
    WorkflowState, 
    WorkflowConfig, 
    initialize_workflow_state,
    add_workflow_step,
    log_error,
    update_metrics
)
from workflows.agents import (
    query_decomposer_agent,
    search_orchestrator_agent,
    single_search_agent, 
    parallel_search_agent,
    result_synthesizer_agent
)
from workflows.response_generator import response_generator_agent
from workflows.langgraph_workflow import LangGraphWorkflow, create_langgraph_workflow
from services.models import IntentResult, SearchResult, RetrievalResult


class TestWorkflowState:
    """Test workflow state management."""
    
    def test_initialize_workflow_state(self):
        """Test workflow state initialization."""
        query = "What is the Customer Summary Utility?"
        request_id = "test_123"
        user_context = {"test": "data"}
        
        state = initialize_workflow_state(query, request_id, user_context)
        
        assert state["original_query"] == query
        assert state["request_id"] == request_id
        assert state["user_context"] == user_context
        assert state["search_results"] == []
        assert state["response_chunks"] == []
        assert state["workflow_path"] == []
        assert state["error_messages"] == []
    
    def test_add_workflow_step(self):
        """Test workflow step tracking."""
        state = initialize_workflow_state("test query")
        update = add_workflow_step(state, "test_agent")
        
        assert update == {"workflow_path": ["test_agent"]}
    
    def test_log_error(self):
        """Test error logging."""
        state = initialize_workflow_state("test query")
        error_msg = "Test error message"
        update = log_error(state, error_msg)
        
        assert update == {"error_messages": [error_msg]}
    
    def test_update_metrics(self):
        """Test metrics updating."""
        state = initialize_workflow_state("test query")
        state["performance_metrics"] = {"existing_metric": 10.0}
        
        new_metrics = {"new_metric": 20.0}
        update = update_metrics(state, new_metrics)
        
        expected_metrics = {"existing_metric": 10.0, "new_metric": 20.0}
        assert update["performance_metrics"] == expected_metrics


class TestWorkflowConfig:
    """Test workflow configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WorkflowConfig()
        
        assert config.simple_query_threshold == 0.3
        assert config.complex_query_threshold == 0.7
        assert config.max_parallel_searches == 3
        assert config.search_timeout_seconds == 30
        assert config.enable_cross_index_search is True
        assert config.max_context_length == 8000
        assert config.diversity_lambda == 0.75
        assert config.enable_streaming is True
        assert config.chunk_size == 50
        assert config.enable_caching is True
        assert config.cache_ttl_seconds == 300
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = WorkflowConfig(
            simple_query_threshold=0.4,
            max_parallel_searches=5,
            enable_streaming=False
        )
        
        assert config.simple_query_threshold == 0.4
        assert config.max_parallel_searches == 5
        assert config.enable_streaming is False


class TestAgentMocking:
    """Test agent functionality with mocked dependencies."""
    
    @pytest.fixture
    def mock_resources(self):
        """Create mock RAG resources."""
        resources = Mock()
        resources.chat_client = AsyncMock()
        resources.embed_client = AsyncMock()
        resources.search_client = Mock()
        resources.settings = Mock()
        resources.settings.chat.model = "test-model"
        resources.settings.embed.model = "test-embed"
        resources.settings.search.index_alias = "test-index"
        resources.get_config_param = Mock(side_effect=lambda key, default: default)
        return resources
    
    @pytest.fixture
    def sample_state(self):
        """Create sample workflow state."""
        return initialize_workflow_state(
            query="Compare Customer Summary Utility and Account Utility login APIs",
            request_id="test_123",
            user_context={"use_mock_corpus": False}
        )
    
    @pytest.fixture
    def sample_intent(self):
        """Create sample intent result."""
        return IntentResult(intent="comparative", confidence=0.8)
    
    @pytest.fixture  
    def sample_search_results(self):
        """Create sample search results."""
        return [
            SearchResult(
                doc_id="doc1",
                content="Customer Summary Utility API has rate limit of 100 req/sec",
                score=0.9,
                metadata={"title": "Customer Summary API Docs", "url": "http://example.com/1"}
            ),
            SearchResult(
                doc_id="doc2", 
                content="Account Utility API supports OAuth 2.0 authentication",
                score=0.8,
                metadata={"title": "Account Utility Auth", "url": "http://example.com/2"}
            )
        ]
    
    @pytest.mark.asyncio
    async def test_query_decomposer_agent_simple(self, mock_resources, sample_state):
        """Test query decomposer for simple queries."""
        # Mock normalize_query
        with patch('workflows.agents.normalize_query', return_value="what is customer summary utility"):
            # Mock determine_intent
            with patch('workflows.agents.determine_intent', return_value=AsyncMock(return_value=IntentResult(intent="factual", confidence=0.6))):
                # Mock LLM structured output
                mock_analysis = Mock()
                mock_analysis.complexity = "simple"
                mock_analysis.decomposition_needed = False
                mock_analysis.query_type = "factual"
                mock_analysis.sub_queries = []
                mock_analysis.entities = ["Customer Summary Utility"]
                mock_analysis.reasoning = "Single factual query"
                
                mock_resources.chat_client.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_analysis)
                
                result = await query_decomposer_agent(sample_state, mock_resources)
                
                assert result["normalized_query"] == "what is customer summary utility"
                assert result["query_complexity"] == "simple"
                assert result["sub_queries"] is None
                assert "query_decomposer" in result["workflow_path"]
                assert "query_analysis_ms" in result["performance_metrics"]
    
    @pytest.mark.asyncio
    async def test_query_decomposer_agent_complex(self, mock_resources, sample_state):
        """Test query decomposer for complex queries.""" 
        sample_state["original_query"] = "Compare login APIs of Customer Summary and Account utilities"
        
        # Mock normalize_query
        with patch('workflows.agents.normalize_query', return_value="compare login apis customer summary account utilities"):
            # Mock determine_intent
            with patch('workflows.agents.determine_intent', return_value=AsyncMock(return_value=IntentResult(intent="comparative", confidence=0.8))):
                # Mock LLM structured output
                mock_analysis = Mock()
                mock_analysis.complexity = "comparative"
                mock_analysis.decomposition_needed = True
                mock_analysis.query_type = "comparative"
                mock_analysis.sub_queries = ["Customer Summary Utility login API", "Account Utility login API"]
                mock_analysis.entities = ["Customer Summary Utility", "Account Utility"]
                mock_analysis.reasoning = "Comparative query requiring separate searches"
                
                mock_resources.chat_client.with_structured_output.return_value.ainvoke = AsyncMock(return_value=mock_analysis)
                
                result = await query_decomposer_agent(sample_state, mock_resources)
                
                assert result["query_complexity"] == "comparative"
                assert result["sub_queries"] == ["Customer Summary Utility login API", "Account Utility login API"]
                assert len(result["sub_queries"]) == 2
    
    @pytest.mark.asyncio
    async def test_query_decomposer_error_handling(self, mock_resources, sample_state):
        """Test query decomposer error handling."""
        # Mock an exception in normalize_query
        with patch('workflows.agents.normalize_query', side_effect=Exception("Normalization failed")):
            result = await query_decomposer_agent(sample_state, mock_resources)
            
            assert "query_decomposer" in result["workflow_path"]
            assert len(result["error_messages"]) > 0
            assert "Normalization failed" in result["error_messages"][0]
    
    @pytest.mark.asyncio
    async def test_single_search_agent_success(self, mock_resources, sample_state, sample_search_results):
        """Test single search agent with successful search."""
        # Set up state
        sample_state["normalized_query"] = "customer summary utility"
        sample_state["intent"] = IntentResult(intent="factual", confidence=0.8)
        
        # Mock search functions
        mock_result = RetrievalResult(
            results=sample_search_results,
            total_found=2,
            retrieval_time_ms=100,
            method="enhanced_rrf"
        )
        
        with patch('workflows.agents.create_single_embedding', return_value=AsyncMock(return_value=[0.1] * 1024)):
            with patch('workflows.agents.enhanced_rrf_search', return_value=AsyncMock(return_value=(mock_result, {}))):
                result = await single_search_agent(sample_state, mock_resources)
                
                assert len(result["search_results"]) == 2
                assert "single_search" in result["result_sources"]
                assert "single_search" in result["workflow_path"]
                assert result["performance_metrics"]["results_found"] == 2
    
    @pytest.mark.asyncio
    async def test_single_search_agent_fallback(self, mock_resources, sample_state, sample_search_results):
        """Test single search agent fallback to BM25."""
        # Set up state
        sample_state["normalized_query"] = "customer summary utility"
        sample_state["intent"] = IntentResult(intent="factual", confidence=0.5)  # Low confidence
        
        # Mock BM25 search
        mock_result = RetrievalResult(
            results=sample_search_results,
            total_found=2,
            retrieval_time_ms=50,
            method="bm25"
        )
        
        with patch('workflows.agents.bm25_search', return_value=AsyncMock(return_value=mock_result)):
            result = await single_search_agent(sample_state, mock_resources)
            
            assert len(result["search_results"]) == 2
            assert result["performance_metrics"]["results_found"] == 2
    
    @pytest.mark.asyncio
    async def test_parallel_search_agent(self, mock_resources, sample_state, sample_search_results):
        """Test parallel search agent."""
        # Set up state for parallel search
        sample_state["current_query"] = "Customer Summary Utility login API"
        sample_state["search_id"] = "search_0"
        sample_state["search_strategy"] = "enhanced_rrf"
        sample_state["search_context"] = "comparative_part_0"
        
        # Mock search
        mock_result = RetrievalResult(
            results=sample_search_results,
            total_found=2,
            retrieval_time_ms=75,
            method="enhanced_rrf"
        )
        
        with patch('workflows.agents.create_single_embedding', return_value=AsyncMock(return_value=[0.1] * 1024)):
            with patch('workflows.agents.enhanced_rrf_search', return_value=AsyncMock(return_value=(mock_result, {}))):
                result = await parallel_search_agent(sample_state, mock_resources)
                
                assert len(result["search_results"]) == 2
                assert "parallel_search_search_0" in result["workflow_path"]
                
                # Check that results are tagged with search context
                for search_result in result["search_results"]:
                    assert search_result.metadata["search_id"] == "search_0"
                    assert search_result.metadata["search_context"] == "comparative_part_0"
    
    @pytest.mark.asyncio
    async def test_result_synthesizer_agent_no_results(self, mock_resources, sample_state):
        """Test result synthesizer with no results."""
        sample_state["search_results"] = []
        sample_state["normalized_query"] = "test query"
        sample_state["query_complexity"] = "simple"
        
        result = await result_synthesizer_agent(sample_state, mock_resources)
        
        assert result["synthesized_context"] == "No relevant information found."
        assert "result_synthesizer" in result["workflow_path"]
    
    @pytest.mark.asyncio
    async def test_result_synthesizer_agent_with_results(self, mock_resources, sample_state, sample_search_results):
        """Test result synthesizer with search results."""
        sample_state["search_results"] = sample_search_results
        sample_state["normalized_query"] = "customer summary utility"
        sample_state["query_complexity"] = "simple"
        
        result = await result_synthesizer_agent(sample_state, mock_resources)
        
        assert result["synthesized_context"] != "No relevant information found."
        assert "Customer Summary Utility API" in result["synthesized_context"]
        assert "result_synthesizer" in result["workflow_path"]
        assert result["performance_metrics"]["total_results_processed"] == 2


class TestWorkflowIntegration:
    """Test workflow integration without LLM calls."""
    
    def test_create_langgraph_workflow(self):
        """Test workflow creation."""
        workflow = create_langgraph_workflow()
        assert isinstance(workflow, LangGraphWorkflow)
        assert workflow.config is not None
        assert workflow._workflow is not None
    
    def test_create_langgraph_workflow_with_config(self):
        """Test workflow creation with custom config."""
        config = WorkflowConfig(max_parallel_searches=5)
        workflow = create_langgraph_workflow(config)
        
        assert workflow.config.max_parallel_searches == 5
    
    def test_workflow_error_handler(self):
        """Test workflow error handling."""
        workflow = create_langgraph_workflow()
        
        # Test error handler
        state = initialize_workflow_state("test query")
        state["error_messages"] = ["Error 1", "Error 2", "Error 3", "Error 4"]
        
        result = asyncio.run(workflow._error_handler(state))
        
        assert "Error" in result["final_answer"]
        assert "error_handler" in result["workflow_path"]
    
    def test_workflow_progress_formatting(self):
        """Test progress update formatting.""" 
        workflow = create_langgraph_workflow()
        
        node_update = {
            "workflow_path": ["test_step"],
            "performance_metrics": {"test_metric": 100.0}
        }
        
        result = workflow._format_progress_update("query_decomposer", node_update, "test_123")
        
        assert result["type"] == "status"
        assert result["message"] == "Analyzing query complexity..."
        assert result["req_id"] == "test_123"
        assert result["node"] == "query_decomposer"
        assert result["details"]["workflow_path"] == ["test_step"]
        assert result["details"]["metrics"] == {"test_metric": 100.0}
    
    def test_workflow_final_result_formatting(self, sample_search_results):
        """Test final result formatting."""
        workflow = create_langgraph_workflow()
        
        final_state = {
            "final_answer": "Test answer",
            "search_results": sample_search_results,
            "intent": IntentResult(intent="factual", confidence=0.8),
            "workflow_path": ["step1", "step2"],
            "performance_metrics": {"total_time": 500.0}
        }
        
        result = workflow._format_final_result(final_state, "test_123")
        
        assert result["type"] == "complete"
        assert result["result"]["answer"] == "Test answer"
        assert len(result["result"]["sources"]) == 2
        assert result["result"]["workflow_path"] == ["step1", "step2"]
        assert result["result"]["metrics"] == {"total_time": 500.0}
    
    @pytest.fixture
    def sample_search_results(self):
        """Create sample search results for testing."""
        return [
            SearchResult(
                doc_id="doc1",
                content="Test content 1", 
                score=0.9,
                metadata={"title": "Doc 1", "url": "http://test1.com"}
            ),
            SearchResult(
                doc_id="doc2",
                content="Test content 2",
                score=0.8,
                metadata={"title": "Doc 2", "url": "http://test2.com"}
            )
        ]


if __name__ == "__main__":
    # Run tests manually for debugging
    pytest.main([__file__, "-v"])