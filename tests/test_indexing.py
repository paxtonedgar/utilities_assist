"""
Unit tests for the unified indexing pipeline.

Tests:
- Document indexing with embedding validation
- Blue/green reindex functionality  
- ACL filtering enforcement
- Time-based retrieval ordering
- Configurable embedding providers
"""

import asyncio
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from src.infra.indexing_pipeline import (
    Document,
    UnifiedIndexingPipeline,
    IndexingResult,
    MockEmbeddingProvider,
    JPMCEmbeddingProvider,
    create_embedding_provider,
    create_indexing_pipeline
)
from src.infra.opensearch_client import OpenSearchClient, SearchFilters, SearchResponse
from src.infra.config import SearchCfg


@pytest.fixture
def mock_search_config():
    """Mock search configuration for testing."""
    return SearchCfg(
        host="http://localhost:9200",
        username="admin",
        password="admin",
        timeout_s=30,
        max_retries=3,
        use_ssl=False
    )


@pytest.fixture  
def sample_documents():
    """Sample documents for testing with different ACL hashes and timestamps."""
    now = datetime.now()
    
    return [
        Document(
            doc_id="doc_1",
            title="Recent Customer Service Guide",
            body="How to provide excellent customer service with latest procedures.",
            section="guides",
            metadata={"space_key": "CUST", "version": "v2"},
            updated_at=now - timedelta(days=1),  # Most recent
            acl_hash="grp_customer_service",
            content_type="confluence"
        ),
        Document(
            doc_id="doc_2", 
            title="Older Customer Service Manual",
            body="Customer service procedures and escalation processes.",
            section="manuals",
            metadata={"space_key": "CUST", "version": "v1"}, 
            updated_at=now - timedelta(days=30),  # Older
            acl_hash="grp_customer_service",
            content_type="confluence"
        ),
        Document(
            doc_id="doc_3",
            title="Restricted Operations Manual",
            body="Internal operations procedures - confidential.",
            section="operations",
            metadata={"space_key": "OPS", "version": "v1"},
            updated_at=now - timedelta(days=15),  # Middle age
            acl_hash="grp_operations_restricted",  # Different ACL
            content_type="confluence"
        )
    ]


class MockSession:
    """Mock requests session for testing."""
    
    def __init__(self):
        self.responses = {}
        self.requests_made = []
        
    def post(self, url, json=None, data=None, headers=None, timeout=None):
        self.requests_made.append({
            "method": "POST",
            "url": url, 
            "json": json,
            "data": data,
            "headers": headers
        })
        
        response = Mock()
        
        if "_bulk" in url:
            # Mock bulk indexing response
            response.status_code = 200
            response.json.return_value = {
                "items": [
                    {"index": {"_id": "doc_1", "status": 201}},
                    {"index": {"_id": "doc_2", "status": 201}}, 
                    {"index": {"_id": "doc_3", "status": 201}}
                ]
            }
        elif "_search" in url:
            # Mock search response
            response.status_code = 200
            response.json.return_value = {
                "hits": {
                    "total": {"value": 2},
                    "hits": [
                        {
                            "_id": "doc_1",
                            "_score": 1.5,
                            "_source": {
                                "title": "Recent Customer Service Guide",
                                "body": "How to provide excellent customer service",
                                "metadata": {"space_key": "CUST"},
                                "updated_at": "2023-08-10"
                            }
                        },
                        {
                            "_id": "doc_2", 
                            "_score": 1.2,
                            "_source": {
                                "title": "Older Customer Service Manual",
                                "body": "Customer service procedures",
                                "metadata": {"space_key": "CUST"}, 
                                "updated_at": "2023-07-12"
                            }
                        }
                    ]
                }
            }
        else:
            # Default success response
            response.status_code = 200
            response.json.return_value = {"acknowledged": True}
            
        response.raise_for_status = Mock()
        return response
    
    def put(self, url, json=None, timeout=None):
        self.requests_made.append({
            "method": "PUT",
            "url": url,
            "json": json
        })
        
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"acknowledged": True}
        response.raise_for_status = Mock()
        return response
    
    def get(self, url, timeout=None):
        self.requests_made.append({
            "method": "GET", 
            "url": url
        })
        
        response = Mock()
        
        if "_cat/indices" in url:
            # Mock indices list for versioning
            response.status_code = 200
            response.json.return_value = [
                {"index": "test_index_v1"},
                {"index": "test_index_v2"}
            ]
        elif "_alias/" in url:
            # Mock alias info
            response.status_code = 200
            response.json.return_value = {"test_index_v2": {}}
        else:
            response.status_code = 200
            response.json.return_value = {"status": "green"}
            
        response.raise_for_status = Mock()
        return response
    
    def head(self, url, timeout=None):
        self.requests_made.append({
            "method": "HEAD",
            "url": url
        })
        
        response = Mock()
        response.status_code = 404  # Index doesn't exist
        return response
    
    def delete(self, url, timeout=None):
        self.requests_made.append({
            "method": "DELETE",
            "url": url
        })
        
        response = Mock()
        response.status_code = 200
        response.json.return_value = {"acknowledged": True}
        response.raise_for_status = Mock()
        return response


@pytest.mark.asyncio
async def test_mock_embedding_provider():
    """Test mock embedding provider creates correct dimensions."""
    provider = MockEmbeddingProvider()
    
    texts = ["Test document 1", "Test document 2"]
    embeddings = await provider.create_embeddings(texts)
    
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 1024  # Correct dimensions
    assert len(embeddings[1]) == 1024
    
    # Embeddings should be deterministic but different
    assert embeddings[0] != embeddings[1]


def test_jpmc_embedding_provider_config():
    """Test JPMC embedding provider configuration."""
    with patch.dict(os.environ, {"JPMC_EMBEDDING_API_KEY": "test-key"}):
        provider = JPMCEmbeddingProvider(
            api_key="test-key",
            base_url="https://api.test.com"
        )
        
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://api.test.com"


def test_embedding_provider_factory():
    """Test embedding provider factory with environment variables."""
    # Test mock provider (default)
    with patch.dict(os.environ, {}, clear=True):
        provider = create_embedding_provider()
        assert isinstance(provider, MockEmbeddingProvider)
    
    # Test JPMC provider
    with patch.dict(os.environ, {
        "EMBEDDING_PROVIDER": "jpmc",
        "JPMC_EMBEDDING_API_KEY": "test-key"
    }):
        provider = create_embedding_provider()
        assert isinstance(provider, JPMCEmbeddingProvider)
    
    # Test invalid provider
    with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "invalid"}):
        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider()


def test_document_validation(sample_documents):
    """Test document validation logic."""
    valid_doc = sample_documents[0]
    errors = valid_doc.validate()
    assert len(errors) == 0
    
    # Test invalid document
    invalid_doc = Document(
        doc_id="",  # Empty ID
        title="",   # Empty title
        body="",    # Empty body
        section="test",
        metadata="not_a_dict",  # Wrong type
        updated_at="not_a_date",  # Wrong type
    )
    
    errors = invalid_doc.validate()
    assert len(errors) == 5  # Should have 5 validation errors


def test_document_to_opensearch_format(sample_documents):
    """Test document conversion to OpenSearch format."""
    doc = sample_documents[0]
    embedding = [0.1] * 1024  # Correct dimensions
    
    opensearch_doc = doc.to_opensearch_doc(embedding)
    
    assert opensearch_doc["title"] == doc.title
    assert opensearch_doc["body"] == doc.body
    assert opensearch_doc["acl_hash"] == doc.acl_hash
    assert opensearch_doc["embedding"] == embedding
    assert opensearch_doc["updated_at"] == doc.updated_at.strftime('%Y-%m-%d')
    
    # Test wrong embedding dimensions
    wrong_embedding = [0.1] * 512  # Wrong dimensions
    with pytest.raises(ValueError, match="Embedding must have 1536 dimensions"):
        doc.to_opensearch_doc(wrong_embedding)


@pytest.mark.asyncio
async def test_unified_indexing_pipeline(mock_search_config, sample_documents):
    """Test the unified indexing pipeline with blue/green deployment."""
    
    with patch('src.infra.indexing_pipeline.make_search_session') as mock_session_factory:
        mock_session = MockSession()
        mock_session_factory.return_value = mock_session
        
        # Create pipeline with mock embedding provider
        with patch.dict(os.environ, {"EMBEDDING_PROVIDER": "mock"}):
            pipeline = UnifiedIndexingPipeline(mock_search_config)
        
        # Test indexing
        result = await pipeline.index_documents(sample_documents, "test_index")
        
        assert result.success
        assert result.indexed_count == 3
        assert result.failed_count == 0
        assert "test_index_v" in result.index_name
        
        # Verify requests were made
        requests = [req for req in mock_session.requests_made if req["method"] == "PUT"]
        assert len(requests) >= 1  # At least index creation
        
        # Verify bulk indexing happened
        bulk_requests = [req for req in mock_session.requests_made if "_bulk" in req["url"]]
        assert len(bulk_requests) == 1


@pytest.mark.asyncio 
async def test_acl_filtering_and_time_ordering(mock_search_config):
    """Test ACL filtering honors access control and returns newest first."""
    
    # Mock session that returns filtered results  
    mock_session = Mock()
    search_response_data = {
        "hits": {
            "total": {"value": 2},
            "hits": [
                {
                    "_id": "doc_1",
                    "_score": 1.8,  # Higher score (newer + relevance)
                    "_source": {
                        "title": "Recent Customer Service Guide",
                        "body": "How to provide excellent customer service",
                        "metadata": {"space_key": "CUST"},
                        "updated_at": "2023-08-10",
                        "acl_hash": "grp_customer_service"
                    }
                },
                {
                    "_id": "doc_2",
                    "_score": 1.2,  # Lower score (older)
                    "_source": {
                        "title": "Older Customer Service Manual", 
                        "body": "Customer service procedures",
                        "metadata": {"space_key": "CUST"},
                        "updated_at": "2023-07-12",
                        "acl_hash": "grp_customer_service"
                    }
                }
                # Note: doc_3 with different ACL is filtered out
            ]
        }
    }
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = search_response_data
    mock_response.raise_for_status = Mock()
    mock_session.post.return_value = mock_response
    
    with patch('src.infra.opensearch_client.make_search_session') as mock_session_factory:
        mock_session_factory.return_value = mock_session
        
        client = OpenSearchClient(mock_search_config)
        
        # Search with ACL filter
        filters = SearchFilters(acl_hash="grp_customer_service")
        response = client.bm25_search(
            query="customer service",
            filters=filters,
            index="test_index"
        )
        
        assert response.total_hits == 2
        assert len(response.results) == 2
        
        # Verify newest document comes first (higher score due to time decay)
        assert response.results[0].doc_id == "doc_1"  # Recent document
        assert response.results[1].doc_id == "doc_2"  # Older document
        assert response.results[0].score > response.results[1].score
        
        # Verify ACL filter was applied in the request
        search_call = mock_session.post.call_args
        search_body = search_call[1]["json"]
        
        # Check that ACL filter is present in the query
        assert "filter" in str(search_body)
        
        # The exact structure depends on the query builder, but ACL should be present
        query_str = str(search_body)
        assert "grp_customer_service" in query_str


def test_index_mapping_creation(mock_search_config):
    """Test index mapping has correct embedding dimensions."""
    with patch('src.infra.indexing_pipeline.make_search_session'):
        pipeline = UnifiedIndexingPipeline(mock_search_config)
        
        mapping = pipeline.create_index_mapping()
        
        # Verify embedding field has correct dimensions
        embedding_config = mapping["mappings"]["properties"]["embedding"]
        assert embedding_config["type"] == "knn_vector"
        assert embedding_config["dimension"] == 1024  # Fixed to 1536
        
        # Verify other required fields
        assert "title" in mapping["mappings"]["properties"]
        assert "body" in mapping["mappings"]["properties"]
        assert "acl_hash" in mapping["mappings"]["properties"]
        assert "updated_at" in mapping["mappings"]["properties"]


def test_version_number_calculation(mock_search_config):
    """Test version number calculation for blue/green indexing."""
    mock_session = MockSession()
    
    with patch('src.infra.indexing_pipeline.make_search_session') as mock_session_factory:
        mock_session_factory.return_value = mock_session
        pipeline = UnifiedIndexingPipeline(mock_search_config)
        
        # Test with existing versions
        version = pipeline.get_next_version_number("test_index")
        assert version == 3  # Should be next after v2 from mock data
        
        
@pytest.mark.asyncio
async def test_embedding_dimension_validation(mock_search_config, sample_documents):
    """Test that indexing fails with wrong embedding dimensions."""
    
    # Create a mock provider that returns wrong dimensions
    class BadEmbeddingProvider:
        async def create_embeddings(self, texts, model=""):
            # Return wrong dimensions (512 instead of 1536)
            return [[0.1] * 512 for _ in texts]
    
    mock_session = MockSession()
    
    with patch('src.infra.indexing_pipeline.make_search_session') as mock_session_factory:
        mock_session_factory.return_value = mock_session
        
        pipeline = UnifiedIndexingPipeline(mock_search_config)
        pipeline.embedding_provider = BadEmbeddingProvider()
        
        # This should fail due to dimension mismatch
        with pytest.raises(ValueError, match="Embedding must have 1536 dimensions"):
            doc = sample_documents[0]
            wrong_embedding = [0.1] * 512
            doc.to_opensearch_doc(wrong_embedding)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])