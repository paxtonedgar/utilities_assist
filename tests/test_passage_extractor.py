"""
Tests for passage extractor module.

Covers the core A1+A2 extraction logic with various OpenSearch hit formats.
"""

from src.services.passage_extractor import (
    extract_passages,
    is_metadata_only_swagger,
    extract_passages_batch,
)
from src.services.models import ExtractorConfig, RankedHit


class TestPassageExtractor:
    """Test core passage extraction functionality."""
    
    def test_extract_from_inner_hits_success(self):
        """Test successful extraction from inner_hits structure."""
        hit = {
            "_id": "doc123",
            "_index": "main-index",
            "_score": 0.8,
            "_source": {
                "title": "Test Document",
                "api_name": "TestAPI",
                "page_url": "https://example.com/doc123"
            },
            "inner_hits": {
                "matched_sections": {
                    "hits": {
                        "hits": [
                            {
                                "_score": 0.9,
                                "_source": {
                                    "content": "This is test content from inner hits",
                                    "title": "Section 1"
                                }
                            },
                            {
                                "_score": 0.7,
                                "_source": {
                                    "content": "More test content here",
                                    "heading": "Section 2"
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        cfg = ExtractorConfig()
        passages = extract_passages(hit, cfg)
        
        assert len(passages) == 2
        assert passages[0].text == "This is test content from inner hits"
        assert passages[0].section_title == "Section 1"
        assert passages[0].score == 0.9
        assert passages[0].doc_id == "doc123"
        assert passages[0].title == "Test Document"
        assert passages[0].api_name == "TestAPI"
        
        assert passages[1].text == "More test content here"
        assert passages[1].section_title == "Section 2"
    
    def test_extract_from_sections_array_fallback(self):
        """Test fallback to sections array when no inner_hits."""
        hit = {
            "_id": "doc456",
            "_index": "main-index", 
            "_score": 0.6,
            "_source": {
                "title": "Document with Sections",
                "sections": [
                    {
                        "content": "Content from sections array",
                        "title": "Array Section 1"
                    },
                    {
                        "text": "Alternative text field",
                        "heading": "Array Section 2"
                    }
                ]
            }
        }
        
        cfg = ExtractorConfig()
        passages = extract_passages(hit, cfg)
        
        assert len(passages) == 2
        assert passages[0].text == "Content from sections array"
        assert passages[0].section_title == "Array Section 1"
        assert passages[1].text == "Alternative text field"
        assert passages[1].section_title == "Array Section 2"
    
    def test_extract_from_doc_level_fields(self):
        """Test extraction from document-level fields as final fallback."""
        hit = {
            "_id": "doc789",
            "_index": "swagger-index",
            "_score": 0.5,
            "_source": {
                "title": "API Documentation",
                "body": "This is the main document content",
                "api_name": "SwaggerAPI"
            }
        }
        
        cfg = ExtractorConfig()
        passages = extract_passages(hit, cfg)
        
        assert len(passages) == 1
        assert passages[0].text == "This is the main document content"
        assert passages[0].section_title is None
        assert passages[0].title == "API Documentation"
        assert passages[0].api_name == "SwaggerAPI"
    
    def test_no_content_found(self):
        """Test handling when no extractable content is found."""
        hit = {
            "_id": "empty_doc",
            "_index": "main-index",
            "_score": 0.3,
            "_source": {
                "title": "Empty Document",
                "api_name": "EmptyAPI",
                # No content fields
            }
        }
        
        cfg = ExtractorConfig()
        passages = extract_passages(hit, cfg)
        
        assert len(passages) == 0
    
    def test_length_filtering(self):
        """Test that passages are filtered by length constraints."""
        hit = {
            "_id": "length_test",
            "_index": "main-index",
            "_score": 0.7,
            "_source": {
                "title": "Length Test",
                "sections": [
                    {
                        "content": "x" * 50,  # Too short (< 80 chars)
                        "title": "Too Short"
                    },
                    {
                        "content": "x" * 200,  # Just right
                        "title": "Good Length"
                    },
                    {
                        "content": "x" * 2000,  # Too long (> 1200 chars)
                        "title": "Too Long"
                    }
                ]
            }
        }
        
        cfg = ExtractorConfig(min_chars=80, max_chars=1200)
        passages = extract_passages(hit, cfg)
        
        # Only the middle one should pass
        assert len(passages) == 1
        assert passages[0].section_title == "Good Length"
        assert len(passages[0].text) == 200
    
    def test_swagger_metadata_only_detection(self):
        """Test A2 Swagger metadata-only filtering."""
        # Metadata-only Swagger hit
        metadata_hit = RankedHit(
            hit={
                "_id": "swagger_meta",
                "_source": {
                    "api_name": "TestAPI",
                    "utility_name": "TestUtil",
                    "sections": []  # Empty sections
                }
            },
            passages=[],  # No passages extracted
            rank_rrf=0,
            index="test-swagger-index"
        )
        
        # Swagger hit with content
        content_hit = RankedHit(
            hit={
                "_id": "swagger_content",
                "_source": {
                    "api_name": "TestAPI",
                    "body": "Real API documentation content"
                }
            },
            passages=[],  # Would have passages if extracted
            rank_rrf=1,
            index="test-swagger-index"
        )
        
        # Non-Swagger hit
        regular_hit = RankedHit(
            hit={
                "_id": "regular_doc",
                "_source": {"api_name": "TestAPI"}
            },
            passages=[],
            rank_rrf=2,
            index="main-index"
        )
        
        cfg = ExtractorConfig(swagger_suffix="-swagger-index")
        
        # Metadata-only Swagger should be filtered
        assert is_metadata_only_swagger(metadata_hit, cfg) is True
        
        # Content Swagger should not be filtered  
        assert is_metadata_only_swagger(content_hit, cfg) is False
        
        # Non-Swagger should not be filtered
        assert is_metadata_only_swagger(regular_hit, cfg) is False
    
    def test_batch_extraction(self):
        """Test batch processing of multiple hits."""
        hits = [
            {
                "_id": "batch1",
                "_index": "main-index",
                "_score": 0.8,
                "_source": {
                    "title": "Batch Doc 1",
                    "body": "Content for first batch document"
                }
            },
            {
                "_id": "batch2", 
                "_index": "main-index",
                "_score": 0.6,
                "_source": {
                    "title": "Batch Doc 2",
                    "body": "Content for second batch document"
                }
            }
        ]
        
        cfg = ExtractorConfig()
        ranked_hits = extract_passages_batch(hits, cfg)
        
        assert len(ranked_hits) == 2
        assert ranked_hits[0].rank_rrf == 0
        assert ranked_hits[1].rank_rrf == 1
        assert len(ranked_hits[0].passages) == 1
        assert len(ranked_hits[1].passages) == 1
        assert ranked_hits[0].passages[0].text == "Content for first batch document"
    
    def test_field_order_preferences(self):
        """Test that field order preferences are respected."""
        hit = {
            "_id": "field_order_test",
            "_index": "main-index",
            "_score": 0.7,
            "_source": {
                "title": "Field Order Test",
                "description": "Should be ignored",
                "content": "Should be used",  # Higher priority
                "text": "Should also be ignored"
            }
        }
        
        cfg = ExtractorConfig(doc_field_order=['content', 'text', 'description'])
        passages = extract_passages(hit, cfg)
        
        assert len(passages) == 1
        assert passages[0].text == "Should be used"
    
    def test_whitespace_normalization(self):
        """Test that whitespace is properly normalized."""
        hit = {
            "_id": "whitespace_test",
            "_index": "main-index", 
            "_score": 0.5,
            "_source": {
                "title": "Whitespace Test",
                "body": "  Multiple   spaces\n\nand\t\ttabs  should  be  normalized  "
            }
        }
        
        cfg = ExtractorConfig()
        passages = extract_passages(hit, cfg)
        
        assert len(passages) == 1
        # Multiple spaces/tabs/newlines should be normalized to single spaces
        assert passages[0].text == "Multiple spaces and tabs should be normalized"


class TestConfigurationOptions:
    """Test different configuration scenarios."""
    
    def test_custom_field_orders(self):
        """Test custom field order configuration."""
        cfg = ExtractorConfig(
            section_field_order=['text', 'content'],  # Prefer text over content
            doc_field_order=['summary', 'body', 'content'],
            max_sections=2,
            min_chars=50,
            max_chars=500
        )
        
        hit = {
            "_id": "config_test",
            "_index": "test-index",
            "_score": 0.6,
            "_source": {
                "sections": [
                    {
                        "content": "Should be ignored",
                        "text": "Should be used",  # Higher priority
                        "title": "Test Section"
                    }
                ]
            }
        }
        
        passages = extract_passages(hit, cfg)
        assert len(passages) == 1
        assert passages[0].text == "Should be used"
    
    def test_swagger_suffix_configuration(self):
        """Test custom Swagger suffix configuration."""
        cfg = ExtractorConfig(swagger_suffix="-api-docs")
        
        hit = RankedHit(
            hit={"_id": "test", "_source": {"api_name": "Test"}},
            passages=[],
            rank_rrf=0,
            index="test-api-docs"  # Matches custom suffix
        )
        
        assert is_metadata_only_swagger(hit, cfg) is True