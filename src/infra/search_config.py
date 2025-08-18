"""
Single source of truth for all OpenSearch configurations.

This centralizes index names, field mappings, query configurations, and search strategies
to eliminate scattered configuration across the codebase.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class IndexConfig:
    """Configuration for a specific OpenSearch index."""
    name: str
    content_fields: List[str]  # Fields that contain document content
    metadata_fields: List[str]  # Fields for metadata/filtering
    vector_field: str  # Field for vector/embedding searches
    title_fields: List[str]  # Fields to use for document titles


@dataclass 
class SearchStrategy:
    """Configuration for search strategies."""
    name: str
    description: str
    uses_vector: bool
    timeout_seconds: float


@dataclass 
class FilterConfig:
    """Configuration for search filters."""
    name: str
    field_path: str  # OpenSearch field path (e.g., "content_type", "metadata.space_key")
    description: str


class OpenSearchConfig:
    """Centralized OpenSearch configuration."""
    
    # === INDEX DEFINITIONS ===
    # Khub cluster embedding dimensions (reduced from 1536/3072 using dimensions parameter)
    EMBEDDING_DIMENSIONS = 1024
    
    MAIN_INDEX = IndexConfig(
        name="khub-opensearch-index",
        # Content fields: From mapping + commonly expected fields
        content_fields=["body", "content", "text", "description", "section"],
        # Metadata fields: From mapping + fields actively used in codebase  
        metadata_fields=[
            # From actual mapping
            "title", "updated_at", "page_id", "canonical_id", "acl_hash", "content_type", "source", "section_anchor",
            # From codebase usage (may be added during indexing)
            "api_name", "utility_name", "page_url", "path", "space", "url", "app_name",
            # From metadata object structure
            "author", "space_key", "version", "labels"
        ],
        vector_field="embedding",
        # Title fields: Fields used to extract document titles
        title_fields=["title", "api_name", "utility_name", "app_name"]
    )
    
    SWAGGER_INDEX = IndexConfig(
        name="khub-opensearch-swagger-index", 
        content_fields=["body", "content", "text", "description", "summary"],
        metadata_fields=["title", "app_name", "utility_name", "api_name", "page_url", "path", "method", "endpoint"],
        vector_field="embedding",
        title_fields=["title", "app_name", "api_name"]
    )
    
    # === SEARCH STRATEGIES ===
    STRATEGIES = {
        "hybrid": SearchStrategy("hybrid", "BM25 + KNN hybrid search", True, 3.0),
        "enhanced_rrf": SearchStrategy("enhanced_rrf", "RRF fusion of BM25 and KNN", True, 2.5),
        "bm25": SearchStrategy("bm25", "Pure BM25 text search", False, 2.0),
        "knn": SearchStrategy("knn", "Pure vector similarity search", True, 2.0)
    }
    
    # === FILTER CONFIGURATIONS ===
    FILTERS = {
        "content_type": FilterConfig("content_type", "content_type", "Filter by document content type"),
        "acl_hash": FilterConfig("acl_hash", "acl_hash", "Filter by access control hash"),
        "space_key": FilterConfig("space_key", "metadata.space_key", "Filter by Confluence space"),
        "updated_after": FilterConfig("updated_after", "updated_at", "Filter by minimum update date"),
        "updated_before": FilterConfig("updated_before", "updated_at", "Filter by maximum update date")
    }
    
    # === INTENT-BASED FILTER VALUES ===
    INTENT_FILTERS = {
        "confluence": {"content_type": "confluence"},
        "swagger": {"content_type": "api_spec"},
        "api_spec": {"content_type": "api_spec"}  # Alias for swagger
    }
    
    # === FIELD CONFIGURATIONS ===
    
    @classmethod
    def get_source_fields(cls, index_name: str) -> List[str]:
        """Get all fields to request in _source for an index."""
        index_config = cls._get_index_config(index_name)
        return index_config.content_fields + index_config.metadata_fields
    
    @classmethod
    def get_content_fields(cls, index_name: str) -> List[str]:
        """Get content fields for an index (for content extraction)."""
        index_config = cls._get_index_config(index_name)
        return index_config.content_fields
    
    @classmethod
    def get_vector_field(cls, index_name: str) -> str:
        """Get vector field name for an index."""
        index_config = cls._get_index_config(index_name)
        return index_config.vector_field
    
    @classmethod
    def get_title_fields(cls, index_name: str) -> List[str]:
        """Get title fields for an index."""
        index_config = cls._get_index_config(index_name)
        return index_config.title_fields
    
    @classmethod
    def get_search_strategy(cls, strategy_name: str) -> SearchStrategy:
        """Get search strategy configuration."""
        return cls.STRATEGIES.get(strategy_name, cls.STRATEGIES["enhanced_rrf"])
    
    @classmethod
    def get_filter_config(cls, filter_name: str) -> Optional[FilterConfig]:
        """Get filter configuration by name."""
        return cls.FILTERS.get(filter_name)
    
    @classmethod
    def get_intent_filters(cls, intent_type: str) -> Dict[str, Any]:
        """Get default filters for an intent type."""
        return cls.INTENT_FILTERS.get(intent_type, {}).copy()
    
    @classmethod
    def build_filter_clause(cls, filter_name: str, value: Any) -> Optional[Dict[str, Any]]:
        """Build a single filter clause for OpenSearch."""
        filter_config = cls.get_filter_config(filter_name)
        if not filter_config:
            return None
            
        if filter_name in ["updated_after", "updated_before"]:
            # Handle date range filters specially
            return None  # Handled by existing range filter logic
        else:
            # Standard term filter
            return {"term": {filter_config.field_path: value}}
    
    @classmethod
    def get_default_index(cls) -> str:
        """Get default index name."""
        return cls.MAIN_INDEX.name
    
    @classmethod
    def get_swagger_index(cls) -> str:
        """Get Swagger API index name.""" 
        return cls.SWAGGER_INDEX.name
    
    @classmethod
    def _get_index_config(cls, index_name: str) -> IndexConfig:
        """Get index configuration by name."""
        if index_name == cls.MAIN_INDEX.name:
            return cls.MAIN_INDEX
        elif index_name == cls.SWAGGER_INDEX.name:
            return cls.SWAGGER_INDEX
        else:
            # Default to main index config for unknown indices
            return cls.MAIN_INDEX


# === QUERY TEMPLATES ===

class QueryTemplates:
    """Reusable query templates for different search types."""
    
    @staticmethod
    def build_hybrid_query(
        text_query: str,
        vector_query: List[float],
        index_name: str,
        k: int = 10
    ) -> Dict[str, Any]:
        """Build a hybrid search query with nested structure for kNN."""
        config = OpenSearchConfig._get_index_config(index_name)
        
        return {
            "size": min(k, 20),  # Limit size to reduce over-fetching
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": text_query,
                                "fields": [f"{field}^3" for field in config.content_fields] + 
                                         [f"{field}^4" for field in config.title_fields],
                                "type": "best_fields"
                            }
                        },
                        {
                            "knn": {
                                config.vector_field: {
                                    "vector": vector_query,
                                    "k": min(k, 20)  # Limit to reduce payload
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": OpenSearchConfig.get_source_fields(index_name),
            "track_total_hits": True,  # Enable for proper error handling
            "highlight": {
                "fields": {field: {} for field in config.content_fields},
                "fragment_size": 160,
                "number_of_fragments": 2
            }
        }
    
    @staticmethod 
    def build_bm25_query(
        text_query: str,
        index_name: str,
        k: int = 10
    ) -> Dict[str, Any]:
        """Build a BM25-only search query."""
        config = OpenSearchConfig._get_index_config(index_name)
        
        return {
            "size": k,
            "query": {
                "multi_match": {
                    "query": text_query,
                    "fields": [f"{field}^2" for field in config.content_fields] + 
                             [f"{field}^6" for field in config.title_fields],
                    "type": "best_fields"
                }
            },
            "_source": OpenSearchConfig.get_source_fields(index_name),
            "highlight": {
                "fields": {field: {} for field in config.content_fields},
                "fragment_size": 160,
                "number_of_fragments": 2
            }
        }
    
    @staticmethod
    def build_knn_query(
        vector_query: List[float],
        index_name: str,
        k: int = 10
    ) -> Dict[str, Any]:
        """Build a KNN-only search query."""
        config = OpenSearchConfig._get_index_config(index_name)
        
        return {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": f"cosineSimilarity(params.query_vector, '{config.vector_field}') + 1.0",
                        "params": {
                            "query_vector": vector_query
                        }
                    }
                }
            },
            "_source": OpenSearchConfig.get_source_fields(index_name),
            "track_scores": True
        }


# === CONVENIENCE FUNCTIONS ===

def get_main_index() -> str:
    """Get main index name."""
    return OpenSearchConfig.get_default_index()

def get_swagger_index() -> str:
    """Get Swagger index name."""
    return OpenSearchConfig.get_swagger_index()

def get_content_fields(index_name: str) -> List[str]:
    """Get content fields for content extraction."""
    return OpenSearchConfig.get_content_fields(index_name)

def get_source_fields(index_name: str) -> List[str]:
    """Get all _source fields to request."""
    return OpenSearchConfig.get_source_fields(index_name)

def get_intent_filters(intent_type: str) -> Dict[str, Any]:
    """Get default filters for an intent type."""
    return OpenSearchConfig.get_intent_filters(intent_type)

def build_filter_clause(filter_name: str, value: Any) -> Optional[Dict[str, Any]]:
    """Build a single filter clause for OpenSearch."""
    return OpenSearchConfig.build_filter_clause(filter_name, value)