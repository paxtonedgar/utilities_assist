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
    field_path: (
        str  # OpenSearch field path (e.g., "content_type", "metadata.space_key")
    )
    description: str


class OpenSearchConfig:
    """Centralized OpenSearch configuration."""

    # === INDEX DEFINITIONS ===
    # Khub cluster embedding dimensions (confirmed from colleague's mapping)
    EMBEDDING_DIMENSIONS = 1536

    MAIN_INDEX = IndexConfig(
        name="khub-opensearch-index",
        # Content fields: Using nested structure as found in v1-working-code
        content_fields=["sections.content"],
        # Metadata fields: From mapping + fields actively used in codebase
        metadata_fields=[
            # From actual mapping
            "title",
            "updated_at",
            "page_id",
            "canonical_id",
            "acl_hash",
            "content_type",
            "source",
            "section_anchor",
            # From codebase usage (may be added during indexing)
            "api_name",
            "utility_name",
            "page_url",
            "path",
            "space",
            "url",
            "app_name",
            # From metadata object structure
            "author",
            "space_key",
            "version",
            "labels",
        ],
        vector_field="sections.embedding",
        # Title fields: Fields used to extract document titles
        title_fields=["title", "api_name", "utility_name", "app_name"],
    )

    SWAGGER_INDEX = IndexConfig(
        name="khub-opensearch-swagger-index",
        content_fields=[
            "sections.content"
        ],  # Swagger also uses nested sections structure
        metadata_fields=[
            "title",
            "app_name",
            "utility_name",
            "api_name",
            "page_url",
            "path",
            "method",
            "endpoint",
        ],
        vector_field="sections.embedding",  # ✅ FIXED: Swagger index DOES have vector search - logs show KNN working
        title_fields=["title", "app_name", "api_name"],
    )

    # === SEARCH STRATEGIES ===
    STRATEGIES = {
        "hybrid": SearchStrategy("hybrid", "BM25 + KNN hybrid search", True, 3.0),
        "enhanced_rrf": SearchStrategy(
            "enhanced_rrf", "RRF fusion of BM25 and KNN", True, 2.5
        ),
        "bm25": SearchStrategy("bm25", "Pure BM25 text search", False, 2.0),
        "knn": SearchStrategy("knn", "Pure vector similarity search", True, 2.0),
    }

    # === FILTER CONFIGURATIONS ===
    FILTERS = {
        "content_type": FilterConfig(
            "content_type", "content_type", "Filter by document content type"
        ),
        "acl_hash": FilterConfig(
            "acl_hash", "acl_hash", "Filter by access control hash"
        ),
        "space_key": FilterConfig(
            "space_key", "metadata.space_key", "Filter by Confluence space"
        ),
        "updated_after": FilterConfig(
            "updated_after", "updated_at", "Filter by minimum update date"
        ),
        "updated_before": FilterConfig(
            "updated_before", "updated_at", "Filter by maximum update date"
        ),
    }

    # === INTENT-BASED FILTER VALUES ===
    INTENT_FILTERS = {
        "confluence": {"content_type": "confluence"},
        "swagger": {"content_type": "api_spec"},
        "api_spec": {"content_type": "api_spec"},  # Alias for swagger
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
    def get_filter_config(cls, filter_name: str) -> Optional[FilterConfig]:
        """Get filter configuration by name."""
        return cls.FILTERS.get(filter_name)

    @classmethod
    def get_intent_filters(cls, intent_type: str) -> Dict[str, Any]:
        """Get default filters for an intent type."""
        return cls.INTENT_FILTERS.get(intent_type, {}).copy()

    @classmethod
    def build_filter_clause(
        cls, filter_name: str, value: Any
    ) -> Optional[Dict[str, Any]]:
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
    def build_bm25_query(
        text_query: str, index_name: str, k: int = 10
    ) -> Dict[str, Any]:
        """Build a BM25-only search query."""
        config = OpenSearchConfig._get_index_config(index_name)

        # Build content query - use nested with inner_hits if content fields have nested structure
        if any("sections." in field for field in config.content_fields):
            # For utility queries, search for both acronym and full phrase
            # This helps find docs that use "CIU" even when query is "Customer Interaction Utility"
            query_parts = []

            # Check if this looks like an expanded utility name
            utility_acronyms = {
                "Customer Interaction Utility": "CIU",
                "Enhanced Transaction Utility": "ETU",
                "Customer Summary Utility": "CSU",
                "Account Utility": "AU",
                "Product Catalog Utility": "PCU",
            }

            # Add original query
            query_parts.append(
                {"match": {"sections.content": {"query": text_query, "boost": 1.0}}}
            )

            # If query contains a utility name, also search for its acronym
            for full_name, acronym in utility_acronyms.items():
                if full_name.lower() in text_query.lower():
                    # Add acronym search with high boost
                    query_parts.append(
                        {
                            "match": {
                                "sections.content": {"query": acronym, "boost": 2.0}
                            }
                        }
                    )
                    # Add fuzzy match for variations
                    query_parts.append(
                        {
                            "match": {
                                "sections.content": {
                                    "query": f"{acronym} utility",
                                    "boost": 1.5,
                                }
                            }
                        }
                    )
                    # Add common documentation patterns
                    query_parts.append(
                        {
                            "match": {
                                "sections.content": {
                                    "query": f"{acronym} data",
                                    "boost": 1.2,
                                }
                            }
                        }
                    )
                    query_parts.append(
                        {
                            "match": {
                                "sections.content": {
                                    "query": f"{acronym} interaction",
                                    "boost": 1.2,
                                }
                            }
                        }
                    )
                    query_parts.append(
                        {
                            "match": {
                                "sections.content": {
                                    "query": f"{acronym} field",
                                    "boost": 1.1,
                                }
                            }
                        }
                    )

            # Use bool query with should clauses if we have multiple parts
            if len(query_parts) > 1:
                nested_query = {
                    "bool": {"should": query_parts, "minimum_should_match": 1}
                }
            else:
                nested_query = query_parts[0]

            query_clause = {
                "nested": {
                    "path": "sections",
                    "query": nested_query,
                    "inner_hits": {
                        "name": "matched_sections",
                        "size": 5,
                        "highlight": {"fields": {"sections.content": {}}},
                        "sort": [{"_score": "desc"}],
                    },
                }
            }
        else:
            # Use regular multi_match for flat fields
            query_clause = {
                "multi_match": {
                    "query": text_query,
                    "fields": [f"{field}^2" for field in config.content_fields]
                    + [f"{field}^6" for field in config.title_fields],
                    "type": "best_fields",
                }
            }

        return {
            "size": k,
            "query": query_clause,
            "_source": ["api_name", "page_url", "title", "utility_name"]
            + OpenSearchConfig.get_source_fields(index_name),
        }

    @staticmethod
    def build_knn_query(
        vector_query: List[float], index_name: str, k: int = 10
    ) -> Dict[str, Any]:
        """Build a KNN-only search query."""
        config = OpenSearchConfig._get_index_config(index_name)

        # Build vector query - only if index supports vectors
        if not config.vector_field:
            # Index doesn't support vector search - return None to indicate KNN not available
            return None

        if "sections." in config.vector_field:
            # Use nested with native knn query for sections.embedding (main index structure)
            query_clause = {
                "nested": {
                    "path": "sections",
                    "query": {
                        "knn": {"sections.embedding": {"vector": vector_query, "k": k}}
                    },
                    "inner_hits": {
                        "name": "matched_sections",
                        "size": 5,
                        "sort": [{"_score": "desc"}],
                    },
                }
            }
        else:
            # Use native knn for flat fields (if properly mapped)
            query_clause = {
                "knn": {config.vector_field: {"vector": vector_query, "k": min(k, 20)}}
            }

        return {
            "size": k,
            "query": query_clause,
            "_source": ["api_name", "page_url", "title", "utility_name"]
            + OpenSearchConfig.get_source_fields(index_name),
            "track_scores": True,
        }


# === CONVENIENCE FUNCTIONS ===




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
