"""
Query Structure Lock - VERIFIED WORKING CONFIGURATION

This file preserves the exact working query structure discovered and validated on 2025-08-19.
DO NOT MODIFY these values without updating QUERY_STRUCTURE_SOURCE_OF_TRUTH.md

Status: PRODUCTION VERIFIED ✅
- Successfully resolves "embedding field not recognized as k-NN vector type"
- Successfully returns search results (hits=8 confirmed)
- Successfully extracts section-level content via inner_hits
- No script compilation errors
"""

# === VERIFIED WORKING CONFIGURATION ===

VERIFIED_WORKING_CONFIG = {
    # Index Structure
    "main_index_name": "khub-opensearch-index",
    "nested_path": "sections",
    
    # Field Mappings (CRITICAL - DO NOT CHANGE)
    "content_field": "sections.content",  # NOT "content"
    "vector_field": "sections.embedding", # NOT "embedding" 
    "embedding_dimensions": 1536,         # NOT 1024
    
    # Query Structure
    "uses_nested_queries": True,          # REQUIRED for sections
    "uses_native_knn": True,             # NOT script_score
    "uses_inner_hits": True,             # REQUIRED for section extraction
    "inner_hits_name": "matched_sections",
    "inner_hits_size": 5,
    
    # Response Processing  
    "extracts_from_inner_hits": True,    # Primary content source
    "falls_back_to_source": True,        # Backup content source
    
    # Query Templates
    "uses_centralized_templates": True,   # QueryTemplates.build_*()
    "template_methods": [
        "QueryTemplates.build_bm25_query",
        "QueryTemplates.build_knn_query", 
        "QueryTemplates.build_hybrid_query"
    ]
}

# === WORKING QUERY SIGNATURES ===

WORKING_BM25_STRUCTURE = {
    "query": {
        "nested": {
            "path": "sections",
            "query": {"match": {"sections.content": {"query": "SEARCH_TERMS"}}},
            "inner_hits": {
                "name": "matched_sections",
                "size": 5,
                "highlight": {"fields": {"sections.content": {}}},
                "sort": [{"_score": "desc"}]
            }
        }
    }
}

WORKING_KNN_STRUCTURE = {
    "query": {
        "nested": {
            "path": "sections", 
            "query": {
                "knn": {
                    "sections.embedding": {
                        "vector": "VECTOR_1536_DIMS",
                        "k": 10
                    }
                }
            },
            "inner_hits": {
                "name": "matched_sections",
                "size": 5,
                "sort": [{"_score": "desc"}]
            }
        }
    }
}

# === ERROR SIGNATURES TO AVOID ===

BROKEN_PATTERNS = {
    "field_errors": [
        "Field 'embedding' is not knn_vector type",  # Use sections.embedding
        "Unknown key for a START_OBJECT in [knn]",   # Use nested wrapper
    ],
    "query_errors": [
        "script_score compilation failed",            # Use native knn
        "No content found in search results",        # Use inner_hits
    ],
    "dimension_errors": [
        "dimension mismatch: got 1024, expected 1536"  # Use 1536 dims
    ]
}

# === VALIDATION FUNCTIONS ===

def validate_query_structure(query_dict: dict, query_type: str) -> bool:
    """Validate query structure against working patterns."""
    if query_type == "bm25":
        return _validate_bm25_structure(query_dict)
    elif query_type == "knn":  
        return _validate_knn_structure(query_dict)
    return False

def _validate_bm25_structure(query: dict) -> bool:
    """Validate BM25 query has correct nested structure."""
    try:
        nested = query["query"]["nested"]
        return (
            nested["path"] == "sections" and
            "sections.content" in nested["query"]["match"] and
            nested["inner_hits"]["name"] == "matched_sections"
        )
    except KeyError:
        return False

def _validate_knn_structure(query: dict) -> bool:
    """Validate KNN query has correct nested structure."""
    try:
        nested = query["query"]["nested"]
        knn_query = nested["query"]["knn"]
        return (
            nested["path"] == "sections" and
            "sections.embedding" in knn_query and
            nested["inner_hits"]["name"] == "matched_sections"
        )
    except KeyError:
        return False

# === ROLLBACK COMMANDS ===

ROLLBACK_COMMANDS = [
    "# Rollback to working query structure:",
    "git checkout v1-working-code -- src/infra/search_config.py",
    "# Or restore from this lock file:",
    "# Update OpenSearchConfig.MAIN_INDEX.content_fields = ['sections.content']",
    "# Update OpenSearchConfig.MAIN_INDEX.vector_field = 'sections.embedding'", 
    "# Update OpenSearchConfig.EMBEDDING_DIMENSIONS = 1536"
]

if __name__ == "__main__":
    print("Query Structure Lock File")
    print("=" * 50)
    print(f"Status: {VERIFIED_WORKING_CONFIG}")
    print("\nTo validate current config against working patterns:")
    print("python -c 'from src.infra.query_structure_lock import validate_query_structure'")