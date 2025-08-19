# OpenSearch Query Structure - Source of Truth

**Version**: 1.0  
**Date**: 2025-08-19  
**Status**: PRODUCTION VERIFIED ✅  

> This document serves as the definitive reference for OpenSearch query structure in utilities_assist. All query modifications should reference and update this document.

## 🎯 Executive Summary

Our OpenSearch system uses **nested document structure** with section-level indexing. Documents contain arrays of `sections`, each with `content`, `embedding`, `heading`, and `anchor` fields. Both BM25 and KNN queries operate at the section level using nested queries with `inner_hits`.

## 📊 Index Structure

### Main Index: `khub-opensearch-index`
```json
{
  "title": "Document Title",
  "page_url": "https://...",
  "api_name": "API Name", 
  "utility_name": "Utility Name",
  "sections": [
    {
      "content": "Section text content",
      "embedding": [1536 float values],
      "heading": "Section Heading",
      "anchor": "section-slug",
      "section_path": "Parent > Child > Section"
    }
  ]
}
```

### Key Configuration
- **Embedding Dimensions**: 1536 (OpenAI standard)
- **Content Fields**: `["sections.content"]` 
- **Vector Field**: `"sections.embedding"`
- **Nested Path**: `"sections"`

## 🔍 Query Templates (VERIFIED WORKING)

### 1. BM25 Nested Query
```json
{
  "query": {
    "nested": {
      "path": "sections",
      "query": {
        "match": {
          "sections.content": {"query": "search terms"}
        }
      },
      "inner_hits": {
        "name": "matched_sections",
        "size": 5,
        "highlight": {
          "fields": {"sections.content": {}}
        },
        "sort": [{"_score": "desc"}]
      }
    }
  }
}
```

### 2. KNN Nested Query  
```json
{
  "query": {
    "nested": {
      "path": "sections",
      "query": {
        "knn": {
          "sections.embedding": {
            "vector": [1536 float values],
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
```

### 3. Hybrid Query Structure
Uses `bool.should` with both BM25 and KNN nested queries:
```json
{
  "query": {
    "bool": {
      "should": [
        {/* BM25 nested query */},
        {/* KNN nested query */}
      ],
      "minimum_should_match": 1
    }
  }
}
```

## ⚙️ Centralized Configuration Status

### ✅ FULLY CENTRALIZED
- `_build_simple_bm25_query()` → `QueryTemplates.build_bm25_query()`
- `_build_simple_knn_query()` → `QueryTemplates.build_knn_query()`  
- `_get_boosted_fields()` → Uses `OpenSearchConfig._get_index_config()`
- All field configurations via `OpenSearchConfig`

### ⚠️ PARTIALLY CENTRALIZED  
- `_build_hybrid_query()` - Uses centralized fields but custom query logic
- `_build_bm25_query()` - Uses centralized fields but complex custom logic
- `_build_knn_query()` - Uses centralized fields but custom logic

**Recommendation**: These complex methods should eventually migrate to QueryTemplates for full centralization.

## 🏗️ Source of Truth Preservation Strategy

### 1. Version-Controlled Query Snapshots
```bash
# Save working query structure before changes
git tag -a query-structure-v1.0 -m "Working nested query structure - 2025-08-19"
```

### 2. Reference Implementation Branch
- Keep `v1-working-code` branch as permanent reference
- Never modify - treat as read-only historical reference
- All new implementations should validate against this structure

### 3. Query Template Testing
Create test cases that validate query structure:

```python
# tests/test_query_structure_preservation.py
def test_bm25_nested_structure():
    """Ensure BM25 queries maintain nested structure with inner_hits"""
    query = QueryTemplates.build_bm25_query("test", "khub-opensearch-index", 10)
    
    assert "nested" in query["query"]
    assert query["query"]["nested"]["path"] == "sections"
    assert "inner_hits" in query["query"]["nested"]
    assert query["query"]["nested"]["inner_hits"]["name"] == "matched_sections"

def test_knn_native_structure():
    """Ensure KNN queries use native knn (not script_score)"""
    vector = [0.1] * 1536
    query = QueryTemplates.build_knn_query(vector, "khub-opensearch-index", 10)
    
    assert "nested" in query["query"] 
    assert "knn" in query["query"]["nested"]["query"]
    assert "sections.embedding" in query["query"]["nested"]["query"]["knn"]
```

### 4. Configuration Lock File
```python
# src/infra/query_structure_lock.py
"""
Query Structure Lock - DO NOT MODIFY WITHOUT UPDATING SOURCE OF TRUTH DOC

This file preserves the exact working query structure discovered on 2025-08-19.
Any changes to these values must be reflected in QUERY_STRUCTURE_SOURCE_OF_TRUTH.md
"""

VERIFIED_WORKING_CONFIG = {
    "embedding_dimensions": 1536,
    "content_field": "sections.content", 
    "vector_field": "sections.embedding",
    "nested_path": "sections",
    "inner_hits_name": "matched_sections",
    "uses_native_knn": True,
    "uses_script_score": False
}
```

### 5. Change Management Process
1. **Before any query changes**: Document current behavior
2. **Test against known working queries**: Use saved examples
3. **Validate with real data**: Run against actual index
4. **Update source of truth**: Keep this document current
5. **Create rollback plan**: Know how to revert changes

## 📝 Working Query Examples (Tested & Verified)

### Example 1: "what is Au" Query
- **Query Type**: BM25 nested
- **Results**: 8 hits with AU Auto Loan API content
- **Inner Hits**: Successfully extracted section content
- **Status**: ✅ WORKING

### Example 2: Vector Search  
- **Query Type**: KNN nested with 1536 dimensions
- **Field**: sections.embedding  
- **Inner Hits**: Section-level similarity scoring
- **Status**: ✅ WORKING

## 🚨 Critical Preservation Points

1. **NEVER** change field names without updating centralized config
2. **NEVER** remove nested structure - flat queries will fail
3. **NEVER** change embedding dimensions without index reindexing
4. **ALWAYS** test query changes against working examples
5. **ALWAYS** preserve inner_hits for section-level extraction

## 🔄 Migration Path for Future Changes

If query structure needs modification:

1. Create feature branch from current working state
2. Update centralized QueryTemplates first
3. Test with isolated query validation
4. Update this source of truth document 
5. Update tests to match new structure
6. Deploy with rollback capability

---

**Next Review Date**: 2025-09-19  
**Responsible Team**: Search Infrastructure  
**Emergency Rollback**: `git checkout v1-working-code -- src/infra/search_config.py`