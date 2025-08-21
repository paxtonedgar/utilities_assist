# Search - Search Infrastructure and Index Mappings

## Purpose
Provides search infrastructure components including OpenSearch index mappings, search configuration schemas, and search-related utilities. Defines the foundational search architecture that supports document indexing, retrieval, and relevance scoring across the utilities assistant system.

## Architecture
Search infrastructure layer supporting document storage and retrieval:

```
Search Queries
      ↓
Search Infrastructure
      ↓
┌─────────────────┬─────────────────┐
│ Index Mappings  │ Search Config   │
│ & Schemas       │ & Templates     │
└─────────────────┴─────────────────┘
      ↓
OpenSearch Cluster
```

### Design Principles
- **Schema-Driven**: Well-defined index structures for consistent data storage
- **Performance Optimized**: Mappings designed for fast search and aggregation
- **Scalable Architecture**: Support for large document collections
- **Relevance Tuned**: Optimized field mappings for relevance scoring

## Key Files

### Core Search Infrastructure
- `mappings/confluence_v2.json` - **OpenSearch index mapping for Confluence documents** (JSON schema)
  - Field definitions and data types for Confluence content
  - Search-optimized text analysis and tokenization
  - Metadata structure for document organization
  - Performance-tuned mapping configuration

## Dependencies

### Internal Dependencies
- `src.infra.opensearch_client` - OpenSearch cluster interaction
- `src.infra.search_config` - Search configuration management
- `src.retrieval.config` - Retrieval-specific search settings
- `src.telemetry.logger` - Search performance monitoring

### External Dependencies
- `opensearch` - OpenSearch cluster for document storage and search
- `elasticsearch` (compatibility) - Legacy Elasticsearch compatibility if needed

## Integration Points

### Index Mapping Structure
```json
{
  "mappings": {
    "properties": {
      "title": {
        "type": "text",
        "analyzer": "standard",
        "fields": {
          "keyword": {
            "type": "keyword",
            "ignore_above": 256
          },
          "suggest": {
            "type": "completion",
            "analyzer": "simple"
          }
        }
      },
      "content": {
        "type": "text",
        "analyzer": "content_analyzer",
        "search_analyzer": "search_analyzer",
        "fields": {
          "exact": {
            "type": "text",
            "analyzer": "keyword"
          }
        }
      },
      "url": {
        "type": "keyword",
        "index": false
      },
      "space": {
        "type": "keyword"
      },
      "page_id": {
        "type": "keyword"
      },
      "created_date": {
        "type": "date",
        "format": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
      },
      "updated_date": {
        "type": "date", 
        "format": "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"
      },
      "author": {
        "type": "keyword"
      },
      "labels": {
        "type": "keyword"
      },
      "ancestors": {
        "type": "nested",
        "properties": {
          "id": {"type": "keyword"},
          "title": {"type": "text"}
        }
      },
      "embedding": {
        "type": "dense_vector",
        "dims": 1536,
        "index": true,
        "similarity": "cosine"
      },
      "chunk_id": {
        "type": "keyword"
      },
      "chunk_index": {
        "type": "integer"
      },
      "total_chunks": {
        "type": "integer"
      }
    }
  }
}
```

### Custom Analyzers for Utility Content
```json
{
  "settings": {
    "analysis": {
      "analyzer": {
        "content_analyzer": {
          "type": "custom",
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "utility_synonyms",
            "technical_terms",
            "stop_words_filter"
          ]
        },
        "search_analyzer": {
          "type": "custom", 
          "tokenizer": "standard",
          "filter": [
            "lowercase",
            "utility_synonyms",
            "technical_terms"
          ]
        },
        "exact_analyzer": {
          "type": "custom",
          "tokenizer": "keyword",
          "filter": ["lowercase"]
        }
      },
      "filter": {
        "utility_synonyms": {
          "type": "synonym",
          "synonyms": [
            "API,api,endpoint,service",
            "Jira,JIRA,ticket,issue",
            "Teams,MS Teams,Microsoft Teams,chat",
            "CIU,Customer Information Utilities",
            "ServiceNow,SNOW,ticket system"
          ]
        },
        "technical_terms": {
          "type": "keyword_marker",
          "keywords": [
            "REST", "API", "JSON", "XML", "HTTP", "HTTPS",
            "OAuth", "JWT", "SSL", "TLS", "LDAP", "SAML"
          ]
        },
        "stop_words_filter": {
          "type": "stop",
          "stopwords": ["the", "is", "at", "which", "on"]
        }
      }
    }
  }
}
```

### Vector Search Configuration
```python
# Integration with vector search capabilities
class VectorSearchConfig:
    """Configuration for vector-based semantic search."""
    
    def __init__(self):
        self.embedding_dimensions = 1536  # OpenAI text-embedding-ada-002
        self.similarity_metric = "cosine"
        self.vector_field = "embedding"
        self.knn_candidates = 100
        self.knn_k = 10
    
    def build_knn_query(
        self,
        query_embedding: List[float],
        k: int = None,
        filter_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Build k-NN query for vector search."""
        
        knn_query = {
            "knn": {
                self.vector_field: {
                    "vector": query_embedding,
                    "k": k or self.knn_k,
                    "num_candidates": self.knn_candidates
                }
            }
        }
        
        # Add filtering if provided
        if filter_context:
            knn_query["knn"][self.vector_field]["filter"] = self._build_filter(filter_context)
        
        return knn_query
    
    def _build_filter(self, filter_context: Dict[str, Any]) -> Dict[str, Any]:
        """Build filter clause for vector search."""
        
        filters = []
        
        # Space/domain filtering
        if "spaces" in filter_context:
            filters.append({
                "terms": {"space": filter_context["spaces"]}
            })
        
        # Date range filtering
        if "date_range" in filter_context:
            date_range = filter_context["date_range"]
            filters.append({
                "range": {
                    "updated_date": {
                        "gte": date_range.get("start"),
                        "lte": date_range.get("end")
                    }
                }
            })
        
        # Author filtering
        if "authors" in filter_context:
            filters.append({
                "terms": {"author": filter_context["authors"]}
            })
        
        if len(filters) == 1:
            return filters[0]
        elif len(filters) > 1:
            return {"bool": {"must": filters}}
        else:
            return {}
```

### Hybrid Search Integration
```python
# Combining BM25 keyword search with vector similarity
class HybridSearchBuilder:
    """Build hybrid search queries combining multiple search strategies."""
    
    def __init__(self, vector_config: VectorSearchConfig):
        self.vector_config = vector_config
        self.bm25_boost = 1.0
        self.vector_boost = 1.0
        self.rrf_rank_constant = 60  # Reciprocal Rank Fusion constant
    
    def build_hybrid_query(
        self,
        text_query: str,
        query_embedding: List[float],
        filter_context: Dict[str, Any] = None,
        boost_config: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """Build hybrid query combining BM25 and vector search."""
        
        if boost_config:
            self.bm25_boost = boost_config.get("bm25", 1.0)
            self.vector_boost = boost_config.get("vector", 1.0)
        
        # Build BM25 query
        bm25_query = self._build_bm25_query(text_query, filter_context)
        
        # Build vector query
        vector_query = self.vector_config.build_knn_query(
            query_embedding, filter_context=filter_context
        )
        
        # Combine with boosting
        hybrid_query = {
            "query": {
                "bool": {
                    "should": [
                        {
                            "bool": {
                                "must": [bm25_query],
                                "boost": self.bm25_boost
                            }
                        }
                    ]
                }
            }
        }
        
        # Add k-NN search separately (OpenSearch syntax)
        if "knn" in vector_query:
            hybrid_query["knn"] = vector_query["knn"]
            hybrid_query["knn"][self.vector_config.vector_field]["boost"] = self.vector_boost
        
        return hybrid_query
    
    def _build_bm25_query(
        self,
        text_query: str,
        filter_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Build BM25 keyword search query."""
        
        # Multi-match query across text fields
        bm25_query = {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": text_query,
                            "fields": [
                                "title^3",      # Boost title matches
                                "content^1",    # Standard content
                                "labels^2"      # Boost label matches
                            ],
                            "type": "best_fields",
                            "fuzziness": "AUTO"
                        }
                    },
                    {
                        "match_phrase": {
                            "content": {
                                "query": text_query,
                                "boost": 2,     # Boost exact phrases
                                "slop": 2       # Allow some word distance
                            }
                        }
                    }
                ]
            }
        }
        
        # Add filtering
        if filter_context:
            filter_clause = self.vector_config._build_filter(filter_context)
            if filter_clause:
                bm25_query["bool"]["filter"] = filter_clause
        
        return bm25_query
    
    def apply_rrf_fusion(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion to combine results."""
        
        # Create rank mappings
        bm25_ranks = {
            result["_id"]: rank + 1 
            for rank, result in enumerate(bm25_results)
        }
        
        vector_ranks = {
            result["_id"]: rank + 1 
            for rank, result in enumerate(vector_results)
        }
        
        # Calculate RRF scores
        all_doc_ids = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        rrf_scores = {}
        
        for doc_id in all_doc_ids:
            bm25_rank = bm25_ranks.get(doc_id, len(bm25_results) + 1)
            vector_rank = vector_ranks.get(doc_id, len(vector_results) + 1)
            
            rrf_score = (
                1.0 / (self.rrf_rank_constant + bm25_rank) +
                1.0 / (self.rrf_rank_constant + vector_rank)
            )
            
            rrf_scores[doc_id] = rrf_score
        
        # Sort by RRF score and return merged results
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create result mapping
        doc_lookup = {}
        for result in bm25_results + vector_results:
            doc_lookup[result["_id"]] = result
        
        # Build final ranked results
        fused_results = []
        for doc_id, rrf_score in sorted_docs:
            if doc_id in doc_lookup:
                result = doc_lookup[doc_id].copy()
                result["_score"] = rrf_score
                result["_fusion_method"] = "rrf"
                fused_results.append(result)
        
        return fused_results
```

### Search Performance Optimization
```python
# Search performance monitoring and optimization
class SearchPerformanceOptimizer:
    """Optimize search performance and monitor metrics."""
    
    def __init__(self, telemetry_integration):
        self.telemetry = telemetry_integration
        self.performance_budgets = {
            "bm25_search_ms": 100,
            "vector_search_ms": 200,
            "hybrid_search_ms": 300,
            "index_size_gb": 50
        }
        self.optimization_strategies = {
            "cache_frequent_queries": True,
            "optimize_field_mappings": True,
            "monitor_shard_performance": True,
            "track_query_patterns": True
        }
    
    def monitor_search_performance(
        self,
        search_type: str,
        query: str,
        execution_time_ms: float,
        result_count: int,
        hit_quality: float = None
    ):
        """Monitor individual search operation performance."""
        
        # Log performance metrics
        self.telemetry.logger.info(
            f"Search performance: {search_type}",
            search_type=search_type,
            execution_time_ms=round(execution_time_ms, 2),
            result_count=result_count,
            hit_quality=hit_quality,
            query_length=len(query)
        )
        
        # Check performance budgets
        budget_key = f"{search_type}_search_ms"
        if budget_key in self.performance_budgets:
            budget = self.performance_budgets[budget_key]
            if execution_time_ms > budget:
                self.telemetry.logger.warning(
                    f"Search performance budget exceeded",
                    search_type=search_type,
                    budget_ms=budget,
                    actual_ms=execution_time_ms,
                    overage_ms=execution_time_ms - budget,
                    query_preview=query[:100]
                )
    
    def analyze_query_patterns(
        self,
        queries: List[str],
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze query patterns for optimization opportunities."""
        
        analysis = {
            "total_queries": len(queries),
            "unique_queries": len(set(queries)),
            "repetition_rate": 1 - len(set(queries)) / len(queries),
            "avg_query_length": sum(len(q) for q in queries) / len(queries),
            "common_terms": self._extract_common_terms(queries),
            "query_complexity": self._analyze_query_complexity(queries)
        }
        
        # Generate optimization recommendations
        recommendations = []
        
        if analysis["repetition_rate"] > 0.3:
            recommendations.append({
                "type": "caching",
                "description": "High query repetition - implement query result caching",
                "impact": "medium"
            })
        
        if analysis["avg_query_length"] > 50:
            recommendations.append({
                "type": "query_optimization",
                "description": "Long queries detected - consider query simplification",
                "impact": "low"
            })
        
        analysis["optimization_recommendations"] = recommendations
        
        return analysis
    
    def _extract_common_terms(self, queries: List[str], top_n: int = 20) -> List[Dict[str, Any]]:
        """Extract most common terms across queries."""
        
        from collections import Counter
        
        # Tokenize and count terms
        all_terms = []
        for query in queries:
            terms = query.lower().split()
            # Filter out common stop words
            filtered_terms = [
                term for term in terms 
                if len(term) > 2 and term not in ["the", "and", "for", "are", "with"]
            ]
            all_terms.extend(filtered_terms)
        
        term_counts = Counter(all_terms)
        
        return [
            {"term": term, "count": count, "frequency": count / len(all_terms)}
            for term, count in term_counts.most_common(top_n)
        ]
    
    def _analyze_query_complexity(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze query complexity metrics."""
        
        complexity_metrics = {
            "simple": 0,    # 1-3 words
            "medium": 0,    # 4-8 words
            "complex": 0,   # 9+ words
            "boolean": 0,   # Contains AND/OR/NOT
            "phrase": 0,    # Contains quoted phrases
            "wildcard": 0   # Contains * or ?
        }
        
        for query in queries:
            word_count = len(query.split())
            
            if word_count <= 3:
                complexity_metrics["simple"] += 1
            elif word_count <= 8:
                complexity_metrics["medium"] += 1
            else:
                complexity_metrics["complex"] += 1
            
            query_upper = query.upper()
            if any(op in query_upper for op in ["AND", "OR", "NOT"]):
                complexity_metrics["boolean"] += 1
            
            if '"' in query:
                complexity_metrics["phrase"] += 1
            
            if any(char in query for char in ["*", "?"]):
                complexity_metrics["wildcard"] += 1
        
        # Convert to percentages
        total = len(queries)
        for key in complexity_metrics:
            complexity_metrics[key] = complexity_metrics[key] / total
        
        return complexity_metrics
```

### Index Management and Maintenance
```python
# Index lifecycle management and optimization
class IndexManager:
    """Manage OpenSearch index lifecycle and optimization."""
    
    def __init__(self, opensearch_client, telemetry_integration):
        self.client = opensearch_client
        self.telemetry = telemetry_integration
    
    async def create_optimized_index(
        self,
        index_name: str,
        mapping_config: Dict[str, Any],
        shard_config: Dict[str, Any] = None
    ) -> bool:
        """Create index with optimized configuration."""
        
        # Default shard configuration
        default_shard_config = {
            "number_of_shards": 3,
            "number_of_replicas": 1,
            "refresh_interval": "1s",
            "max_result_window": 50000
        }
        
        if shard_config:
            default_shard_config.update(shard_config)
        
        index_config = {
            "settings": {
                "index": default_shard_config,
                **mapping_config.get("settings", {})
            },
            "mappings": mapping_config.get("mappings", {})
        }
        
        try:
            response = await self.client.indices.create(
                index=index_name,
                body=index_config
            )
            
            self.telemetry.logger.info(
                f"Index created successfully: {index_name}",
                index_name=index_name,
                shards=default_shard_config["number_of_shards"],
                replicas=default_shard_config["number_of_replicas"]
            )
            
            return True
            
        except Exception as e:
            self.telemetry.logger.error(
                f"Failed to create index: {index_name}",
                error=e,
                index_name=index_name
            )
            return False
    
    async def optimize_index_performance(self, index_name: str) -> Dict[str, Any]:
        """Optimize index for better search performance."""
        
        optimization_results = {}
        
        try:
            # Force merge to optimize segments
            merge_response = await self.client.indices.forcemerge(
                index=index_name,
                max_num_segments=1,
                wait_for_completion=True
            )
            optimization_results["force_merge"] = "completed"
            
            # Refresh index
            await self.client.indices.refresh(index=index_name)
            optimization_results["refresh"] = "completed"
            
            # Get index stats
            stats = await self.client.indices.stats(index=index_name)
            optimization_results["stats"] = {
                "docs_count": stats["indices"][index_name]["total"]["docs"]["count"],
                "store_size": stats["indices"][index_name]["total"]["store"]["size_in_bytes"],
                "segments_count": stats["indices"][index_name]["total"]["segments"]["count"]
            }
            
            self.telemetry.logger.info(
                f"Index optimization completed: {index_name}",
                index_name=index_name,
                **optimization_results["stats"]
            )
            
        except Exception as e:
            self.telemetry.logger.error(
                f"Index optimization failed: {index_name}",
                error=e,
                index_name=index_name
            )
            optimization_results["error"] = str(e)
        
        return optimization_results
```

## Current Implementation Details

### Confluence Document Processing
The current implementation focuses on Confluence document indexing with optimized field mappings for utilities content:

- **Text Analysis**: Custom analyzers for utility-specific terminology
- **Vector Integration**: Dense vector fields for semantic search
- **Metadata Preservation**: Comprehensive document metadata for filtering
- **Chunk Management**: Support for document chunking and reassembly

### Performance Characteristics
- **Index Size**: Optimized for 10,000+ documents
- **Search Latency**: Sub-200ms for most queries
- **Throughput**: 100+ concurrent search operations
- **Storage**: Efficient field mappings minimizing storage overhead

## Future Enhancement Opportunities

### Advanced Search Capabilities
- **Multi-Index Search**: Search across multiple document types
- **Faceted Search**: Dynamic filtering and categorization
- **Query Suggestions**: Auto-complete and query enhancement
- **Personalized Search**: User-specific result ranking

### Index Optimization
- **Dynamic Mapping**: Automatic field detection and optimization
- **Hot/Warm Architecture**: Tiered storage for different data ages
- **Compression**: Advanced compression for large document collections
- **Shard Optimization**: Dynamic shard allocation based on usage

### Integration Enhancements
- **Real-Time Indexing**: Stream processing for immediate document updates
- **Multi-Modal Search**: Support for images, videos, and other media
- **External Data Sources**: Integration with additional document repositories
- **Search Analytics**: Comprehensive search behavior analysis

## Monitoring and Maintenance

### Search Health Monitoring
- **Index Health**: Regular index status and performance monitoring
- **Query Performance**: Search latency and throughput tracking
- **Resource Usage**: Memory and CPU utilization monitoring
- **Error Tracking**: Search failure analysis and alerting

### Maintenance Procedures
- **Index Optimization**: Regular index maintenance and optimization
- **Mapping Updates**: Schema evolution and backward compatibility
- **Data Lifecycle**: Automated data retention and archival
- **Backup and Recovery**: Index backup and disaster recovery procedures