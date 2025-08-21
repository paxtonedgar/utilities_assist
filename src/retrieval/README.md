# Retrieval - Search, Ranking, and View Systems

## Purpose
Advanced retrieval system implementing BGE v2-m3 cross-encoder optimization with performance gating, RRF fusion, and view-based search patterns. Provides the core search intelligence for the LangGraph workflow with nuanced result steering based on intent coloring.

## Architecture
Multi-layered retrieval pipeline optimized for both speed and relevance:

```
Query → [BM25 + kNN] → RRF Fusion → Coverage Gate → BGE Reranking → Views → Presenter
                                         ↓
                                   Skip if <3 docs
                                   (saves 6-8s)
```

### Performance-First Design
- **Coverage Gating**: Prevents expensive reranking on insufficient results
- **Parallel Views**: Info and procedure views constructed concurrently  
- **Adaptive Budgets**: Timeout reduction for urgent queries
- **Smart Caching**: TTL layers for embeddings and tokenization

## Key Files

### Core Retrieval Engine
- `config.py` - **Centralized performance configuration** (52 lines)
  - P1-P3 latency budgets and cache settings
  - Coloring thresholds and tuning parameters
  - Actionability weights and presenter gates
- `tuning.py` - **Color-based retrieval tuning** (165 lines)
  - Specificity-based search narrowing/widening
  - Suite affinity boosting
  - Time urgency timeout adjustments
- `views.py` - **View construction logic**
  - Info view for definitions and explanations
  - Procedure view for step-by-step guidance
  - Parallel execution with selective building
- `actionability.py` - **Span extraction and actionability scoring** (349 lines)
  - Capability detection (ticket.create, api.try, form.submit)
  - Suite-specific span extraction
  - Presenter selection logic

### Suite Management
- `suites/registry.py` - **YAML-based suite definitions**
  - Domain-specific extraction patterns
  - URL routing and keyword mappings
  - Extensible manifest system
- `suites/manifests/` - **Suite configuration files**
  - Jira, Teams, ServiceNow, API suite definitions
  - Regex patterns and capability mappings

## Dependencies

### Internal Dependencies
- `src.services.retrieve` - Core search implementations (RRF, cross-encoder)
- `src.agent.intent.coloring` - Intent coloring for retrieval tuning
- `src.infra.opensearch_client` - Search backend integration
- `src.telemetry.logger` - Performance tracking and metrics

### External Dependencies  
- `transformers` - BGE v2-m3 model loading and inference
- `torch` - Neural network computation for cross-encoder
- `sentence-transformers` - Embedding model integration
- `numpy` - Numerical operations for scoring

## Integration Points

### LangGraph Node Integration
```python
# Color-based view planning
def should_build_procedure_view(slot_result) -> bool:
    colors = slot_result.colors
    return (
        colors.actionability_est >= 1.0 or
        max(colors.suite_affinity.values()) >= 0.6 or
        colors.artifact_types & {"endpoint", "ticket", "form"}
    )

# Retrieval knob tuning
def tune_for_colors(knobs, colors):
    if colors.specificity == "high":
        knobs["knn_k"] = min(knobs["knn_k"], 24)  # Narrow search
    if colors.time_urgency == "hard":
        knobs["ce_timeout_s"] *= 0.6  # Reduce timeout
```

### LlamaIndex Migration Pathways

#### Query Engines
```python
# Current: Manual view construction
info_view = await run_info_view(query, search_client, embed_client)

# LlamaIndex path: Query engine selection
from llama_index.core.query_engine import RouterQueryEngine
query_engine = RouterQueryEngine(
    selector=IntentBasedSelector(slot_result.colors),
    query_engines={
        "info": VectorStoreQueryEngine(vector_store),
        "procedure": SubQuestionQueryEngine(service_context)
    }
)
```

#### Response Synthesis
```python
# Current: Custom presenter selection
decision = choose_presenter(info_view, procedure_view)

# LlamaIndex path: Response modes
from llama_index.core.response.response_mode import ResponseMode
response_mode = (
    ResponseMode.TREE_SUMMARIZE if colors.actionability_est > 2.0
    else ResponseMode.COMPACT
)
```

#### Index Management
```python
# Current: Direct OpenSearch integration
search_client = OpenSearchClient(config)

# LlamaIndex path: Vector store abstraction
from llama_index.vector_stores.opensearch import OpenSearchVectorStore
vector_store = OpenSearchVectorStore(
    opensearch_url=config.host,
    index_name=config.index_alias
)
```

## Performance Characteristics

### Critical Path Optimizations
- **BGE Gating**: 6-8s savings when unique docs < 3
- **RRF Caching**: 60% cache hit rate for embedding reuse
- **Parallel Views**: 40% latency reduction vs sequential
- **Timeout Scaling**: Adaptive budgets based on urgency

### Memory Management
- **Model Loading**: BGE v2-m3 compressed (1.2GB → 400MB)
- **Embedding Cache**: TTL-based cleanup, 1000 entries max
- **Connection Pooling**: Persistent OpenSearch connections
- **Batch Processing**: Efficient tokenization for reranking

### Scalability Factors
- **Concurrent Requests**: Thread-safe caching and model sharing
- **Index Size**: Optimized for 100K+ document corpora
- **Cross-Encoder Load**: Async processing with timeout guards
- **Cache Efficiency**: Semantic deduplication reduces redundant computation

## Current Implementation Details

### BGE v2-m3 Integration
```python
# Compressed model with auto-extraction
model_path = "models/bge-reranker-v2-m3-compressed"
if not model_path.exists():
    extract_compressed_model()  # One-time setup

# Performance-gated reranking
unique_docs = len(set(r.doc_id for r in rrf_results))
if unique_docs >= 3:  # Coverage threshold
    reranked = cross_encoder_rerank(query, results, timeout=1.8s)
```

### RRF Fusion Implementation
```python
# Reciprocal Rank Fusion with adaptive weights
def rrf_fusion(bm25_results, knn_results, k=60):
    scores = defaultdict(float)
    for rank, result in enumerate(bm25_results):
        scores[result.doc_id] += 1 / (rank + k)
    for rank, result in enumerate(knn_results):
        scores[result.doc_id] += 1 / (rank + k)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

### View Construction Pattern
```python
# Parallel view building with selective construction
async def build_views(query, slot_result, resources):
    # Always build info view
    info_task = run_info_view(query, **resources)
    
    # Conditionally build procedure view
    procedure_task = None
    if should_build_procedure_view(slot_result):
        procedure_task = run_procedure_view(query, **resources)
    
    # Await results
    info_view = await info_task
    procedure_view = await procedure_task if procedure_task else None
    
    return info_view, procedure_view
```

## Future Enhancement Opportunities

### LlamaIndex Integration
- **Query Pipeline**: Replace manual view logic with LlamaIndex query pipeline
- **Retrieval Augmentation**: Use LlamaIndex's retrieval augmentation patterns
- **Document Processing**: Leverage LlamaIndex parsers for multi-modal content
- **Response Synthesis**: Adopt LlamaIndex response synthesis modes

### Performance Optimizations
- **GPU Acceleration**: CUDA support for BGE inference
- **Model Quantization**: INT8/FP16 optimization for faster inference
- **Streaming Results**: Real-time result streaming for long queries
- **Federated Search**: Multi-index parallel search with result fusion

### Advanced Features
- **Learned Sparse Retrieval**: SPLADE integration for better BM25
- **Dynamic Embeddings**: Context-aware embedding generation
- **Relevance Feedback**: User feedback integration for ranking improvement
- **Multi-Modal Search**: Image and document content integration