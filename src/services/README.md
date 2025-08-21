# Services - Core Business Logic and Processing Services

## Purpose
Contains core business logic services that handle specific processing tasks including intent classification, response generation, document retrieval, and semantic reranking. These services implement the fundamental algorithms and processing pipelines that power the utilities assistant system.

## Architecture
Service-oriented architecture with focused, single-responsibility modules:

```
Business Logic Services
         ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Intent & Query  │ Retrieval &     │ Response &      │
│   Processing    │  Reranking     │  Generation     │
└─────────────────┴─────────────────┴─────────────────┘
```

### Design Principles
- **Single Responsibility**: Each service handles one specific business function
- **Stateless Operations**: Services maintain no internal state for scalability
- **Type Safety**: Strong typing with Pydantic models for data validation
- **Performance Focus**: Optimized algorithms for sub-second response times

## Key Files

### Core Processing Services
- `intent.py` - **LLM-based intent classification** (284 lines)
  - Query intent determination and context extraction
  - Integration with Azure OpenAI for natural language understanding
  - Fallback mechanisms and error handling
  - Performance monitoring and caching

- `normalize.py` - **Query normalization and preprocessing** (142 lines)
  - Text cleaning and standardization
  - Stop word removal and stemming
  - Query expansion and synonym handling
  - Input validation and sanitization

- `reranker.py` - **BGE cross-encoder semantic reranking** (376 lines)
  - BAAI BGE-reranker-v2-m3 model integration
  - Compressed model loading and inference
  - Batch processing for efficiency
  - Performance gating and fallback strategies

### Retrieval and Response
- `retrieve.py` - **Document retrieval coordination** (198 lines)
  - Multi-strategy search orchestration
  - BM25 and vector search combination
  - Result filtering and deduplication
  - Performance metrics collection

- `respond.py` - **Response generation and formatting** (156 lines)
  - Context-aware response synthesis
  - Template-based formatting
  - Source citation management
  - Output format standardization

- `models.py` - **Data models and type definitions** (89 lines)
  - Pydantic models for type safety
  - Request/response schemas
  - Validation rules and constraints
  - Serialization/deserialization logic

## Dependencies

### Internal Dependencies
- `src.infra.clients` - Azure OpenAI and OpenSearch client management
- `src.infra.config` - Configuration and settings access
- `src.telemetry.logger` - Performance monitoring and logging
- `src.util.cache` - Caching utilities for performance optimization
- `src.util.timing` - Performance measurement and profiling

### External Dependencies
- `openai` - Azure OpenAI integration for LLM services
- `sentence_transformers` - BGE model loading and inference
- `pydantic` - Data validation and type safety
- `numpy` - Numerical operations for embedding processing
- `torch` - PyTorch backend for transformer models

## Integration Points

### Intent Classification Service
```python
# LLM-based intent classification with caching
async def determine_intent(
    query: str,
    context: Dict[str, Any],
    llm_client: AzureChatOpenAI
) -> IntentResult:
    """Classify user intent using Azure OpenAI."""
    
    # Check cache first
    cache_key = generate_intent_cache_key(query, context)
    cached_result = get_cached_intent(cache_key)
    if cached_result:
        return cached_result
    
    # Prepare prompt with context
    prompt = format_intent_prompt(query, context)
    
    # Call LLM with retry logic
    with performance_monitor("intent_classification"):
        response = await llm_client.ainvoke(prompt)
    
    # Parse and validate response
    intent_result = parse_intent_response(response.content)
    
    # Cache result
    cache_intent_result(cache_key, intent_result)
    
    return intent_result
```

### BGE Reranking Service
```python
# High-performance semantic reranking
class BGEReranker:
    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize BGE reranker with compressed model."""
        self.model = SentenceTransformer(model_path, device=device)
        self.max_length = 512
        self.batch_size = 32
    
    def rerank(
        self,
        query: str,
        documents: List[SearchResult],
        top_k: int = 10
    ) -> List[RankedResult]:
        """Rerank documents using BGE cross-encoder."""
        
        # Prepare query-document pairs
        pairs = [(query, doc.content) for doc in documents]
        
        # Batch processing for efficiency
        scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            batch_scores = self.model.similarity(
                [pair[0] for pair in batch],
                [pair[1] for pair in batch]
            )
            scores.extend(batch_scores.tolist())
        
        # Combine scores with documents
        ranked_results = [
            RankedResult(
                document=doc,
                score=score,
                rank=idx
            )
            for idx, (doc, score) in enumerate(
                sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            )
        ]
        
        return ranked_results[:top_k]
```

### Query Normalization Service
```python
# Text preprocessing and normalization
def normalize_query(raw_query: str) -> NormalizedQuery:
    """Normalize user query for better search performance."""
    
    # Basic cleaning
    query = raw_query.strip().lower()
    
    # Remove special characters but preserve meaningful punctuation
    query = re.sub(r'[^\w\s\-\.\?\!]', ' ', query)
    
    # Normalize whitespace
    query = ' '.join(query.split())
    
    # Expand common abbreviations
    query = expand_abbreviations(query)
    
    # Extract key phrases
    key_phrases = extract_key_phrases(query)
    
    # Generate search variants
    search_variants = generate_search_variants(query, key_phrases)
    
    return NormalizedQuery(
        original=raw_query,
        normalized=query,
        key_phrases=key_phrases,
        search_variants=search_variants,
        metadata={
            "length": len(query),
            "word_count": len(query.split()),
            "has_question": "?" in query
        }
    )
```

## Performance Considerations

### Caching Strategy
- **Intent Results**: TTL cache with 10-minute expiration
- **Reranking Models**: Singleton pattern for model loading
- **Normalized Queries**: In-memory cache for session duration
- **LLM Responses**: Semantic similarity-based caching

### Batch Processing
- **BGE Reranking**: Process documents in batches of 32
- **Embedding Generation**: Vectorized operations where possible
- **Response Synthesis**: Template compilation caching
- **Model Inference**: GPU acceleration when available

### Memory Management
- **Model Loading**: Lazy loading with memory monitoring
- **Cache Eviction**: LRU eviction with size limits
- **Batch Size Tuning**: Dynamic adjustment based on available memory
- **Resource Cleanup**: Explicit cleanup of large objects

## Current Implementation Details

### Intent Classification Pipeline
```python
# Multi-stage intent processing
async def process_intent_pipeline(
    query: str,
    user_context: Dict[str, Any],
    llm_client: AzureChatOpenAI
) -> ProcessedIntent:
    """Complete intent processing pipeline."""
    
    # Stage 1: Normalize query
    normalized = normalize_query(query)
    
    # Stage 2: Extract context features
    context_features = extract_context_features(user_context)
    
    # Stage 3: Classify intent
    intent_result = await determine_intent(
        normalized.normalized,
        context_features,
        llm_client
    )
    
    # Stage 4: Validate and enrich
    validated_intent = validate_intent_result(intent_result)
    enriched_intent = enrich_with_domain_knowledge(validated_intent)
    
    return ProcessedIntent(
        original_query=query,
        normalized_query=normalized,
        intent_result=enriched_intent,
        processing_metadata={
            "normalization_time": normalized.metadata["processing_time"],
            "classification_time": intent_result.metadata["processing_time"],
            "confidence": intent_result.confidence,
            "fallback_used": intent_result.metadata.get("fallback_used", False)
        }
    )
```

### Retrieval Coordination
```python
# Multi-strategy retrieval coordination
async def coordinate_retrieval(
    query: NormalizedQuery,
    intent: IntentResult,
    search_config: SearchConfig
) -> RetrievalResult:
    """Coordinate multiple retrieval strategies."""
    
    # Parallel strategy execution
    tasks = []
    
    # BM25 keyword search
    if search_config.enable_bm25:
        tasks.append(execute_bm25_search(query, intent))
    
    # Vector semantic search
    if search_config.enable_vector:
        tasks.append(execute_vector_search(query, intent))
    
    # Domain-specific search
    if intent.domain and search_config.enable_domain_specific:
        tasks.append(execute_domain_search(query, intent))
    
    # Execute all strategies in parallel
    strategy_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    valid_results = [
        result for result in strategy_results
        if not isinstance(result, Exception)
    ]
    
    # Combine and deduplicate
    combined_results = combine_search_results(valid_results)
    deduplicated = deduplicate_results(combined_results)
    
    return RetrievalResult(
        documents=deduplicated,
        strategy_metadata={
            "strategies_used": len(valid_results),
            "total_documents": len(combined_results),
            "deduplicated_count": len(deduplicated),
            "execution_time": sum(r.execution_time for r in valid_results)
        }
    )
```

### Response Generation Service
```python
# Context-aware response generation
async def generate_response(
    query: str,
    retrieval_result: RetrievalResult,
    intent: IntentResult,
    llm_client: AzureChatOpenAI
) -> GeneratedResponse:
    """Generate contextual response from retrieval results."""
    
    # Select relevant documents
    relevant_docs = select_relevant_documents(
        retrieval_result.documents,
        intent,
        max_docs=5
    )
    
    # Build context prompt
    context_prompt = build_context_prompt(
        query=query,
        documents=relevant_docs,
        intent=intent,
        template_name="answer"
    )
    
    # Generate response with LLM
    with performance_monitor("response_generation"):
        response = await llm_client.ainvoke(context_prompt)
    
    # Post-process and validate
    processed_response = post_process_response(response.content)
    validated_response = validate_response_quality(processed_response)
    
    # Extract citations
    citations = extract_citations(relevant_docs, processed_response)
    
    return GeneratedResponse(
        content=validated_response,
        citations=citations,
        confidence=calculate_response_confidence(validated_response, relevant_docs),
        metadata={
            "documents_used": len(relevant_docs),
            "response_length": len(validated_response),
            "generation_time": response.metadata["processing_time"],
            "model_used": llm_client.model_name
        }
    )
```

## Error Handling and Resilience

### Service-Level Error Handling
```python
# Comprehensive error handling with fallbacks
async def safe_intent_classification(
    query: str,
    context: Dict[str, Any],
    llm_client: AzureChatOpenAI
) -> IntentResult:
    """Intent classification with multiple fallback strategies."""
    
    try:
        # Primary LLM classification
        return await determine_intent(query, context, llm_client)
    
    except OpenAIRateLimitError:
        logger.warning("Rate limit hit, using cached fallback")
        return get_fallback_intent_from_cache(query)
    
    except OpenAIServiceError as e:
        logger.error(f"OpenAI service error: {e}")
        return get_rule_based_intent_fallback(query)
    
    except Exception as e:
        logger.error(f"Unexpected intent classification error: {e}")
        return IntentResult(
            intent="unknown",
            confidence=0.0,
            metadata={"error": str(e), "fallback_used": True}
        )

# Circuit breaker pattern for external services
class ServiceCircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_time: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, service_func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_time:
                self.state = "half-open"
            else:
                raise ServiceUnavailableError("Circuit breaker is open")
        
        try:
            result = await service_func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e
```

### Graceful Degradation
```python
# Multi-level fallback strategies
async def resilient_retrieval(
    query: NormalizedQuery,
    intent: IntentResult
) -> RetrievalResult:
    """Retrieval with graceful degradation."""
    
    try:
        # Primary: Full multi-strategy retrieval
        return await coordinate_retrieval(query, intent, full_config)
    
    except OpenSearchConnectionError:
        logger.warning("OpenSearch unavailable, using cached results")
        return get_cached_retrieval_results(query)
    
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        
        # Fallback: Simple keyword matching
        fallback_results = simple_keyword_retrieval(query.normalized)
        
        return RetrievalResult(
            documents=fallback_results,
            strategy_metadata={
                "fallback_used": True,
                "error": str(e),
                "strategy": "simple_keyword"
            }
        )
```

## Performance Monitoring

### Service Metrics Collection
```python
# Comprehensive service performance monitoring
@performance_tracker
async def monitored_service_call(
    service_name: str,
    operation: str,
    service_func,
    *args,
    **kwargs
):
    """Track service performance with detailed metrics."""
    
    start_time = time.perf_counter()
    start_memory = get_memory_usage()
    
    try:
        result = await service_func(*args, **kwargs)
        
        # Success metrics
        record_service_metric(
            service=service_name,
            operation=operation,
            status="success",
            duration_ms=(time.perf_counter() - start_time) * 1000,
            memory_delta_mb=(get_memory_usage() - start_memory) / 1024 / 1024
        )
        
        return result
    
    except Exception as e:
        # Error metrics
        record_service_metric(
            service=service_name,
            operation=operation,
            status="error",
            error_type=type(e).__name__,
            duration_ms=(time.perf_counter() - start_time) * 1000
        )
        raise
```

### Health Monitoring
```python
# Service health checks
async def check_service_health() -> Dict[str, Any]:
    """Comprehensive service health check."""
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {}
    }
    
    # Check individual services
    services_to_check = [
        ("intent", test_intent_classification),
        ("reranker", test_reranker_service),
        ("retrieval", test_retrieval_service),
        ("normalization", test_normalization_service)
    ]
    
    for service_name, test_func in services_to_check:
        try:
            test_result = await asyncio.wait_for(test_func(), timeout=5.0)
            health_status["services"][service_name] = {
                "status": "healthy",
                "response_time": test_result.get("response_time", 0),
                "last_check": time.time()
            }
        except Exception as e:
            health_status["services"][service_name] = {
                "status": "unhealthy",
                "error": str(e),
                "last_check": time.time()
            }
            health_status["status"] = "degraded"
    
    return health_status
```

## Future Enhancement Opportunities

### Advanced Processing Capabilities
- **Multi-Modal Input**: Support for image and document processing
- **Real-Time Learning**: Adaptive models that improve with usage
- **Batch Processing**: Efficient handling of bulk operations
- **Stream Processing**: Real-time data pipeline integration

### Performance Optimization
- **Model Quantization**: Reduced memory footprint for BGE models
- **Async Everywhere**: Full async/await adoption across all services
- **Connection Pooling**: Optimized database and API connections
- **Predictive Caching**: ML-based cache warming strategies

### Integration Enhancements
- **Plugin Architecture**: Dynamic service registration and discovery
- **Message Queues**: Asynchronous service communication
- **Event Sourcing**: Complete audit trail of service operations
- **Microservice Migration**: Containerized service deployment

## Testing and Validation

### Service Testing Strategy
- **Unit Tests**: Individual service function validation
- **Integration Tests**: Service interaction and data flow
- **Performance Tests**: Latency and throughput validation
- **Error Injection**: Resilience and fallback testing

### Quality Assurance
- **Type Safety**: Comprehensive Pydantic model validation
- **Input Sanitization**: Security-focused input processing
- **Output Validation**: Response format and content verification
- **Performance Regression**: Automated performance monitoring