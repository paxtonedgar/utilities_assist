# Controllers - Request Orchestration and Graph Integration

## Purpose
Orchestrates request processing by integrating LangGraph workflows with the broader system architecture. Handles the coordination between user interfaces, workflow execution, and response formatting while maintaining enterprise-grade error handling and performance monitoring.

## Architecture
Request orchestration layer that bridges user interfaces with LangGraph workflows:

```
User Interface (Streamlit/API)
         ↓
   Controllers Layer
         ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Graph Workflow  │ Resource Mgmt   │ Error Handling  │
│   Execution     │  & Auth        │ & Monitoring    │
└─────────────────┴─────────────────┴─────────────────┘
         ↓
  LangGraph Workflow
```

### Design Principles
- **Single Entry Point**: Centralized request processing for all interfaces
- **Error Resilience**: Comprehensive error handling with graceful degradation
- **Performance Monitoring**: Built-in telemetry and performance tracking
- **Resource Management**: Efficient resource allocation and cleanup

## Key Files

### Core Orchestration
- `graph_integration.py` - **Primary LangGraph workflow controller** (628 lines)
  - Request preprocessing and validation
  - LangGraph workflow execution and monitoring
  - Response post-processing and formatting
  - Error handling and recovery mechanisms
  - Performance metrics collection and reporting

### Request Flow Management
- User request validation and sanitization
- Resource allocation and authentication handling
- Workflow state management and persistence
- Response assembly and format conversion
- Cleanup and resource deallocation

## Dependencies

### Internal Dependencies
- `src.agent.graph` - LangGraph workflow definitions
- `src.infra.resource_manager` - Infrastructure resource access
- `src.infra.persistence` - LangGraph persistence and user context
- `src.telemetry.logger` - Performance tracking and observability
- `src.services.models` - Data models and response structures

### External Dependencies
- `langgraph` - Workflow execution framework
- `asyncio` - Asynchronous operation coordination
- `pydantic` - Request/response validation
- `typing` - Type safety and validation

## Integration Points

### LangGraph Workflow Execution
```python
async def handle_turn_langgraph(
    query: str,
    user_context: Dict[str, Any],
    stream_updates: bool = False
) -> TurnResult:
    """Main entry point for LangGraph workflow execution."""
    
    # Initialize resources and authentication
    resources = get_resources()
    
    # Extract user context for enterprise environments
    enhanced_context = extract_user_context(resources)
    enhanced_context.update(user_context)
    
    # Generate thread ID for conversation persistence
    thread_id = generate_thread_id(
        enhanced_context.get("user_id", "anonymous"),
        enhanced_context.get("session_metadata")
    )
    
    # Create LangGraph configuration
    config = create_langgraph_config(enhanced_context, thread_id)
    
    try:
        # Execute workflow with monitoring
        with performance_monitor("langgraph_execution"):
            result = await execute_workflow_with_checkpointing(
                query=query,
                config=config,
                resources=resources,
                stream_updates=stream_updates
            )
        
        # Process and validate response
        return format_turn_result(result, enhanced_context)
        
    except Exception as e:
        # Comprehensive error handling
        return handle_workflow_error(e, query, enhanced_context)
```

### Resource Management Integration
```python
async def execute_workflow_with_checkpointing(
    query: str,
    config: Dict[str, Any],
    resources: RAGResources,
    stream_updates: bool = False
) -> Dict[str, Any]:
    """Execute workflow with proper resource management."""
    
    # Initialize workflow with persistence
    checkpointer, store = get_checkpointer_and_store()
    
    if checkpointer:
        # Persistent workflow execution
        workflow = create_workflow_with_persistence(
            checkpointer=checkpointer,
            store=store
        )
    else:
        # In-memory workflow execution
        workflow = create_basic_workflow()
    
    # Prepare initial state
    initial_state = {
        "original_query": query,
        "resources": resources,
        "user_context": config.get("configurable", {}),
        "workflow_path": [],
        "performance_metrics": {}
    }
    
    # Execute with streaming support
    if stream_updates:
        return await execute_streaming_workflow(workflow, initial_state, config)
    else:
        return await workflow.ainvoke(initial_state, config=config)
```

### Error Handling and Recovery
```python
def handle_workflow_error(
    error: Exception,
    query: str,
    user_context: Dict[str, Any]
) -> TurnResult:
    """Comprehensive error handling with user-friendly responses."""
    
    # Log error with context
    logger.error(f"Workflow execution failed: {error}", extra={
        "query": query[:100],  # Truncate for logging
        "user_id": user_context.get("user_id"),
        "error_type": type(error).__name__
    })
    
    # Generate user-friendly error response
    if isinstance(error, TimeoutError):
        return create_timeout_response(query)
    elif isinstance(error, ValidationError):
        return create_validation_error_response(query, error)
    elif isinstance(error, AuthenticationError):
        return create_auth_error_response()
    else:
        return create_generic_error_response(query)

def create_timeout_response(query: str) -> TurnResult:
    """Create response for timeout errors."""
    return TurnResult(
        assistant_response=(
            "Your request is taking longer than expected. "
            "Please try a simpler query or check back later."
        ),
        sources=[],
        performance_metrics={"error_type": "timeout"},
        user_feedback_enabled=True,
        suggestions=[
            f"Try: What is {extract_key_terms(query)[0]}?",
            "Try: List available utilities",
            "Try: Help with common tasks"
        ]
    )
```

## Performance Considerations

### Request Processing Optimization
- **Resource Pooling**: Efficient reuse of expensive resources
- **Connection Management**: Persistent connections for external services
- **Caching Integration**: Multi-level caching for repeated requests
- **Async Coordination**: Proper async/await patterns for I/O operations

### Memory Management
- **State Cleanup**: Automatic cleanup of workflow state after completion
- **Resource Limits**: Configurable limits on request size and complexity
- **Memory Monitoring**: Built-in memory usage tracking
- **Garbage Collection**: Explicit cleanup of large objects

### Scalability Features
- **Connection Pooling**: Database and API connection management
- **Load Balancing**: Request distribution across multiple workers
- **Circuit Breakers**: Automatic fallback for failing services
- **Rate Limiting**: Request throttling for fair resource usage

## Current Implementation Details

### Request Validation and Preprocessing
```python
def validate_and_preprocess_request(
    query: str,
    user_context: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Validate and preprocess incoming requests."""
    
    # Input validation
    if not query or len(query.strip()) == 0:
        raise ValidationError("Query cannot be empty")
    
    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(f"Query too long (max {MAX_QUERY_LENGTH} characters)")
    
    # Sanitization
    sanitized_query = sanitize_user_input(query)
    
    # Context validation
    validated_context = validate_user_context(user_context)
    
    # Security checks
    perform_security_checks(sanitized_query, validated_context)
    
    return sanitized_query, validated_context

def sanitize_user_input(query: str) -> str:
    """Sanitize user input to prevent injection attacks."""
    
    # Remove potential script injection
    query = re.sub(r'<script.*?</script>', '', query, flags=re.IGNORECASE)
    
    # Limit special characters
    query = re.sub(r'[^\w\s\-\.\?\!\,\:\;\/\(\)]', '', query)
    
    # Normalize whitespace
    query = ' '.join(query.split())
    
    return query.strip()
```

### Response Formatting and Assembly
```python
def format_turn_result(
    workflow_result: Dict[str, Any],
    user_context: Dict[str, Any]
) -> TurnResult:
    """Format workflow result into standardized response."""
    
    # Extract core response components
    assistant_response = workflow_result.get("final_answer", "")
    search_results = workflow_result.get("search_results", [])
    performance_metrics = workflow_result.get("performance_metrics", {})
    
    # Convert search results to source citations
    sources = [
        SourceChip(
            title=result.title,
            url=result.url,
            snippet=result.content[:200] + "..." if len(result.content) > 200 else result.content,
            confidence=getattr(result, 'score', 0.0)
        )
        for result in search_results[:5]  # Limit to top 5 sources
    ]
    
    # Add performance context
    enhanced_metrics = {
        **performance_metrics,
        "total_sources": len(search_results),
        "response_length": len(assistant_response),
        "user_context": {
            "user_id": user_context.get("user_id"),
            "session_length": user_context.get("session_length", 1)
        }
    }
    
    return TurnResult(
        assistant_response=assistant_response,
        sources=sources,
        performance_metrics=enhanced_metrics,
        user_feedback_enabled=True,
        conversation_state=workflow_result.get("conversation_state", {})
    )
```

### Streaming Response Handling
```python
async def execute_streaming_workflow(
    workflow: CompiledGraph,
    initial_state: Dict[str, Any],
    config: Dict[str, Any]
) -> AsyncGenerator[StreamingChunk, None]:
    """Execute workflow with real-time streaming updates."""
    
    # Stream workflow execution
    async for event in workflow.astream(initial_state, config=config):
        # Process different event types
        if event.get("type") == "node_start":
            yield StreamingChunk(
                type="status",
                content=f"Processing: {event['node']}",
                metadata={"node": event["node"], "timestamp": time.time()}
            )
        
        elif event.get("type") == "node_end":
            # Stream partial results if available
            if "partial_response" in event.get("data", {}):
                yield StreamingChunk(
                    type="partial_content",
                    content=event["data"]["partial_response"],
                    metadata={"node": event["node"]}
                )
        
        elif event.get("type") == "error":
            yield StreamingChunk(
                type="error",
                content="An error occurred during processing",
                metadata={"error": str(event.get("error", "Unknown error"))}
            )
    
    # Final result
    yield StreamingChunk(
        type="complete",
        content="",
        metadata={"status": "completed", "timestamp": time.time()}
    )
```

## Error Handling Strategies

### Exception Classification
```python
class WorkflowError(Exception):
    """Base class for workflow-related errors."""
    pass

class ValidationError(WorkflowError):
    """Input validation failures."""
    pass

class ResourceError(WorkflowError):
    """Resource allocation or access failures."""
    pass

class TimeoutError(WorkflowError):
    """Operation timeout failures."""
    pass

class AuthenticationError(WorkflowError):
    """Authentication or authorization failures."""
    pass

def classify_and_handle_error(error: Exception) -> TurnResult:
    """Classify error and generate appropriate response."""
    
    error_handlers = {
        ValidationError: handle_validation_error,
        ResourceError: handle_resource_error,
        TimeoutError: handle_timeout_error,
        AuthenticationError: handle_auth_error,
        Exception: handle_generic_error  # Catch-all
    }
    
    for error_type, handler in error_handlers.items():
        if isinstance(error, error_type):
            return handler(error)
    
    return handle_generic_error(error)
```

### Recovery Mechanisms
```python
async def execute_with_retry(
    operation: Callable,
    max_retries: int = 3,
    backoff_factor: float = 1.5
) -> Any:
    """Execute operation with exponential backoff retry."""
    
    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except Exception as e:
            if attempt == max_retries:
                raise e
            
            # Calculate backoff delay
            delay = backoff_factor ** attempt
            logger.warning(f"Operation failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
            
            await asyncio.sleep(delay)
    
    raise Exception("Max retries exceeded")
```

## Security and Authentication

### Request Security
```python
def perform_security_checks(
    query: str,
    user_context: Dict[str, Any]
) -> None:
    """Perform security validation on requests."""
    
    # Check for injection patterns
    injection_patterns = [
        r'<script.*?>',
        r'javascript:',
        r'data:text/html',
        r'eval\s*\(',
        r'exec\s*\('
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValidationError("Potentially malicious input detected")
    
    # Rate limiting check
    user_id = user_context.get("user_id", "anonymous")
    if not check_rate_limit(user_id):
        raise ValidationError("Rate limit exceeded")
    
    # Content policy validation
    if not validate_content_policy(query):
        raise ValidationError("Content violates usage policy")
```

### Authentication Integration
```python
def validate_user_context(user_context: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and enhance user authentication context."""
    
    # Extract authentication info
    user_id = user_context.get("user_id")
    session_token = user_context.get("session_token")
    
    # Validate authentication
    if not validate_session_token(user_id, session_token):
        raise AuthenticationError("Invalid or expired session")
    
    # Enhance context with permissions
    user_permissions = get_user_permissions(user_id)
    
    return {
        **user_context,
        "permissions": user_permissions,
        "validated_at": time.time()
    }
```

## Monitoring and Observability

### Performance Metrics Collection
```python
@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for performance monitoring."""
    
    start_time = time.perf_counter()
    start_memory = get_memory_usage()
    
    try:
        yield
    finally:
        end_time = time.perf_counter()
        end_memory = get_memory_usage()
        
        # Record metrics
        metrics = {
            "operation": operation_name,
            "duration_ms": (end_time - start_time) * 1000,
            "memory_delta_mb": (end_memory - start_memory) / 1024 / 1024,
            "timestamp": time.time()
        }
        
        # Log performance data
        logger.info(f"Performance: {operation_name}", extra=metrics)
        
        # Store in metrics system
        record_performance_metric(metrics)
```

### Health Monitoring
```python
async def health_check() -> Dict[str, Any]:
    """Comprehensive system health check."""
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {}
    }
    
    # Check LangGraph workflow
    try:
        test_result = await execute_test_workflow()
        health_status["checks"]["workflow"] = "healthy"
    except Exception as e:
        health_status["checks"]["workflow"] = f"unhealthy: {e}"
        health_status["status"] = "degraded"
    
    # Check resource manager
    try:
        resources = get_resources()
        health_status["checks"]["resources"] = "healthy"
    except Exception as e:
        health_status["checks"]["resources"] = f"unhealthy: {e}"
        health_status["status"] = "degraded"
    
    # Check external dependencies
    for service in ["opensearch", "azure_openai"]:
        try:
            await check_service_health(service)
            health_status["checks"][service] = "healthy"
        except Exception as e:
            health_status["checks"][service] = f"unhealthy: {e}"
            health_status["status"] = "degraded"
    
    return health_status
```

## Future Enhancement Opportunities

### Advanced Orchestration
- **Workflow Composition**: Dynamic workflow assembly based on query complexity
- **Parallel Processing**: Concurrent execution of independent workflow branches
- **Adaptive Routing**: ML-based routing to optimal workflow paths
- **Resource Optimization**: Dynamic resource allocation based on load

### Enterprise Features
- **Multi-tenancy**: Tenant isolation and resource partitioning
- **Compliance Logging**: Comprehensive audit trails for regulatory compliance
- **Custom Workflows**: User-defined workflow templates
- **Integration APIs**: RESTful APIs for external system integration

### Performance Optimization
- **Predictive Caching**: ML-based cache warming for anticipated queries
- **Load Balancing**: Intelligent load distribution across workflow instances
- **Resource Pooling**: Advanced connection and resource pooling strategies
- **Streaming Optimization**: Enhanced real-time response streaming