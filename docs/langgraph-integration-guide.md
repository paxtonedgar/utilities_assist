# LangGraph Integration Guide

## Overview
This guide explains how to integrate and use the new LangGraph-based multi-agent workflow system in the utilities assistant application.

## Architecture Components

### 1. Workflow State (`workflows/state.py`)
- **WorkflowState**: Main state container passed between agents
- **WorkflowConfig**: Configuration parameters for workflow behavior
- **Utility Functions**: State management helpers

### 2. Workflow Agents (`workflows/agents.py`)
- **Query Decomposer Agent**: Analyzes query complexity and decomposes complex queries
- **Search Orchestrator Agent**: Coordinates parallel search operations
- **Single Search Agent**: Performs optimized search for simple queries
- **Parallel Search Agent**: Handles individual searches in multi-search workflows
- **Result Synthesizer Agent**: Intelligently combines results from multiple sources

### 3. Response Generation (`workflows/response_generator.py`)
- **Response Generator Agent**: Creates final answers with multi-source context awareness

### 4. Main Workflow (`workflows/langgraph_workflow.py`)
- **LangGraphWorkflow**: Main workflow orchestrator using LangGraph StateGraph
- **Stream Interface**: Compatible with existing turn controller interface

### 5. Integration Controller (`controllers/langgraph_controller.py`)
- **Backward Compatibility**: Drop-in replacement for existing turn controller
- **Smart Routing**: Automatically detects when to use LangGraph vs traditional processing
- **Performance Monitoring**: Tracks metrics for both workflow types

## Integration Patterns

### Pattern 1: Gradual Rollout (Recommended)
Use the LangGraph controller manager for controlled rollout:

```python
from controllers.langgraph_controller import langgraph_manager

# Enable LangGraph for testing
langgraph_manager.enable_langgraph()

# Use enhanced turn handler
async for update in handle_turn_enhanced(user_input, resources):
    process_update(update)
```

### Pattern 2: Explicit Control
Force specific workflow usage:

```python
from controllers.langgraph_controller import handle_turn_langgraph

# Force LangGraph workflow
async for update in handle_turn_langgraph(
    user_input=user_input, 
    resources=resources,
    enable_langgraph=True
):
    process_update(update)

# Force traditional workflow  
async for update in handle_turn_langgraph(
    user_input=user_input,
    resources=resources, 
    enable_langgraph=False
):
    process_update(update)
```

### Pattern 3: Direct Workflow Usage
Use the workflow directly for maximum control:

```python
from workflows.langgraph_workflow import create_langgraph_workflow
from workflows.state import WorkflowConfig

# Create custom configuration
config = WorkflowConfig(
    max_parallel_searches=5,
    diversity_lambda=0.8,
    enable_streaming=True
)

# Create and use workflow
workflow = create_langgraph_workflow(config)
async for update in workflow.stream(user_input, resources):
    process_update(update)
```

## Configuration Options

### Workflow Configuration
```python
from workflows.state import WorkflowConfig

config = WorkflowConfig(
    # Query complexity thresholds
    simple_query_threshold=0.3,      # Below this = simple query
    complex_query_threshold=0.7,     # Above this = complex query
    
    # Search parameters
    max_parallel_searches=3,         # Max concurrent searches
    search_timeout_seconds=30,       # Search timeout
    enable_cross_index_search=True,  # Enable multi-index searches
    
    # Result synthesis
    max_context_length=8000,         # Max context for LLM
    diversity_lambda=0.75,           # MMR diversification parameter
    
    # Response generation
    enable_streaming=True,           # Stream response chunks
    chunk_size=50,                   # Characters per chunk
    
    # Performance
    enable_caching=True,             # Cache intermediate results
    cache_ttl_seconds=300            # Cache TTL
)
```

## Query Types and Routing

### Automatic LangGraph Detection
The system automatically uses LangGraph for:

1. **Comparative Queries**
   - "Compare X and Y"
   - "What's the difference between A and B?"
   - "X versus Y"

2. **Multi-Part Questions**
   - "What is X and also how do I configure Y?"
   - "List all utilities and their rate limits"

3. **Complex Procedural Queries**
   - Long queries (>15 words)
   - Multiple question words
   - Complex API documentation requests

4. **List-Based Queries**
   - "List all authentication methods"
   - "Show me all utilities that support OAuth"

### Manual Override
```python
# Force LangGraph for any query
async for update in langgraph_manager.handle_turn(
    user_input=query,
    force_langgraph=True
):
    process_update(update)

# Force traditional processing
async for update in langgraph_manager.handle_turn(
    user_input=query,
    force_langgraph=False
):
    process_update(update)
```

## Performance Monitoring

### Getting Performance Stats
```python
from controllers.langgraph_controller import langgraph_manager

stats = langgraph_manager.get_performance_stats()
print(f"LangGraph avg time: {stats.get('langgraph', {}).get('avg_time', 0)}ms")
print(f"Traditional avg time: {stats.get('traditional', {}).get('avg_time', 0)}ms")
```

### Workflow-Level Metrics
Each workflow execution provides detailed metrics:
```python
async for update in workflow.stream(user_input, resources):
    if update.get("type") == "complete":
        metrics = update["result"]["metrics"]
        print(f"Query analysis: {metrics.get('query_analysis_ms', 0)}ms")
        print(f"Search time: {metrics.get('single_search_ms', 0)}ms") 
        print(f"Synthesis time: {metrics.get('synthesis_ms', 0)}ms")
```

## Integration with Existing Streamlit App

### Option 1: Replace in turn_controller.py
```python
# In turn_controller.py, replace the handle_turn function
from controllers.langgraph_controller import handle_turn_enhanced as handle_turn
```

### Option 2: Feature Flag in Chat Interface
```python
# In chat_interface.py
from controllers.langgraph_controller import handle_turn_langgraph

# Add feature toggle
use_langgraph = st.session_state.get("use_langgraph", False)

if st.button(f"LangGraph: {'ON' if use_langgraph else 'OFF'}"):
    st.session_state.use_langgraph = not use_langgraph

# Use appropriate handler
async for chunk in handle_turn_langgraph(
    user_input,
    st.session_state.resources,
    chat_history=[],
    use_mock_corpus=st.session_state.use_mock_corpus,
    enable_langgraph=use_langgraph
):
    # Process chunk
```

## Testing and Validation

### Unit Testing
Run the comprehensive test suite:
```bash
python test_langgraph_components.py
```

### Integration Testing
Test with real queries:
```python
from controllers.langgraph_controller import langgraph_manager
from infra.resource_manager import get_resources

# Enable LangGraph
langgraph_manager.enable_langgraph()

# Test comparative query
test_queries = [
    "Compare Customer Summary Utility and Account Utility login APIs",
    "What authentication methods are available and which is most secure?",
    "List all utilities that support OAuth 2.0"
]

resources = get_resources()
for query in test_queries:
    print(f"Testing: {query}")
    async for update in langgraph_manager.handle_turn(query, resources):
        if update.get("type") == "complete":
            print(f"Answer: {update['result']['answer'][:100]}...")
            break
```

### Performance Comparison
```python
# Test both workflows on the same queries
test_query = "Compare login APIs for different utilities"

# Traditional workflow
start_time = time.time()
async for update in langgraph_manager.handle_turn(
    test_query, resources, force_langgraph=False
):
    if update.get("type") == "complete":
        traditional_time = time.time() - start_time
        break

# LangGraph workflow  
start_time = time.time()
async for update in langgraph_manager.handle_turn(
    test_query, resources, force_langgraph=True
):
    if update.get("type") == "complete":
        langgraph_time = time.time() - start_time
        break

print(f"Traditional: {traditional_time:.2f}s")
print(f"LangGraph: {langgraph_time:.2f}s")
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```python
   # Make sure LangGraph is installed
   pip install langgraph==0.6.0
   ```

2. **State Management Issues**
   ```python
   # Ensure state initialization is correct
   from workflows.state import initialize_workflow_state
   state = initialize_workflow_state(query, request_id, user_context)
   ```

3. **Resource Injection**
   ```python
   # Make sure resources are properly injected
   if not resources:
       resources = get_resources()
   ```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.getLogger("workflows").setLevel(logging.DEBUG)
logging.getLogger("controllers.langgraph_controller").setLevel(logging.DEBUG)
```

### Fallback Behavior
The system automatically falls back to traditional processing if LangGraph fails:
- Errors in workflow execution
- Missing dependencies
- Configuration issues

## Migration Strategy

### Phase 1: Setup and Testing
1. Install dependencies: `pip install langgraph==0.6.0`
2. Run unit tests: `python test_langgraph_components.py`
3. Test with sample queries

### Phase 2: Gradual Rollout
1. Enable LangGraph for specific query types
2. Monitor performance and quality
3. Collect user feedback

### Phase 3: Full Integration
1. Replace default turn controller
2. Configure optimal parameters
3. Remove traditional fallback (optional)

### Phase 4: Advanced Features
1. Add human-in-the-loop capabilities
2. Implement conversation memory
3. Add custom workflow templates

## Best Practices

1. **Start with Conservative Settings**
   - Use default thresholds initially
   - Gradually tune parameters based on performance

2. **Monitor Both Quality and Performance**
   - Track response quality metrics
   - Compare latency between workflows
   - Monitor resource usage

3. **Maintain Backward Compatibility**
   - Keep traditional workflow as fallback
   - Use feature flags for controlled rollout

4. **Test Thoroughly**
   - Test edge cases and error conditions
   - Validate with diverse query types
   - Performance test under load

This integration provides a sophisticated multi-agent system while maintaining compatibility with the existing Phase 1 performance optimizations and architecture.