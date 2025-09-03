# Agent - LangGraph Workflow Nodes and Intent System

## Purpose
Implements LangGraph workflow nodes for intent classification, search orchestration, and response generation. The agent system replaces expensive LLM-based intent classification with regex-first patterns and provides nuanced coloring attributes for retrieval steering.

## Architecture
Central to the LangGraph workflow, the agent system processes user queries through discrete, stateful nodes:

```
User Query → Intent Node → Search Nodes → Combine Node → Response
              ↓
        Coloring System (actionability, suite affinity, specificity)
```

### LangGraph Integration Pattern
- **Stateful Nodes**: Each node preserves and enhances workflow state
- **Immutable State**: Clean state transitions following LangGraph patterns  
- **Error Resilience**: Graceful fallbacks and error state preservation
- **Parallel Execution**: Concurrent view construction where applicable

## Key Files

### Intent Classification
- `intent/slotter.py` - **Core regex-first intent classifier** (315 lines)
  - Replaces 4.5s LLM calls with <1ms regex patterns
  - Returns `Colors` object with nuanced routing attributes
  - TTL caching for repeated queries
- `intent/coloring/builder.py` - **Coloring system orchestrator**
  - Dependency injection for calculator modules
  - Singleton pattern for performance
  - Error handling with DEFAULT_COLORS fallback

### Coloring System Modules
- `intent/coloring/actionability.py` - **[0,3] actionability scoring**
  - HTTP endpoints → +1.0, ticket verbs → +1.0, steps → +1.0
  - Artifact type detection (endpoint, ticket, form, runbook)
- `intent/coloring/suite_affinity.py` - **Domain-specific routing scores**
  - Reuses existing YAML suite manifests (DRY compliance)
  - URL patterns (+0.8), keywords (+0.2), regex hints (+0.5)
- `intent/coloring/specificity.py` - **Query specificity classification**
  - Distinct anchor counting (project keys, versions, UUIDs)
  - Set-based deduplication prevents inflation
- `intent/coloring/safety.py` - **PII/credential detection**
  - Conservative patterns with false-positive guards
  - Never logs raw content, only flag names

### Workflow Nodes
- `nodes/search_nodes.py` - **LangGraph nodes: IntentNode, SearchNode, CombineNode**
  - IntentNode uses micro-router (regex-first) for routing
  - SearchNode executes retrieval and passes filters/context through
  - CombineNode composes evidence-gated briefings (preferred for Answer)
- `nodes/summarize.py` - **Query normalization node** (rule-based, no LLM)
- `nodes/processing_nodes.py` - **AnswerNode and helpers** (uses final_briefing when present)

### Tools and Routing
- `tools/search.py` - **LangGraph search tools** (445 lines)
  - Wraps retrieval functions for LangGraph integration
  - Cross-encoder gating logic
  - Enhanced RRF with BGE optimization
- `routing/` - **Workflow routing logic**
  - Conditional routing based on intent and colors
  - Error path handling

## Dependencies

### Internal Dependencies
- `src.retrieval.views` - View construction (info_view, procedure_view)
- `src.retrieval.config` - Centralized configuration and thresholds
- `src.services.retrieve` - Core search implementations
- `src.telemetry.logger` - Performance tracking and observability

### External Dependencies
- `langgraph` - Workflow orchestration framework
- `pydantic` - Data validation and serialization
- Built-in `re` module for regex patterns (no external regex libs)

## Integration Points

### LangGraph Compatibility
```python
# Node signature pattern (BaseNodeHandler)
class IntentNode(BaseNodeHandler):
    async def execute(self, state: dict, config: dict | None = None) -> dict:
        # Return a new state dict; preserve existing fields
        return {**state, "next_action": "general", "workflow_path": state.get("workflow_path", []) + ["intent"]}
```

### Future LlamaIndex Integration
- **Query Engines**: Intent classification can drive LlamaIndex query engine selection
- **Response Synthesis**: Coloring attributes can inform LlamaIndex response modes
- **Multi-Modal**: Safety detection ready for document/image content

## Performance Considerations

### Critical Optimizations
- **Regex Compilation**: All patterns compiled once in `patterns.py`
- **Singleton ColorsBuilder**: Avoid repeated initialization overhead
- **TTL Caching**: Slot results cached for 10 minutes
- **Async Patterns**: Parallel view construction where possible

### Performance Metrics
- **Intent Classification**: <1ms (99.98% improvement over LLM)
- **Color Computation**: <2ms target for all attributes
- **Memory Usage**: Minimal - regex patterns + small caches
- **Cache Hit Rate**: 60-80% for repeated patterns

### Bottlenecks
- **Large Query Text**: Regex performance scales with input size
- **Suite Registry**: YAML loading on ColorsBuilder initialization
- **State Size**: Large workflow states impact serialization

## Testing Strategy
- **Doctests**: Inline examples for core functions
- **Golden Tests**: Known input/output pairs for regression testing
- **Property Tests**: Verify regex patterns don't conflict
- **Performance Tests**: Latency budgets and cache effectiveness

## Future Enhancements
- **ONNX Fallback**: Optional ML model for ambiguous cases
- **Dynamic Patterns**: Runtime pattern updates without restarts
- **Advanced Caching**: Semantic similarity-based cache keys
- **Workflow Validation**: Quality gates for LangGraph state consistency
