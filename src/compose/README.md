# Compose - Content Presentation and Rendering

## Purpose
Handles content assembly and presentation rendering with clean separation between data processing and output formatting. Provides testable presentation logic supporting multiple output formats (Streamlit, API, CLI) while maintaining separation from retrieval concerns.

## Architecture
Modular presentation pipeline following separation of concerns:

```
Search Results → Material Assembly → Presenter Selection → Format Rendering
                        ↓                    ↓              ↓
                   Structured Data    Decision Logic    Output Format
```

### Design Philosophy
- **Data/Presentation Separation**: Clear boundary between content and formatting
- **Format Agnostic**: Support multiple output channels without duplication
- **Testable Components**: Pure functions for reliable testing
- **Extensible Rendering**: Easy addition of new output formats

## Key Files

### Core Presentation Logic
- `present.py` - **Main presentation orchestrator**
  - Material assembly from search results
  - Presenter selection based on actionability scores
  - Format-specific rendering coordination
  - Error handling and fallback presentation

### Material Assembly
- Content aggregation from multiple sources
- Citation formatting and source attribution
- Metadata preservation for traceability
- Structured data preparation for rendering

### Presenter Selection
- Actionability-based presenter choice
- Procedure vs info presenter logic
- Fallback handling for edge cases
- Suite-specific presentation rules

## Dependencies

### Internal Dependencies
- `src.retrieval.actionability` - Span extraction and scoring
- `src.services.models` - Data models and structured types
- `src.agent.intent.slotter` - Intent and coloring information

### External Dependencies
- `jinja2` - Template rendering for structured output
- `markdown` - Markdown processing for rich text
- `typing` - Type hints for presentation contracts

## Integration Points

### LangGraph Node Integration
```python
# Presentation as final workflow step
async def present_node(state: GraphState, config: RunnableConfig) -> GraphState:
    search_results = state.get("search_results", [])
    intent_result = state.get("intent")
    
    # Assemble material
    material = assemble_material(search_results, intent_result)
    
    # Select presenter
    presenter = select_presenter(material, intent_result.colors)
    
    # Render output
    rendered = render_presentation(material, presenter, format="streamlit")
    
    return {**state, "final_answer": rendered}
```

### LlamaIndex Response Synthesis Integration
```python
# Current: Custom presenter selection
decision = choose_presenter(info_view, procedure_view)
material = materialize(decision)
rendered = render(material)

# Future LlamaIndex path
from llama_index.core.response_synthesizers import ResponseSynthesizer
synthesizer = ResponseSynthesizer(
    response_mode=get_response_mode(colors.actionability_est),
    service_context=service_context
)
response = synthesizer.synthesize(query, nodes)
```

## Performance Considerations

### Rendering Efficiency
- **Template Caching**: Compiled Jinja templates cached in memory
- **Lazy Rendering**: Content rendered only when requested
- **Format Streaming**: Large outputs streamed to prevent memory buildup
- **Citation Deduplication**: Efficient source reference handling

### Memory Management
- **Structured Assembly**: Avoid string concatenation for large content
- **Reference Tracking**: Weak references for large material objects
- **Format-Specific Optimization**: Tailored rendering for each output type
- **Error Boundary**: Graceful handling of rendering failures

## Current Implementation Details

### Material Assembly Pattern
```python
@dataclass
class PresentationMaterial:
    title: str
    content: List[ContentBlock]
    citations: List[Citation]
    metadata: Dict[str, Any]
    presenter_type: str

def assemble_material(search_results, intent_result):
    # Extract actionable spans
    spans = extract_actionable_spans(search_results)
    
    # Build content blocks
    content_blocks = []
    for span in spans:
        block = ContentBlock(
            type=span.type,
            content=span.text,
            source=span.source,
            confidence=span.confidence
        )
        content_blocks.append(block)
    
    return PresentationMaterial(
        title=generate_title(intent_result.intent),
        content=content_blocks,
        citations=extract_citations(search_results),
        metadata=build_metadata(intent_result),
        presenter_type=determine_presenter(spans, intent_result.colors)
    )
```

### Presenter Selection Logic
```python
def select_presenter(material, colors):
    """Select appropriate presenter based on actionability and content type."""
    actionable_score = calculate_actionable_score(material.content)
    
    if actionable_score >= 2.0 and colors.actionability_est >= 1.5:
        return "procedure"  # Step-by-step guidance
    elif len(material.content) > 0 and any(b.type == "definition" for b in material.content):
        return "info"  # Informational content
    else:
        return "fallback"  # Basic listing
```

### Format-Specific Rendering
```python
def render_presentation(material, presenter, format="streamlit"):
    """Render material using specified presenter and format."""
    
    if format == "streamlit":
        return render_streamlit(material, presenter)
    elif format == "api":
        return render_api_response(material, presenter)
    elif format == "markdown":
        return render_markdown(material, presenter)
    else:
        raise ValueError(f"Unsupported format: {format}")

def render_streamlit(material, presenter):
    """Streamlit-specific rendering with interactive elements."""
    if presenter == "procedure":
        return render_procedure_steps(material)
    elif presenter == "info":
        return render_info_cards(material)
    else:
        return render_fallback_list(material)
```

## Testing Strategy

### Unit Testing
- **Pure Functions**: All rendering functions are pure and easily testable
- **Mock Material**: Standardized test fixtures for different content types
- **Format Validation**: Output format compliance testing
- **Edge Cases**: Error handling and malformed input testing

### Integration Testing
- **End-to-End**: Full pipeline from search results to rendered output
- **Format Compatibility**: Cross-format consistency validation
- **Performance Testing**: Rendering latency under various loads
- **Accessibility**: Output format accessibility compliance

## Future Enhancement Opportunities

### Multi-Modal Support
- **Rich Media**: Image and video content integration
- **Interactive Elements**: Dynamic components for complex procedures
- **Progressive Disclosure**: Expandable content sections
- **Responsive Design**: Adaptive layouts for different screen sizes

### Advanced Formatting
- **Custom Templates**: User-defined presentation templates
- **Brand Compliance**: Corporate styling and formatting rules
- **Internationalization**: Multi-language content support
- **Accessibility**: WCAG compliance for all output formats

### LlamaIndex Integration
```python
# Response synthesis with LlamaIndex patterns
from llama_index.core.response_synthesizers import (
    TreeSummarize,
    CompactAndRefine,
    Accumulate
)

class AdaptiveResponseSynthesizer:
    def __init__(self):
        self.synthesizers = {
            "procedure": TreeSummarize(),  # Step-by-step organization
            "info": CompactAndRefine(),    # Comprehensive information
            "fallback": Accumulate()       # Simple aggregation
        }
    
    def synthesize(self, query, nodes, presenter_type):
        synthesizer = self.synthesizers[presenter_type]
        return synthesizer.synthesize(query, nodes)
```

### Performance Optimization
- **Streaming Rendering**: Real-time output for long content
- **Parallel Processing**: Concurrent rendering of different sections
- **Caching Strategies**: Intelligent caching of rendered content
- **Compression**: Output compression for network efficiency

## Error Handling and Resilience

### Graceful Degradation
```python
def safe_render(material, presenter, format):
    """Render with graceful fallback on errors."""
    try:
        return render_presentation(material, presenter, format)
    except TemplateError as e:
        logger.warning(f"Template error: {e}, falling back to simple format")
        return render_simple_fallback(material)
    except Exception as e:
        logger.error(f"Rendering failed: {e}")
        return render_error_message(material, str(e))
```

### Validation and Sanitization
- **Input Validation**: Material structure validation before rendering
- **Content Sanitization**: XSS prevention for web outputs
- **Citation Validation**: Source URL and reference validation
- **Format Compliance**: Output format schema validation