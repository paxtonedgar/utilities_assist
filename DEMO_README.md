# Utilities Assistant Demo UI

This demo showcases the clean chat interface design with sources, citations, and diagnostics.

## Features Demonstrated

### Core Interface
- **Clean chat input**: Simple text input with submit button
- **Thinking spinner**: Shows "Thinking..." while processing
- **Markdown responses**: Properly formatted answers
- **Source citations**: Clickable chips in format "Title §Section • 2025-07-31"

### Diagnostics Sidebar
- **Index selection**: Toggle between Production and Mock corpus
- **Intent classification**: Shows detected intent with confidence score
- **Top documents**: BM25 and kNN retrieval results
- **Stage latencies**: Breakdown of processing times (Intent, BM25, kNN, Fuse, LLM)
- **Token counts**: Input and output token usage
- **Raw context toggle**: Expandable view of context sent to LLM
- **Warning banners**: Non-blocking alerts for budget overruns or errors

### Source Format Examples
- `Start Service Process (v4) §overview • 2025-01-15`
- `Account Utility API (v3) §api_spec • 2024-11-25`
- `Payment Plan Setup (v2) §eligibility • 2024-11-10`

## Running the Demo

```bash
# Quick start
make run-demo

# Or directly
streamlit run demo_ui.py
```

Open http://localhost:8501 in your browser.

## Sample Queries to Try

- "How do I start utility service?"
- "Show me the Account API documentation"
- "What payment plans are available?"
- "Tell me about safety requirements"

The demo includes realistic mock responses and telemetry data to demonstrate the full interface capabilities.

## Key UI Elements

1. **Input Box**: Multiline text input at bottom
2. **Source Chips**: Clickable citations below each response  
3. **Diagnostics Panel**: Collapsible sidebar with performance metrics
4. **Warning Banners**: Appear above responses when needed
5. **Raw Context**: Toggle to show context passed to LLM
6. **Index Selection**: Switch between production and mock data sources

This demonstrates the complete user experience for enterprise knowledge retrieval with full observability and debugging capabilities.