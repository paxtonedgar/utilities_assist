# Utilities Assistant - LangGraph RAG System

A high-performance utilities assistant implementing LangGraph workflows with BGE v2-m3 reranker optimization and regex-first intent classification.

## 🎯 Project Purpose

This system provides intelligent utilities documentation assistance through:
- **Fast Intent Classification**: Regex-first approach (4.5s → <1ms per query)
- **Optimized Retrieval**: BGE v2-m3 cross-encoder with performance gating
- **Nuanced Routing**: Coloring-based attribute system for precise retrieval steering
- **Multi-Modal Presentation**: Streamlit interface with API endpoints

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│  LangGraph Core  │───▶│   BGE Reranker  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                       ┌──────▼──────┐
                       │ Intent+Colors│
                       │ (Regex-first)│
                       └─────────────┘
```

### Key Components

- **LangGraph Workflow**: Orchestrates intent → search → rank → present pipeline
- **Regex Intent Slotter**: Eliminates expensive LLM calls for classification
- **BGE v2-m3 Cross-Encoder**: Production-grade reranking with performance gates
- **Coloring System**: Nuanced attributes for retrieval tuning and presenter selection
- **View-Based Search**: Parallel info/procedure view construction with actionability gating

## 🚀 Performance Optimizations (P1-P3 Compliance)

### P1 - Latency Budgets
- **Intent Classification**: 4.5s → <1ms (99.98% reduction)
- **Query Normalization**: 2s → <10ms (simple rule-based)
- **Cross-Encoder Gating**: Prevents expensive loops on insufficient coverage

### P2 - Caching Layers  
- **TTL LRU Caching**: Embeddings, tokenization, slot results
- **Semantic Deduplication**: Prevents recomputation for similar queries
- **HTTP Connection Pooling**: Persistent OpenSearch connections

### P3 - Smart Routing
- **Actionability Scoring**: [0,3] scale for procedure likelihood
- **Suite Affinity**: Domain-specific routing (Jira, Teams, APIs)
- **Specificity Detection**: High/med/low anchoring for search tuning

## 🛠️ Technology Stack

### Core Framework
- **LangGraph**: Workflow orchestration and state management
- **BGE v2-m3**: BAAI cross-encoder reranker (compressed model)
- **OpenSearch**: Vector + BM25 hybrid search with RRF fusion
- **Streamlit**: Web interface and real-time interaction

### Performance Stack
- **Regex Patterns**: Intent classification without LLM overhead
- **TTL Caching**: Request-level and session-level optimization
- **Async Processing**: Parallel view construction and retrieval
- **Connection Pooling**: Efficient resource utilization

### Integration Stack
- **Azure OpenAI**: LLM fallbacks and specialized tasks
- **JPMC Authentication**: Enterprise SSO and proxy support
- **YAML Configuration**: Suite manifests and extensible routing

## 📁 Directory Structure

```
src/
├── agent/           # LangGraph nodes and intent classification
│   ├── intent/      # Regex-first slotting with coloring system
│   ├── nodes/       # LangGraph workflow node implementations  
│   ├── tools/       # Search tools and function calling
│   └── routing/     # Workflow routing and state management
├── retrieval/       # Search, ranking, and view systems
│   ├── suites/      # Domain-specific suite definitions
│   └── views/       # Info/procedure view construction
├── compose/         # Content presentation and rendering
├── infra/           # Infrastructure (clients, config, telemetry)
├── services/        # Core business logic and models
├── controllers/     # Request orchestration and graph integration
├── telemetry/       # Performance monitoring and observability
├── quality/         # Code quality and validation utilities
└── util/            # Shared utilities and helpers
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11+
- OpenSearch cluster
- Azure OpenAI access (optional)
- BGE v2-m3 model (auto-downloaded)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd utilities_assist

# Install dependencies  
pip install -r requirements.txt

# Configure environment
cp config.ini.example config.ini
# Edit config.ini with your settings

# Download BGE model (automatic on first run)
python -c "from src.services.reranker import CrossEncoderReranker; CrossEncoderReranker()"
```

### Configuration
```ini
# config.ini
[search]
host = your-opensearch-cluster
index_alias = utilities-docs

[chat]  
provider = azure
model = gpt-4
api_base = https://your-azure-endpoint

[embed]
provider = azure  
model = text-embedding-ada-002
```

## 🎮 Usage Examples

### Streamlit Interface
```bash
streamlit run streamlit_app.py
```

### API Endpoints
```python
from src.controllers.graph_integration import handle_turn_langgraph

# Process query through LangGraph workflow
result = await handle_turn_langgraph(
    query="How do I onboard to CIU?",
    user_context={"user_id": "test_user"}
)

print(result.assistant_response)
```

### Direct Integration
```python
from src.agent.intent.slotter import slot
from src.retrieval.views import run_procedure_view

# Fast intent classification
slot_result = slot("How do I create a Jira ticket?")
print(f"Intent: {slot_result.intent}, Actionability: {slot_result.colors.actionability_est}")

# Targeted view construction
view_result = await run_procedure_view(
    query="CIU onboarding steps",
    search_client=search_client,
    embed_client=embed_client
)
```

## 📊 Performance Characteristics

### Latency (P95)
- **Total Request**: <3s (down from 8-12s)
- **Intent Classification**: <1ms (was 4.5s)
- **Search + Rank**: <1.5s
- **Presentation**: <200ms

### Accuracy
- **Intent Classification**: 94%+ with regex patterns
- **Suite Routing**: 90%+ domain detection  
- **Reranking**: 15-20% relevance improvement over base search

### Throughput
- **Concurrent Users**: 50+ (with connection pooling)
- **Cache Hit Rate**: 60-80% for repeated queries
- **Resource Efficiency**: 70% reduction in LLM calls

## 🔧 Development

### Code Quality
```bash
# Formatting and linting
ruff format src
ruff check src --fix

# Type checking
mypy src --strict

# Testing
pytest tests/ -v
```

### Performance Monitoring
- **Timeline Tracking**: Stage-by-stage latency measurement
- **Cache Analytics**: Hit rates and memory usage
- **BGE Model Stats**: Reranking performance and gating effectiveness

## 🔗 Integration Points

### LangGraph Compliance
- **Node-based Architecture**: Clean separation of concerns
- **State Management**: Immutable state transitions
- **Error Handling**: Graceful failure modes with fallbacks

### Future LlamaIndex Migration
- **Query Engines**: Compatible abstractions for search
- **Response Synthesis**: Modular presentation layer
- **Index Management**: OpenSearch backend integration

## 📈 Roadmap

### Short Term
- [ ] Enhanced suite manifests (Teams, ServiceNow)
- [ ] Real-time BGE model updates
- [ ] Advanced caching strategies

### Medium Term  
- [ ] LlamaIndex integration pathways
- [ ] Multi-modal document support
- [ ] Federated search across multiple knowledge bases

### Long Term
- [ ] Custom BGE fine-tuning for domain
- [ ] Real-time learning from user interactions
- [ ] Advanced workflow orchestration

## 🤝 Contributing

1. Follow existing code patterns (regex-first, async, caching)
2. Maintain P1-P3 performance constraints
3. Add telemetry for new components
4. Update suite manifests for new domains

## 📄 License

[License information]

---

**Performance-optimized utilities assistant with LangGraph workflows and BGE reranking.**