# Infra - Infrastructure and Configuration Management

## Purpose
Provides enterprise-grade infrastructure components including client factories, configuration management, authentication, and telemetry systems. Designed for production deployment with JPMC proxy support, Azure integration, and comprehensive observability.

## Architecture
Centralized infrastructure layer following dependency injection patterns:

```
Application Layer
       ↓
Resource Manager (Singleton)
       ↓
┌─────────────┬─────────────┬─────────────┐
│   Clients   │   Config    │ Telemetry   │
│ (LRU Cache) │ (Settings)  │ (OpenTel)   │
└─────────────┴─────────────┴─────────────┘
```

### Design Principles
- **No Singletons for Business Logic**: Only infrastructure components
- **LRU Caching**: Efficient client connection pooling
- **Enterprise Authentication**: Azure AD + JPMC proxy support
- **Observability-First**: Comprehensive telemetry and monitoring

## Key Files

### Client Management
- `clients.py` - **Client factory with LRU caching** (403 lines)
  - Azure OpenAI client creation with authentication
  - OpenSearch session management with retries
  - JPMC proxy configuration and connection pooling
  - Token-based authentication with certificate support
- `resource_manager.py` - **Centralized resource access** (312 lines)
  - Singleton pattern for infrastructure resources
  - Lazy initialization and dependency injection
  - Resource lifecycle management
- `opensearch_client.py` - **OpenSearch integration** (991 lines)
  - High-level search operations (BM25, kNN, hybrid)
  - Index management and mapping operations
  - Error handling and retry logic with exponential backoff

### Configuration System
- `config.py` - **Configuration management** (419 lines)
  - Pydantic-based settings with validation
  - Environment-specific configurations
  - Secret management and secure defaults
- `settings.py` - **Settings loader and validation** (536 lines)
  - INI file parsing with overrides
  - Environment variable integration
  - Type-safe configuration objects
- `search_config.py` - **Search-specific configuration** (419 lines)
  - Index configuration and mapping definitions
  - Intent-based filtering rules
  - Performance tuning parameters

### Authentication & Security
- `azure_auth.py` - **Azure AD integration**
  - Certificate-based authentication
  - Token refresh and lifecycle management
  - JPMC-specific authentication flows
- `persistence.py` - **LangGraph persistence management** (252 lines)
  - Checkpointer and store configuration
  - User context extraction for enterprise environments
  - Thread ID generation and session management

### Observability
- `telemetry.py` - **OpenTelemetry integration** (472 lines)
  - Distributed tracing configuration
  - Metrics collection and export
  - Custom instrumentation for RAG workflows
  - Performance hotspot identification

## Dependencies

### Internal Dependencies
- `src.retrieval.config` - Performance budgets and thresholds
- `src.util.cache` - TTL caching utilities
- `src.telemetry.logger` - Application-level logging

### External Dependencies
- `pydantic` - Configuration validation and serialization
- `requests` - HTTP client with connection pooling
- `boto3` - AWS authentication for JPMC OpenSearch
- `opentelemetry` - Distributed tracing and metrics
- `langgraph` - Persistence layer integration

## Integration Points

### LangGraph Persistence
```python
# Checkpointer and store initialization
checkpointer, store = get_checkpointer_and_store()

# User context extraction
user_context = extract_user_context(resources)
thread_id = generate_thread_id(user_context["user_id"])

# LangGraph configuration
config = create_langgraph_config(user_context, thread_id)
```

### Enterprise Authentication
```python
# Azure client with certificate authentication
def make_chat_client(cfg: ChatCfg, token_provider: Callable = None):
    if cfg.provider == "azure":
        return _create_azure_client(
            "chat", cfg.api_version, cfg.api_base, token_provider
        )
    return _cached_chat_client(cfg.provider, cfg.model, ...)

# JPMC proxy configuration
def _setup_jpmc_proxy():
    if os.getenv("CLOUD_PROFILE") == "jpmc_azure":
        os.environ["https_proxy"] = "proxy.jpmchase.net:10443"
```

### Future LlamaIndex Integration
- **Service Context**: Replace custom resource manager with LlamaIndex ServiceContext
- **Vector Store Integration**: Abstract OpenSearch client behind LlamaIndex VectorStore
- **Callback Handling**: Use LlamaIndex callback system for observability

## Performance Considerations

### Connection Management
- **LRU Caching**: Minimal cache sizes (2 entries) for active connections
- **Connection Pooling**: HTTP session reuse with keep-alive
- **Retry Logic**: Exponential backoff with jitter for resilience
- **Timeout Configuration**: Separate connect and read timeouts

### Memory Efficiency
- **Lazy Loading**: Resources initialized on first access
- **Cache Cleanup**: TTL-based eviction for expired entries
- **Secret Management**: Secure handling without logging sensitive data
- **Configuration Sharing**: Single configuration instance across application

### Enterprise Scalability
- **Multi-tenancy**: User context isolation and resource partitioning
- **Proxy Support**: JPMC proxy routing with no_proxy exceptions
- **Health Checks**: Built-in health monitoring for external dependencies
- **Circuit Breakers**: Graceful degradation on service failures

## Configuration Examples

### Basic Setup
```ini
# config.ini
[search]
host = https://opensearch.company.com
index_alias = utilities-docs
username = search_user
timeout_s = 5.0

[chat]
provider = azure
model = gpt-4
api_base = https://openai.company.com
api_version = 2024-06-01

[embed]
provider = azure
model = text-embedding-ada-002
dims = 1536
```

### Enterprise JPMC Setup
```ini
[search]
host = https://opensearch.jpmchase.net
use_aws_auth = true
timeout_s = 10.0

[azure_openai]
api_key = your-api-key-here

[environment]
cloud_profile = jpmc_azure
jpmc_user_sid = ${JPMC_USER_SID}
```

## Error Handling Patterns

### Graceful Degradation
```python
# Client factory with fallbacks
try:
    client = make_azure_client(config, token_provider)
except AuthenticationError:
    logger.warning("Certificate auth failed, using API key only")
    client = make_azure_client(config, token_provider=None)
except Exception:
    logger.error("Azure client failed, falling back to cache")
    return cached_client_or_none()
```

### Circuit Breaker Pattern
```python
# OpenSearch client with health checks
class OpenSearchClient:
    def __init__(self):
        self._health_check_cache = {}
        self._circuit_breaker = CircuitBreaker()
    
    async def search(self, query):
        if self._circuit_breaker.is_open():
            raise ServiceUnavailableError("Circuit breaker open")
        
        try:
            return await self._perform_search(query)
        except Exception as e:
            self._circuit_breaker.record_failure()
            raise
```

## Security Considerations

### Secret Management
- **No Hardcoded Secrets**: All sensitive data via environment variables
- **Secure Logging**: Sensitive fields excluded from logs
- **Token Rotation**: Automatic refresh of short-lived tokens
- **Least Privilege**: Minimal permissions for service accounts

### Network Security
- **Proxy Support**: JPMC corporate proxy compliance
- **Certificate Validation**: TLS verification for all external calls
- **Timeout Controls**: Prevent hanging connections and resource exhaustion
- **Rate Limiting**: Built-in retry limits to prevent abuse

## Monitoring and Observability

### Telemetry Integration
```python
# OpenTelemetry instrumentation
from src.infra.telemetry import setup_telemetry

setup_telemetry(
    service_name="utilities-assistant",
    endpoint="https://jaeger.company.com"
)

# Custom metrics
from src.infra.telemetry import get_metrics
metrics = get_metrics()
metrics.record_latency("search.duration", duration_ms)
```

### Health Monitoring
- **Service Health**: Built-in health check endpoints
- **Dependency Health**: External service monitoring
- **Performance Metrics**: Latency, throughput, error rates
- **Resource Usage**: Memory, CPU, connection pool stats

## Future Enhancement Opportunities

### Cloud-Native Features
- **Kubernetes Integration**: Helm charts and operator patterns
- **Service Mesh**: Istio integration for advanced networking
- **Auto-scaling**: Horizontal pod autoscaling based on metrics
- **Config Management**: Kubernetes ConfigMaps and Secrets

### Advanced Observability
- **Distributed Tracing**: Cross-service request correlation
- **Custom Dashboards**: Grafana integration for RAG-specific metrics
- **Alerting**: PagerDuty integration for critical failures
- **Log Aggregation**: Structured logging with ELK stack integration