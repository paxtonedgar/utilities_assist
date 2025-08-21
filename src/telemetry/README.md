# Telemetry - Performance Monitoring and Observability

## Purpose
Provides comprehensive telemetry, logging, and performance monitoring capabilities for the utilities assistant system. Enables real-time visibility into system performance, error tracking, and operational metrics for debugging, optimization, and reliability monitoring.

## Architecture
Centralized observability layer with structured logging and metrics collection:

```
Application Components
         ↓
Telemetry Collection Layer
         ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Structured      │ Performance     │ Error Tracking  │
│   Logging       │   Metrics      │ & Monitoring    │
└─────────────────┴─────────────────┴─────────────────┘
         ↓
External Monitoring Systems
```

### Design Principles
- **Structured Logging**: Consistent, machine-readable log formats
- **Performance First**: Minimal overhead on application performance
- **Privacy Aware**: No sensitive data in logs or metrics
- **Correlation Ready**: Request tracing and context preservation

## Key Files

### Core Telemetry Infrastructure
- `logger.py` - **Primary logging and metrics system** (245 lines)
  - Structured JSON logging with context preservation
  - Performance timing and measurement utilities
  - Error tracking and exception handling
  - OpenTelemetry integration for distributed tracing
  - Memory and resource usage monitoring

## Dependencies

### Internal Dependencies
- `src.infra.config` - Configuration and environment settings
- `src.util.timing` - High-precision timing utilities
- Application modules (for contextual logging)

### External Dependencies
- `structlog` - Structured logging framework
- `opentelemetry` - Distributed tracing and metrics
- `psutil` - System resource monitoring
- `logging` - Python standard logging integration
- `time` - Performance timing measurements

## Integration Points

### Structured Logging System
```python
# High-performance structured logging
import structlog
from typing import Dict, Any, Optional

class UtilitiesLogger:
    def __init__(self, service_name: str, environment: str):
        """Initialize structured logger with context."""
        
        # Configure structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger()
        self.base_context = {
            "service": service_name,
            "environment": environment,
            "version": get_app_version()
        }
    
    def info(self, message: str, **context):
        """Log info message with context."""
        self.logger.info(message, **self.base_context, **context)
    
    def warning(self, message: str, **context):
        """Log warning with context."""
        self.logger.warning(message, **self.base_context, **context)
    
    def error(self, message: str, error: Exception = None, **context):
        """Log error with exception details."""
        error_context = {
            **self.base_context,
            **context
        }
        
        if error:
            error_context.update({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_traceback": format_exception(error)
            })
        
        self.logger.error(message, **error_context)
```

### Performance Metrics Collection
```python
# Comprehensive performance monitoring
import time
import psutil
from contextlib import contextmanager
from typing import Generator, Dict, Any

class PerformanceMonitor:
    def __init__(self, logger: UtilitiesLogger):
        self.logger = logger
        self.active_operations = {}
    
    @contextmanager
    def measure_operation(
        self, 
        operation_name: str, 
        context: Dict[str, Any] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Context manager for operation performance measurement."""
        
        operation_id = f"{operation_name}_{int(time.time() * 1000)}"
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss
        
        # Track active operation
        operation_context = {
            "operation_id": operation_id,
            "operation_name": operation_name,
            "start_time": start_time,
            "context": context or {}
        }
        
        self.active_operations[operation_id] = operation_context
        
        try:
            yield operation_context
        finally:
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss
            
            # Calculate metrics
            duration_ms = (end_time - start_time) * 1000
            memory_delta_mb = (end_memory - start_memory) / 1024 / 1024
            
            # Log performance metrics
            self.logger.info(
                f"Operation completed: {operation_name}",
                operation_id=operation_id,
                duration_ms=round(duration_ms, 2),
                memory_delta_mb=round(memory_delta_mb, 2),
                **operation_context["context"]
            )
            
            # Clean up
            del self.active_operations[operation_id]
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Collect current system resource metrics."""
        
        process = psutil.Process()
        
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "process_memory_mb": process.memory_info().rss / 1024 / 1024,
            "process_cpu_percent": process.cpu_percent(),
            "active_operations": len(self.active_operations),
            "timestamp": time.time()
        }
```

### Request Tracing and Correlation
```python
# Distributed tracing integration
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import uuid

class RequestTracer:
    def __init__(self, service_name: str, jaeger_endpoint: str = None):
        """Initialize OpenTelemetry tracing."""
        
        # Set up tracer provider
        trace.set_tracer_provider(TracerProvider())
        self.tracer = trace.get_tracer(service_name)
        
        # Configure Jaeger exporter if endpoint provided
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=6831,
            )
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
    
    @contextmanager
    def trace_operation(
        self, 
        operation_name: str, 
        attributes: Dict[str, Any] = None
    ) -> Generator[trace.Span, None, None]:
        """Create a traced operation span."""
        
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add operation attributes
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, str(value))
            
            # Add request correlation ID
            correlation_id = str(uuid.uuid4())
            span.set_attribute("correlation_id", correlation_id)
            
            try:
                yield span
            except Exception as e:
                # Record exception in span
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
    
    def get_current_trace_id(self) -> str:
        """Get current trace ID for correlation."""
        current_span = trace.get_current_span()
        if current_span:
            return format(current_span.get_span_context().trace_id, '032x')
        return "no-trace"
```

### Application Integration
```python
# Easy integration with application components
class TelemetryIntegration:
    def __init__(self, service_name: str, environment: str):
        self.logger = UtilitiesLogger(service_name, environment)
        self.performance = PerformanceMonitor(self.logger)
        self.tracer = RequestTracer(service_name)
    
    def log_request_start(
        self, 
        request_id: str, 
        user_id: str, 
        query: str,
        **context
    ):
        """Log start of user request processing."""
        
        # Sanitize query for logging (remove PII)
        sanitized_query = self._sanitize_for_logging(query)
        
        self.logger.info(
            "Request processing started",
            request_id=request_id,
            user_id=user_id,
            query_length=len(query),
            query_preview=sanitized_query[:100],
            **context
        )
    
    def log_request_complete(
        self,
        request_id: str,
        success: bool,
        duration_ms: float,
        response_length: int = None,
        **context
    ):
        """Log completion of user request processing."""
        
        log_level = "info" if success else "error"
        message = "Request completed successfully" if success else "Request failed"
        
        getattr(self.logger, log_level)(
            message,
            request_id=request_id,
            success=success,
            duration_ms=round(duration_ms, 2),
            response_length=response_length,
            **context
        )
    
    def log_component_performance(
        self,
        component: str,
        operation: str,
        duration_ms: float,
        success: bool,
        **metrics
    ):
        """Log performance metrics for system components."""
        
        self.logger.info(
            f"{component} operation performance",
            component=component,
            operation=operation,
            duration_ms=round(duration_ms, 2),
            success=success,
            **metrics
        )
    
    def _sanitize_for_logging(self, text: str) -> str:
        """Remove sensitive information from text for safe logging."""
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Remove potential tokens (long alphanumeric strings)
        text = re.sub(r'\b[A-Za-z0-9]{32,}\b', '[TOKEN]', text)
        
        # Remove UUIDs
        text = re.sub(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', '[UUID]', text, flags=re.IGNORECASE)
        
        return text
```

## Current Implementation Details

### LangGraph Workflow Integration
```python
# Telemetry integration with LangGraph nodes
async def telemetry_aware_node(
    state: GraphState, 
    config: RunnableConfig,
    node_name: str,
    telemetry: TelemetryIntegration
) -> GraphState:
    """LangGraph node with integrated telemetry."""
    
    request_id = config.get("configurable", {}).get("request_id", "unknown")
    
    with telemetry.performance.measure_operation(
        f"node_{node_name}",
        {"request_id": request_id, "node": node_name}
    ):
        with telemetry.tracer.trace_operation(
            f"langgraph_node_{node_name}",
            {"node_name": node_name, "request_id": request_id}
        ) as span:
            
            try:
                # Execute node logic
                result = await execute_node_logic(state, config)
                
                # Log successful completion
                telemetry.logger.info(
                    f"Node {node_name} completed successfully",
                    request_id=request_id,
                    node_name=node_name,
                    input_state_keys=list(state.keys()),
                    output_state_keys=list(result.keys())
                )
                
                # Add telemetry to span
                span.set_attribute("success", True)
                span.set_attribute("output_keys", str(list(result.keys())))
                
                return result
                
            except Exception as e:
                # Log error with context
                telemetry.logger.error(
                    f"Node {node_name} failed",
                    error=e,
                    request_id=request_id,
                    node_name=node_name,
                    input_state=sanitize_state_for_logging(state)
                )
                
                # Record error in span
                span.set_attribute("success", False)
                span.set_attribute("error_type", type(e).__name__)
                
                raise
```

### Performance Budget Monitoring
```python
# Monitor performance against defined budgets
class PerformanceBudgetMonitor:
    def __init__(self, telemetry: TelemetryIntegration):
        self.telemetry = telemetry
        self.budgets = {
            "intent_classification_ms": 1,      # P1 requirement
            "query_normalization_ms": 10,      # P2 requirement
            "search_operation_ms": 2000,       # P2 requirement
            "total_request_ms": 3000,          # P1 requirement
        }
        self.violations = {}
    
    def check_budget_compliance(
        self,
        operation: str,
        actual_duration_ms: float,
        request_id: str
    ):
        """Check if operation stayed within performance budget."""
        
        budget_key = f"{operation}_ms"
        if budget_key not in self.budgets:
            return
        
        budget_ms = self.budgets[budget_key]
        over_budget = actual_duration_ms > budget_ms
        
        if over_budget:
            # Track violation
            violation_id = f"{request_id}_{operation}"
            self.violations[violation_id] = {
                "operation": operation,
                "budget_ms": budget_ms,
                "actual_ms": actual_duration_ms,
                "overage_ms": actual_duration_ms - budget_ms,
                "overage_percent": ((actual_duration_ms - budget_ms) / budget_ms) * 100,
                "timestamp": time.time(),
                "request_id": request_id
            }
            
            # Log budget violation
            self.telemetry.logger.warning(
                f"Performance budget violation: {operation}",
                operation=operation,
                budget_ms=budget_ms,
                actual_ms=round(actual_duration_ms, 2),
                overage_ms=round(actual_duration_ms - budget_ms, 2),
                overage_percent=round(((actual_duration_ms - budget_ms) / budget_ms) * 100, 1),
                request_id=request_id
            )
        else:
            # Log successful compliance
            self.telemetry.logger.debug(
                f"Performance budget met: {operation}",
                operation=operation,
                budget_ms=budget_ms,
                actual_ms=round(actual_duration_ms, 2),
                margin_ms=round(budget_ms - actual_duration_ms, 2),
                request_id=request_id
            )
    
    def get_violation_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent performance budget violations."""
        
        cutoff_time = time.time() - (time_window_hours * 3600)
        recent_violations = {
            vid: violation for vid, violation in self.violations.items()
            if violation["timestamp"] > cutoff_time
        }
        
        if not recent_violations:
            return {"status": "all_budgets_met", "violations": 0}
        
        # Analyze violations by operation
        by_operation = {}
        for violation in recent_violations.values():
            op = violation["operation"]
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(violation)
        
        summary = {
            "status": "budget_violations_detected",
            "total_violations": len(recent_violations),
            "time_window_hours": time_window_hours,
            "by_operation": {}
        }
        
        for operation, violations in by_operation.items():
            summary["by_operation"][operation] = {
                "count": len(violations),
                "avg_overage_ms": sum(v["overage_ms"] for v in violations) / len(violations),
                "max_overage_ms": max(v["overage_ms"] for v in violations),
                "avg_overage_percent": sum(v["overage_percent"] for v in violations) / len(violations)
            }
        
        return summary
```

### Error Classification and Alerting
```python
# Intelligent error classification and alerting
class ErrorClassifier:
    def __init__(self, telemetry: TelemetryIntegration):
        self.telemetry = telemetry
        self.error_patterns = {
            "transient": [
                "timeout", "connection", "rate limit", "throttle"
            ],
            "configuration": [
                "authentication", "permission", "key", "config"
            ],
            "data": [
                "validation", "format", "schema", "parse"
            ],
            "resource": [
                "memory", "disk", "cpu", "quota"
            ]
        }
        self.alert_thresholds = {
            "transient": 10,      # 10 in 5 minutes
            "configuration": 3,   # 3 in 5 minutes
            "data": 5,           # 5 in 5 minutes
            "resource": 1,       # 1 in 5 minutes
        }
        self.recent_errors = {}
    
    def classify_and_log_error(
        self,
        error: Exception,
        operation: str,
        request_id: str,
        context: Dict[str, Any] = None
    ):
        """Classify error and determine if alerting is needed."""
        
        error_message = str(error).lower()
        error_type = type(error).__name__
        
        # Classify error
        error_category = "unknown"
        for category, patterns in self.error_patterns.items():
            if any(pattern in error_message for pattern in patterns):
                error_category = category
                break
        
        # Log classified error
        self.telemetry.logger.error(
            f"Classified error in {operation}",
            error=error,
            error_category=error_category,
            error_type=error_type,
            operation=operation,
            request_id=request_id,
            **(context or {})
        )
        
        # Track for alerting
        self._track_error_for_alerting(error_category, error_type, operation)
    
    def _track_error_for_alerting(
        self,
        category: str,
        error_type: str,
        operation: str
    ):
        """Track error frequency for alerting decisions."""
        
        current_time = time.time()
        window_start = current_time - 300  # 5 minute window
        
        # Clean old errors
        self.recent_errors = {
            key: errors for key, errors in self.recent_errors.items()
            if any(error_time > window_start for error_time in errors)
        }
        
        # Add current error
        key = f"{category}_{operation}"
        if key not in self.recent_errors:
            self.recent_errors[key] = []
        
        self.recent_errors[key] = [
            error_time for error_time in self.recent_errors[key]
            if error_time > window_start
        ]
        self.recent_errors[key].append(current_time)
        
        # Check alert threshold
        threshold = self.alert_thresholds.get(category, 999)
        if len(self.recent_errors[key]) >= threshold:
            self._trigger_alert(category, operation, len(self.recent_errors[key]))
    
    def _trigger_alert(self, category: str, operation: str, count: int):
        """Trigger alert for high error frequency."""
        
        self.telemetry.logger.error(
            f"HIGH ERROR FREQUENCY ALERT",
            alert_type="error_frequency",
            error_category=category,
            operation=operation,
            error_count_5min=count,
            threshold=self.alert_thresholds.get(category),
            requires_investigation=True
        )
```

## Health Monitoring and Diagnostics

### System Health Checks
```python
# Comprehensive health monitoring
async def comprehensive_health_check(
    telemetry: TelemetryIntegration
) -> Dict[str, Any]:
    """Perform comprehensive system health check."""
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "checks": {},
        "metrics": {}
    }
    
    # System resource check
    with telemetry.performance.measure_operation("health_check_system"):
        system_metrics = telemetry.performance.get_system_metrics()
        health_status["metrics"]["system"] = system_metrics
        
        # Check resource usage
        if system_metrics["memory_percent"] > 90:
            health_status["checks"]["memory"] = "critical"
            health_status["status"] = "degraded"
        elif system_metrics["memory_percent"] > 80:
            health_status["checks"]["memory"] = "warning"
        else:
            health_status["checks"]["memory"] = "healthy"
    
    # Performance budget compliance check
    budget_monitor = PerformanceBudgetMonitor(telemetry)
    violation_summary = budget_monitor.get_violation_summary(1)  # Last hour
    
    if violation_summary["status"] == "budget_violations_detected":
        health_status["checks"]["performance_budgets"] = "degraded"
        health_status["status"] = "degraded"
        health_status["metrics"]["performance_violations"] = violation_summary
    else:
        health_status["checks"]["performance_budgets"] = "healthy"
    
    # Error rate check
    error_classifier = ErrorClassifier(telemetry)
    error_summary = error_classifier.get_error_summary(1)  # Last hour
    
    if error_summary.get("total_errors", 0) > 50:  # High error threshold
        health_status["checks"]["error_rate"] = "critical"
        health_status["status"] = "degraded"
    elif error_summary.get("total_errors", 0) > 20:  # Medium error threshold
        health_status["checks"]["error_rate"] = "warning"
    else:
        health_status["checks"]["error_rate"] = "healthy"
    
    health_status["metrics"]["errors"] = error_summary
    
    return health_status
```

## Future Enhancement Opportunities

### Advanced Monitoring Capabilities
- **Real-Time Dashboards**: Grafana integration for live monitoring
- **Predictive Alerting**: ML-based anomaly detection
- **User Journey Tracking**: Complete request lifecycle tracing
- **Custom Metrics**: Domain-specific performance indicators

### Integration Enhancements
- **APM Integration**: New Relic, DataDog, or Azure Monitor
- **Log Aggregation**: ELK stack or Azure Log Analytics
- **Metrics Export**: Prometheus integration for time-series data
- **Alerting Integration**: PagerDuty, Slack, or Teams notifications

### Privacy and Compliance
- **Data Masking**: Automatic PII detection and redaction
- **Audit Logging**: Comprehensive compliance logging
- **Data Retention**: Configurable log retention policies
- **Access Controls**: Role-based access to telemetry data

## Performance Considerations

### Minimal Overhead Design
- **Asynchronous Logging**: Non-blocking log writes
- **Sampling**: Configurable sampling rates for high-volume metrics
- **Buffer Management**: Efficient log buffer handling
- **Memory Optimization**: Minimal memory footprint for telemetry

### Scalability Features
- **Distributed Tracing**: Support for microservice architectures
- **Load Balancing**: Telemetry collection across multiple instances
- **Data Partitioning**: Efficient storage and retrieval of metrics
- **Compression**: Log compression for storage optimization