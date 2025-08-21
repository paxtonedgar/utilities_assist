# Util - Common Utilities and Helper Functions

## Purpose
Provides reusable utility functions and helper modules that support core system operations including caching, filtering, timing, and common data processing tasks. These utilities promote code reuse, maintainability, and consistent implementation patterns across the application.

## Architecture
Service utility layer providing foundational support functions:

```
Application Components
         ↓
Utility Functions Layer
         ↓
┌─────────────────┬─────────────────┬─────────────────┐
│ Performance     │ Data Processing │ System Helpers  │
│   Utilities     │   & Filtering   │ & Diagnostics   │
└─────────────────┴─────────────────┴─────────────────┘
```

### Design Principles
- **Reusability**: Pure functions that can be used across components
- **Performance**: Optimized implementations for frequently used operations
- **Type Safety**: Strong typing for reliable integration
- **Minimal Dependencies**: Lightweight implementations with few external dependencies

## Key Files

### Core Utility Modules
- `cache.py` - **TTL-based caching utilities** (168 lines)
  - Time-to-live (TTL) cache implementations
  - LRU eviction policies
  - Cache statistics and monitoring
  - Thread-safe caching operations

- `timing.py` - **High-precision timing and performance measurement** (124 lines)
  - Performance timing decorators and context managers
  - Execution time measurement utilities
  - Benchmark and profiling helpers
  - Statistical timing analysis

- `filters.py` - **Data filtering and validation utilities** (203 lines)
  - Content filtering and sanitization
  - Data validation and type checking
  - Search result filtering and ranking
  - Input sanitization for security

## Dependencies

### Internal Dependencies
- `src.telemetry.logger` - Logging and monitoring integration
- Application modules (for contextual utilities)

### External Dependencies
- `functools` - Decorator and caching utilities
- `time` - Timing and performance measurement
- `threading` - Thread-safe operations
- `typing` - Type hints and validation
- `re` - Regular expression utilities

## Integration Points

### TTL Caching System
```python
# High-performance TTL cache with LRU eviction
import time
import threading
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from functools import wraps

T = TypeVar('T')

class TTLCache(Generic[T]):
    """Thread-safe TTL cache with LRU eviction."""
    
    def __init__(self, maxsize: int = 128, ttl_seconds: int = 300):
        """Initialize TTL cache with size and time limits."""
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Any] = {}
        self._access_times: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0
        }
    
    def get(self, key: str) -> Optional[T]:
        """Get value from cache, checking TTL."""
        with self._lock:
            current_time = time.time()
            
            # Check if key exists and is not expired
            if key in self._cache:
                creation_time = self._creation_times[key]
                if current_time - creation_time <= self.ttl_seconds:
                    # Update access time and return value
                    self._access_times[key] = current_time
                    self._stats["hits"] += 1
                    return self._cache[key]
                else:
                    # Expired, remove from cache
                    self._remove_key(key)
                    self._stats["expirations"] += 1
            
            self._stats["misses"] += 1
            return None
    
    def put(self, key: str, value: T) -> None:
        """Store value in cache with TTL."""
        with self._lock:
            current_time = time.time()
            
            # Check if we need to evict
            if key not in self._cache and len(self._cache) >= self.maxsize:
                self._evict_lru()
            
            # Store value with timestamps
            self._cache[key] = value
            self._access_times[key] = current_time
            self._creation_times[key] = current_time
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        # Find LRU key
        lru_key = min(self._access_times.keys(), 
                      key=lambda k: self._access_times[k])
        self._remove_key(lru_key)
        self._stats["evictions"] += 1
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all internal structures."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
        self._creation_times.pop(key, None)
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries, return count removed."""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, creation_time in self._creation_times.items()
                if current_time - creation_time > self.ttl_seconds
            ]
            
            for key in expired_keys:
                self._remove_key(key)
                self._stats["expirations"] += 1
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                "hit_rate": round(hit_rate, 3),
                "size": len(self._cache),
                "max_size": self.maxsize,
                "ttl_seconds": self.ttl_seconds
            }

# Decorator for easy function caching
def ttl_cache(maxsize: int = 128, ttl_seconds: int = 300):
    """Decorator to add TTL caching to functions."""
    
    def decorator(func: Callable) -> Callable:
        cache = TTLCache(maxsize=maxsize, ttl_seconds=ttl_seconds)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key from arguments
            cache_key = _generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            return result
        
        # Add cache management methods
        wrapper.cache_info = cache.get_stats
        wrapper.cache_clear = lambda: cache._cache.clear()
        wrapper.cache_cleanup = cache.cleanup_expired
        
        return wrapper
    
    return decorator

def _generate_cache_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Generate deterministic cache key from function arguments."""
    import hashlib
    
    # Create hashable representation
    key_data = f"{func_name}:{args}:{sorted(kwargs.items())}"
    return hashlib.md5(key_data.encode()).hexdigest()
```

### Performance Timing Utilities
```python
# High-precision timing and performance measurement
import time
import statistics
from contextlib import contextmanager
from typing import List, Dict, Any, Generator, Callable
from functools import wraps

class PerformanceTimer:
    """High-precision performance timer with statistics."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.measurements: List[float] = []
        self.start_time: Optional[float] = None
        self.is_running = False
    
    def start(self) -> None:
        """Start timing measurement."""
        if self.is_running:
            raise RuntimeError("Timer is already running")
        
        self.start_time = time.perf_counter()
        self.is_running = True
    
    def stop(self) -> float:
        """Stop timing and record measurement."""
        if not self.is_running:
            raise RuntimeError("Timer is not running")
        
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        self.measurements.append(duration)
        self.is_running = False
        
        return duration
    
    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """Context manager for timing operations."""
        self.start()
        try:
            yield
        finally:
            self.stop()
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive timing statistics."""
        if not self.measurements:
            return {"count": 0}
        
        measurements_ms = [m * 1000 for m in self.measurements]  # Convert to ms
        
        return {
            "count": len(measurements_ms),
            "total_ms": sum(measurements_ms),
            "mean_ms": statistics.mean(measurements_ms),
            "median_ms": statistics.median(measurements_ms),
            "min_ms": min(measurements_ms),
            "max_ms": max(measurements_ms),
            "std_dev_ms": statistics.stdev(measurements_ms) if len(measurements_ms) > 1 else 0,
            "p95_ms": statistics.quantiles(measurements_ms, n=20)[18] if len(measurements_ms) >= 20 else max(measurements_ms),
            "p99_ms": statistics.quantiles(measurements_ms, n=100)[98] if len(measurements_ms) >= 100 else max(measurements_ms)
        }
    
    def reset(self) -> None:
        """Reset all measurements."""
        self.measurements.clear()
        self.is_running = False

# Performance timing decorator
def time_it(name: str = None, print_results: bool = False):
    """Decorator to time function execution."""
    
    def decorator(func: Callable) -> Callable:
        timer_name = name or f"{func.__module__}.{func.__name__}"
        timer = PerformanceTimer(timer_name)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timer.measure():
                result = func(*args, **kwargs)
            
            if print_results:
                stats = timer.get_statistics()
                print(f"{timer_name}: {stats['mean_ms']:.2f}ms (last measurement)")
            
            return result
        
        # Add timer access
        wrapper.get_timing_stats = timer.get_statistics
        wrapper.reset_timing_stats = timer.reset
        
        return wrapper
    
    return decorator

# Benchmark utilities
class BenchmarkSuite:
    """Suite for comparing performance of different implementations."""
    
    def __init__(self, name: str):
        self.name = name
        self.timers: Dict[str, PerformanceTimer] = {}
    
    def benchmark(self, implementation_name: str, func: Callable, *args, **kwargs):
        """Benchmark a specific implementation."""
        if implementation_name not in self.timers:
            self.timers[implementation_name] = PerformanceTimer(implementation_name)
        
        timer = self.timers[implementation_name]
        
        with timer.measure():
            result = func(*args, **kwargs)
        
        return result
    
    def run_comparison(
        self,
        implementations: Dict[str, Callable],
        iterations: int = 100,
        *args,
        **kwargs
    ) -> Dict[str, Dict[str, float]]:
        """Run comparative benchmark across implementations."""
        
        results = {}
        
        for name, func in implementations.items():
            print(f"Benchmarking {name}...")
            
            for _ in range(iterations):
                self.benchmark(name, func, *args, **kwargs)
            
            results[name] = self.timers[name].get_statistics()
        
        return results
    
    def print_comparison_report(self, results: Dict[str, Dict[str, float]]):
        """Print formatted comparison report."""
        print(f"\n{self.name} Benchmark Results:")
        print("=" * 60)
        
        # Sort by mean execution time
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].get("mean_ms", float("inf"))
        )
        
        baseline_mean = sorted_results[0][1]["mean_ms"]
        
        for name, stats in sorted_results:
            relative_speed = baseline_mean / stats["mean_ms"]
            print(f"{name:20} | {stats['mean_ms']:8.2f}ms | {relative_speed:6.2f}x")
        
        print("=" * 60)
```

### Data Filtering and Validation Utilities
```python
# Comprehensive data filtering and validation
import re
import html
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar

T = TypeVar('T')

class ContentFilter:
    """Advanced content filtering and sanitization."""
    
    def __init__(self):
        # Security patterns
        self.xss_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'onclick\s*='
        ]
        
        # PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b'
        }
        
        # Compiled patterns for performance
        self._compiled_xss = [re.compile(pattern, re.IGNORECASE) for pattern in self.xss_patterns]
        self._compiled_pii = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.pii_patterns.items()
        }
    
    def sanitize_html(self, text: str, allow_basic_formatting: bool = True) -> str:
        """Sanitize HTML content for safe display."""
        
        # Remove dangerous script elements
        for pattern in self._compiled_xss:
            text = pattern.sub('', text)
        
        if allow_basic_formatting:
            # Preserve basic formatting tags
            safe_tags = ['<b>', '</b>', '<i>', '</i>', '<em>', '</em>', '<strong>', '</strong>']
            protected = {}
            
            # Temporarily protect safe tags
            for i, tag in enumerate(safe_tags):
                placeholder = f"__SAFE_TAG_{i}__"
                protected[placeholder] = tag
                text = text.replace(tag, placeholder)
            
            # Escape remaining HTML
            text = html.escape(text)
            
            # Restore protected tags
            for placeholder, tag in protected.items():
                text = text.replace(placeholder, tag)
        else:
            # Escape all HTML
            text = html.escape(text)
        
        return text
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect potential PII in text."""
        
        detected = {}
        
        for pii_type, pattern in self._compiled_pii.items():
            matches = pattern.findall(text)
            if matches:
                detected[pii_type] = matches
        
        return detected
    
    def mask_pii(self, text: str, mask_char: str = '*') -> str:
        """Mask detected PII in text."""
        
        masked_text = text
        
        for pii_type, pattern in self._compiled_pii.items():
            if pii_type == 'email':
                # Preserve domain for emails
                masked_text = pattern.sub(
                    lambda m: f"{mask_char * 5}@{m.group().split('@')[1]}",
                    masked_text
                )
            else:
                # Full masking for other PII types
                masked_text = pattern.sub(
                    lambda m: mask_char * len(m.group()),
                    masked_text
                )
        
        return masked_text
    
    def validate_query_safety(self, query: str) -> Dict[str, Any]:
        """Comprehensive query safety validation."""
        
        issues = []
        
        # Check for XSS patterns
        for pattern in self._compiled_xss:
            if pattern.search(query):
                issues.append("potential_xss")
                break
        
        # Check for PII
        pii_detected = self.detect_pii(query)
        if pii_detected:
            issues.append("contains_pii")
        
        # Check query length
        if len(query) > 1000:
            issues.append("excessive_length")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\.\./',           # Directory traversal
            r'union\s+select',  # SQL injection
            r'exec\s*\(',      # Code execution
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append("suspicious_pattern")
                break
        
        return {
            "is_safe": len(issues) == 0,
            "issues": issues,
            "pii_detected": pii_detected,
            "sanitized_query": self.sanitize_html(query, allow_basic_formatting=False)
        }

class SearchResultFilter:
    """Filter and rank search results."""
    
    def __init__(self):
        self.quality_thresholds = {
            "min_content_length": 50,
            "min_title_length": 10,
            "max_title_length": 200,
            "min_confidence": 0.1
        }
    
    def filter_by_quality(
        self,
        results: List[Dict[str, Any]],
        strict: bool = False
    ) -> List[Dict[str, Any]]:
        """Filter results based on quality metrics."""
        
        filtered = []
        
        for result in results:
            # Check content length
            content = result.get("content", "")
            if len(content) < self.quality_thresholds["min_content_length"]:
                continue
            
            # Check title quality
            title = result.get("title", "")
            if (len(title) < self.quality_thresholds["min_title_length"] or
                len(title) > self.quality_thresholds["max_title_length"]):
                continue
            
            # Check confidence score
            confidence = result.get("confidence", 0.0)
            if confidence < self.quality_thresholds["min_confidence"]:
                continue
            
            # Additional strict checks
            if strict:
                # Check for duplicate content
                if self._is_duplicate_content(result, filtered):
                    continue
                
                # Check content relevance
                if not self._has_relevant_content(result):
                    continue
            
            filtered.append(result)
        
        return filtered
    
    def deduplicate_results(
        self,
        results: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        
        deduplicated = []
        
        for result in results:
            is_duplicate = False
            
            for existing in deduplicated:
                similarity = self._calculate_content_similarity(
                    result.get("content", ""),
                    existing.get("content", "")
                )
                
                if similarity > similarity_threshold:
                    # Keep the one with higher confidence
                    if result.get("confidence", 0) > existing.get("confidence", 0):
                        deduplicated.remove(existing)
                        deduplicated.append(result)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(result)
        
        return deduplicated
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity score."""
        
        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _is_duplicate_content(
        self,
        result: Dict[str, Any],
        existing_results: List[Dict[str, Any]]
    ) -> bool:
        """Check if result is duplicate of existing results."""
        
        for existing in existing_results:
            # Check URL similarity
            if result.get("url") == existing.get("url"):
                return True
            
            # Check title similarity
            title_similarity = self._calculate_content_similarity(
                result.get("title", ""),
                existing.get("title", "")
            )
            if title_similarity > 0.9:
                return True
        
        return False
    
    def _has_relevant_content(self, result: Dict[str, Any]) -> bool:
        """Check if result has relevant, meaningful content."""
        
        content = result.get("content", "").lower()
        
        # Filter out common non-content indicators
        non_content_indicators = [
            "page not found",
            "404 error",
            "access denied",
            "login required",
            "under construction",
            "coming soon"
        ]
        
        for indicator in non_content_indicators:
            if indicator in content:
                return False
        
        # Check for minimum meaningful words
        words = content.split()
        meaningful_words = [w for w in words if len(w) > 3]
        
        return len(meaningful_words) >= 10

# Validation utilities
def validate_type(value: Any, expected_type: type, field_name: str = "value") -> T:
    """Validate value type with informative error messages."""
    
    if not isinstance(value, expected_type):
        raise TypeError(
            f"{field_name} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    
    return value

def validate_range(
    value: Union[int, float],
    min_val: Optional[Union[int, float]] = None,
    max_val: Optional[Union[int, float]] = None,
    field_name: str = "value"
) -> Union[int, float]:
    """Validate numeric value is within specified range."""
    
    if min_val is not None and value < min_val:
        raise ValueError(f"{field_name} must be >= {min_val}, got {value}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"{field_name} must be <= {max_val}, got {value}")
    
    return value

def validate_list_not_empty(
    value: List[T],
    field_name: str = "list"
) -> List[T]:
    """Validate list is not empty."""
    
    if not value:
        raise ValueError(f"{field_name} cannot be empty")
    
    return value
```

## Current Implementation Details

### Cache Integration with Application Components
```python
# Application-specific caching utilities
from src.util.cache import ttl_cache, TTLCache
from src.telemetry.logger import get_logger

logger = get_logger(__name__)

# Intent classification caching
@ttl_cache(maxsize=2048, ttl_seconds=600)  # 10 minute TTL
def cached_intent_classification(query: str, context_hash: str) -> Dict[str, Any]:
    """Cache intent classification results."""
    # This would be implemented by the actual intent service
    pass

# Embedding caching for expensive operations
class EmbeddingCache:
    """Specialized cache for embedding operations."""
    
    def __init__(self):
        self.embedding_cache = TTLCache(maxsize=1000, ttl_seconds=3600)  # 1 hour TTL
        self.model_cache = TTLCache(maxsize=10, ttl_seconds=86400)      # 24 hour TTL
    
    def get_cached_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get cached text embedding."""
        cache_key = f"{model_name}:{hash(text)}"
        return self.embedding_cache.get(cache_key)
    
    def cache_embedding(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Cache text embedding."""
        cache_key = f"{model_name}:{hash(text)}"
        self.embedding_cache.put(cache_key, embedding)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            "embedding_cache": self.embedding_cache.get_stats(),
            "model_cache": self.model_cache.get_stats()
        }

# Global cache instances
embedding_cache = EmbeddingCache()
```

### Performance Monitoring Integration
```python
# Integration with telemetry system
from src.util.timing import time_it, PerformanceTimer, BenchmarkSuite
from src.telemetry.logger import get_logger

logger = get_logger(__name__)

# Timed operations for performance monitoring
@time_it(name="search_operation", print_results=False)
async def timed_search_operation(query: str, **kwargs) -> List[Dict[str, Any]]:
    """Search operation with automatic timing."""
    # Implementation would be provided by search service
    pass

# Performance budget validation
class PerformanceBudgetValidator:
    """Validate operations stay within performance budgets."""
    
    def __init__(self):
        self.budgets = {
            "intent_classification": 1.0,    # 1ms budget
            "query_normalization": 10.0,     # 10ms budget
            "search_execution": 2000.0,      # 2s budget
            "response_generation": 1000.0,   # 1s budget
        }
        self.violations = []
    
    def check_budget(self, operation: str, duration_ms: float, context: Dict[str, Any] = None):
        """Check if operation exceeded budget."""
        
        budget = self.budgets.get(operation)
        if not budget:
            return  # No budget defined
        
        if duration_ms > budget:
            violation = {
                "operation": operation,
                "budget_ms": budget,
                "actual_ms": duration_ms,
                "overage_ms": duration_ms - budget,
                "overage_percent": ((duration_ms - budget) / budget) * 100,
                "timestamp": time.time(),
                "context": context or {}
            }
            
            self.violations.append(violation)
            
            logger.warning(
                f"Performance budget exceeded: {operation}",
                **violation
            )
```

### Security and Validation Integration
```python
# Security utilities integration
from src.util.filters import ContentFilter, SearchResultFilter

# Global filter instances
content_filter = ContentFilter()
result_filter = SearchResultFilter()

def safe_process_user_query(query: str) -> Dict[str, Any]:
    """Safely process user query with comprehensive validation."""
    
    # Validate query safety
    safety_check = content_filter.validate_query_safety(query)
    
    if not safety_check["is_safe"]:
        logger.warning(
            "Unsafe query detected",
            issues=safety_check["issues"],
            pii_detected=bool(safety_check["pii_detected"])
        )
        
        # Use sanitized version
        query = safety_check["sanitized_query"]
    
    # Additional processing would continue here
    return {
        "processed_query": query,
        "safety_check": safety_check
    }

def safe_process_search_results(
    results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Process search results with quality filtering."""
    
    # Filter by quality
    quality_filtered = result_filter.filter_by_quality(results, strict=True)
    
    # Remove duplicates
    deduplicated = result_filter.deduplicate_results(quality_filtered)
    
    # Sanitize content
    for result in deduplicated:
        if "content" in result:
            result["content"] = content_filter.sanitize_html(result["content"])
        if "title" in result:
            result["title"] = content_filter.sanitize_html(result["title"])
    
    logger.info(
        "Search results processed",
        original_count=len(results),
        filtered_count=len(quality_filtered),
        final_count=len(deduplicated)
    )
    
    return deduplicated
```

## Future Enhancement Opportunities

### Advanced Caching Strategies
- **Semantic Caching**: Cache based on semantic similarity rather than exact matches
- **Predictive Caching**: ML-based cache warming for anticipated queries
- **Distributed Caching**: Redis integration for multi-instance deployments
- **Cache Hierarchies**: Multi-level caching with different TTL strategies

### Enhanced Performance Monitoring
- **Real-Time Profiling**: Continuous performance profiling with sampling
- **Memory Tracking**: Detailed memory usage patterns and leak detection
- **Database Query Analysis**: Slow query detection and optimization
- **Custom Metrics**: Domain-specific performance indicators

### Security Enhancements
- **Advanced PII Detection**: ML-based sensitive data identification
- **Content Classification**: Automatic content categorization and filtering
- **Threat Detection**: Real-time security threat identification
- **Audit Logging**: Comprehensive security event logging

## Testing and Validation

### Utility Testing Strategy
- **Unit Tests**: Individual function validation with edge cases
- **Performance Tests**: Benchmark validation for timing utilities
- **Security Tests**: Validation of filtering and sanitization
- **Integration Tests**: End-to-end utility chain testing

### Quality Assurance
- **Type Safety**: Comprehensive type checking and validation
- **Error Handling**: Graceful degradation for all utility functions
- **Documentation**: Comprehensive docstrings and usage examples
- **Performance Regression**: Automated performance monitoring