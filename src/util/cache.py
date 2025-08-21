# src/util/cache.py
"""
TTL LRU cache utilities for performance optimization.
Provides time-based expiration with LRU eviction for slotter and tokenization caching.
"""

import time
from functools import wraps
from typing import Any, Dict, Optional, Tuple, Callable
from threading import Lock


class TTLCache:
    """Thread-safe TTL cache with LRU eviction."""

    def __init__(self, maxsize: int = 1000, ttl_s: int = 300):
        self.maxsize = maxsize
        self.ttl_s = ttl_s
        self._cache: Dict[Any, Tuple[Any, float]] = {}  # key -> (value, expires_at)
        self._access_order: Dict[Any, float] = {}  # key -> last_access_time
        self._lock = Lock()

    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache, return None if expired or missing."""
        with self._lock:
            now = time.time()

            if key in self._cache:
                value, expires_at = self._cache[key]

                # Check expiration
                if now > expires_at:
                    del self._cache[key]
                    self._access_order.pop(key, None)
                    return None

                # Update access time for LRU
                self._access_order[key] = now
                return value

            return None

    def put(self, key: Any, value: Any) -> None:
        """Put value in cache with TTL expiration."""
        with self._lock:
            now = time.time()
            expires_at = now + self.ttl_s

            # Add/update entry
            self._cache[key] = (value, expires_at)
            self._access_order[key] = now

            # Evict if over capacity
            if len(self._cache) > self.maxsize:
                self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict least recently used entries until under capacity."""
        # Sort by access time and remove oldest
        sorted_keys = sorted(self._access_order.items(), key=lambda x: x[1])

        evict_count = len(self._cache) - (
            self.maxsize * 3 // 4
        )  # Evict to 75% capacity

        for key, _ in sorted_keys[:evict_count]:
            self._cache.pop(key, None)
            self._access_order.pop(key, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)


def ttl_lru(maxsize: int = 1000, ttl_s: int = 300):
    """
    Decorator for TTL LRU caching function results.

    Args:
        maxsize: Maximum cache size before LRU eviction
        ttl_s: Time-to-live in seconds

    Returns:
        Decorated function with caching
    """
    cache = TTLCache(maxsize=maxsize, ttl_s=ttl_s)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from args/kwargs
            key = _make_cache_key(args, kwargs)

            # Try cache first
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(key, result)
            return result

        # Expose cache for inspection/clearing
        wrapper.cache = cache
        return wrapper

    return decorator


def _make_cache_key(args: tuple, kwargs: dict) -> tuple:
    """Create hashable cache key from function arguments."""
    # Convert kwargs to sorted tuple for consistent hashing
    kwargs_items = tuple(sorted(kwargs.items())) if kwargs else ()

    # Handle unhashable types
    hashable_args = []
    for arg in args:
        if isinstance(arg, (list, dict, set)):
            if isinstance(arg, list):
                hashable_args.append(tuple(arg))
            elif isinstance(arg, dict):
                hashable_args.append(tuple(sorted(arg.items())))
            elif isinstance(arg, set):
                hashable_args.append(tuple(sorted(arg)))
        else:
            hashable_args.append(arg)

    return (tuple(hashable_args), kwargs_items)


# Specialized caches for common use cases
slot_cache = TTLCache(maxsize=2048, ttl_s=600)  # 10 min slot cache
ce_token_cache = TTLCache(maxsize=8192, ttl_s=300)  # 5 min CE tokenization cache
