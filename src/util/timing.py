# src/util/timing.py
"""
Simple timing decorator for P1 latency budgets.
"""

import time
import functools
import logging
from typing import Callable, Any
from src.retrieval.config import BUDGETS

logger = logging.getLogger(__name__)


def time_budget(budget_key: str, warn_only: bool = True):
    """
    Decorator to enforce P1 latency budgets.

    Args:
        budget_key: Key in BUDGETS config dict (e.g., "slotting_ms_regex")
        warn_only: If True, only log warnings. If False, raise TimeoutError.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            budget_ms = BUDGETS.get(budget_key, 1000)  # Default 1s if not found
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                if elapsed_ms > budget_ms:
                    msg = f"{func.__name__} exceeded budget: {elapsed_ms:.1f}ms > {budget_ms}ms"
                    if warn_only:
                        logger.warning(msg)
                    else:
                        raise TimeoutError(msg)
                else:
                    logger.debug(
                        f"{func.__name__}: {elapsed_ms:.1f}ms (budget: {budget_ms}ms)"
                    )

                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(f"{func.__name__} failed after {elapsed_ms:.1f}ms: {e}")
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            budget_ms = BUDGETS.get(budget_key, 1000)
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                elapsed_ms = (time.time() - start_time) * 1000

                if elapsed_ms > budget_ms:
                    msg = f"{func.__name__} exceeded budget: {elapsed_ms:.1f}ms > {budget_ms}ms"
                    if warn_only:
                        logger.warning(msg)
                    else:
                        raise TimeoutError(msg)
                else:
                    logger.debug(
                        f"{func.__name__}: {elapsed_ms:.1f}ms (budget: {budget_ms}ms)"
                    )

                return result
            except Exception as e:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.error(f"{func.__name__} failed after {elapsed_ms:.1f}ms: {e}")
                raise

        # Return appropriate wrapper based on function type
        return (
            async_wrapper
            if hasattr(func, "__call__") and hasattr(func, "__await__")
            else sync_wrapper
        )

    return decorator


# Convenience decorators for common budgets
def slot_timing(func):
    """Decorator for slot timing budget."""
    return time_budget("slotting_ms_regex")(func)


def rrf_timing(func):
    """Decorator for RRF timing budget."""
    return time_budget("rrf_ms")(func)
