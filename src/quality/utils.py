# src/quality/utils.py
"""
Centralized utilities for coverage evaluation system.
Eliminates code duplication across router.py and search_to_compose.py
"""

from typing import List, Dict, Any
from .coverage import Passage

# Single global coverage gate instance (eliminating duplication)
_COVERAGE_GATE = None








