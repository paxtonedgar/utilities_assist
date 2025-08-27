# src/agent/nodes/search_to_compose.py
"""
Search result processing with coverage gate integration.
Replaces the old handwritten coverage math with academic IR-style assessment.
"""

import logging
from typing import List
from src.quality.coverage import Passage

logger = logging.getLogger(__name__)


# NOTE: This parallel node has been removed and integrated into the existing
# CoverageChecker in router.py to follow DRY/SOLID principles




