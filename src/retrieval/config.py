# src/retrieval/config.py
"""
Centralized configuration for P1-P3 performance optimizations.
Single source of truth for budgets, thresholds, and weights.
"""

# P1 - Latency budgets (milliseconds/seconds)
BUDGETS = {
    "slotting_ms_regex": 1,  # Max time for regex-based intent slotting
    "slotting_ms_onnx": 50,  # Max time for ONNX classification (if enabled)
    "rrf_ms": 3,  # Max time for RRF fusion
    "ce_timeout_s": 1.8,  # Cross-encoder timeout in seconds
}

# P1 - Cache settings
CACHES = {
    "slot_ttl_s": 600,  # Slot cache TTL (10 min)
    "ce_token_ttl_s": 300,  # CE tokenization cache TTL (5 min)
    "slot_cache_size": 2048,  # Maximum slot cache entries
    "ce_token_cache_size": 8192,  # Maximum CE token cache entries
}

# P3 - Actionability thresholds
THRESHOLDS = {
    "procedure_min_spans": 3,  # Minimum spans for procedure presenter
    "info_min_passages": 1,  # Minimum passages for info presenter
    "span_confidence_min": 0.3,  # Minimum span confidence to count
}

# P3 - Presenter weights and scoring
WEIGHTS = {
    "step": 1.0,  # Step bullets weight
    "jira": 0.8,  # JIRA/ITSM spans weight
    "owner": 0.6,  # Contact/owner spans weight
    "endpoint": 0.9,  # API endpoint spans weight
    "table": 0.4,  # Table spans weight
}

# View-specific settings
VIEW_SETTINGS = {
    "max_passages_per_view": 8,  # Top-K passages to analyze per view
    "passage_clip_chars": 800,  # Max chars per passage text
    "citation_format": "[{title} ▸ {section}]",  # Citation template
}

# Debug and logging settings
DEBUG_SETTINGS = {
    "log_timeline_ms": True,  # Log stage→ms timeline
    "log_span_details": False,  # Log detailed span extraction
    "log_presenter_choice": True,  # Log presenter selection reasoning
}
