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
    
    # View building logic (uses colors for planning)
    "build_procedure_if": {
        "doish": True,                               # Always if doish=True
        "intent_mixed": True,                        # Always if intent=mixed
        "actionability_threshold": 1.0,              # Build if actionability ≥ 1.0
        "suite_affinity_threshold": 0.6,             # Build if any suite ≥ 0.6
        "required_artifacts": {"endpoint", "ticket", "form", "runbook"},  # OR logic
    },
}

# Debug and logging settings
DEBUG_SETTINGS = {
    "log_timeline_ms": True,  # Log stage→ms timeline
    "log_span_details": False,  # Log detailed span extraction
    "log_presenter_choice": True,  # Log presenter selection reasoning
}

# Coloring system configuration
COLORING_WEIGHTS = {
    # Actionability scoring (unit steps toward 3.0 max)
    "api_endpoint": 1.0,        # POST /path → +1.0
    "jira_ticket": 1.0,         # concrete ticket verbs → +1.0  
    "runbook_step": 1.0,        # explicit step/procedure cues → +1.0
    "form_pipeline": 0.5,       # forms/pipelines/channels → +0.5 each
    "secondary_artifacts": 1.0,  # max from secondary artifacts
    
    # Suite affinity scoring
    "suite_hints_any": 0.5,     # regex match from hints.any → +0.5
    "suite_keywords": 0.2,      # keyword match → +0.2  
    "suite_url_patterns": 0.8,  # URL pattern match → +0.8
}

COLORING_THRESHOLDS = {
    # Actionability gates
    "procedure_build_score": 1.0,     # Min actionability to build procedure view
    "suite_affinity_strong": 0.6,     # Min suite score for strong affinity
    
    # Specificity classification  
    "specificity_med_anchors": 1,      # Min anchors for "med" specificity
    "specificity_high_anchors": 2,     # Min anchors for "high" specificity
    
    # Time urgency detection
    "urgency_soft_keywords": 1,        # Min urgent keywords for "soft"
    "urgency_hard_keywords": 2,        # Min urgent keywords for "hard"
    
    # Safety detection
    "cred_min_token_length": 16,       # Min chars for credential pattern
}

# Retrieval tuning knobs (used by tune_for_colors)
TUNING_ADJUSTMENTS = {
    "specificity_high": {
        "knn_k_max": 24,           # Cap kNN k for high specificity
        "bm25_size_max": 40,       # Cap BM25 size for high specificity
    },
    "specificity_low": {
        "knn_k_min": 36,           # Min kNN k for low specificity  
        "bm25_size_min": 60,       # Min BM25 size for low specificity
    },
    "time_urgency_hard": {
        "ce_timeout_factor": 0.6,   # Reduce CE timeout by 40%
        "skip_rerank_if_time_low": True,
    },
}
