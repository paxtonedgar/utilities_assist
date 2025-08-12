"""Text normalization service."""

import re
import json
import logging
from typing import Dict, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


# Optional synonym caching - can be disabled if synonyms change frequently  
@lru_cache(maxsize=1)  # Single synonym dict only
def _load_synonyms() -> Dict[str, str]:
    """Load and cache synonym mappings."""
    try:
        # Try to load synonyms from data directory
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        synonyms_path = os.path.join(base_dir, "data", "synonyms.json")
        
        if os.path.exists(synonyms_path):
            with open(synonyms_path, 'r', encoding='utf-8') as f:
                synonyms = json.load(f)
                logger.info(f"Loaded {len(synonyms)} synonym mappings")
                return synonyms
        else:
            logger.warning(f"Synonyms file not found at {synonyms_path}")
    except Exception as e:
        logger.error(f"Failed to load synonyms: {e}")
    
    # Fallback to basic synonyms
    return {
        "csu": "Customer Summary Utility",
        "customer summary": "Customer Summary Utility",
        "gcp": "Global Customer Platform",
        "etu": "Enhanced Transaction Utility",
        "transaction utility": "Enhanced Transaction Utility",
        "au": "Account Utility",
        "ciu": "Customer Interaction Utility",
        "de": "Digital Events",
        "pcu": "Product Catalog Utility",
        "digev": "Digital Events",
        "apg": "APG"
    }


def normalize_query(text: str) -> str:
    """Normalize user input by replacing synonyms and cleaning text.
    
    Args:
        text: Raw user input text
        
    Returns:
        Normalized text with synonyms expanded and cleaned
    """
    if not text or not text.strip():
        return ""
    
    # Get synonym mappings
    synonym_mapping = _load_synonyms()
    
    # Convert all synonym keys to lowercase for case-insensitive matching
    lower_synonyms = {key.lower(): value for key, value in synonym_mapping.items()}
    
    # Build regex pattern for all synonyms (case-insensitive)
    if not lower_synonyms:
        return text.strip()
    
    pattern = r'\b(' + '|'.join(re.escape(synonym) for synonym in lower_synonyms.keys()) + r')\b'
    
    def replace_match(match):
        """Replace matched synonym with canonical form."""
        return lower_synonyms[match.group(0).lower()]
    
    # Apply synonym replacement
    normalized_text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
    
    # Additional normalization
    normalized_text = normalized_text.strip()
    normalized_text = re.sub(r'\s+', ' ', normalized_text)  # Normalize whitespace
    
    if normalized_text != text.strip():
        logger.info(f"Normalized '{text.strip()}' -> '{normalized_text}'")
    
    return normalized_text