# src/agent/acronym_map.py
"""
Dynamic acronym expansion using actual corpus data.

Loads acronym mappings from data/synonyms.json and data/swagger_keyword.json
to ensure consistency with the actual API and utility names in the system.
"""

import json
from typing import Dict, List
from pathlib import Path

# Cache for loaded data
_ACRONYM_CACHE = None
_API_DATA_CACHE = None


def _load_acronym_data() -> Dict[str, str]:
    """Load acronym mappings from data files."""
    global _ACRONYM_CACHE

    if _ACRONYM_CACHE is not None:
        return _ACRONYM_CACHE

    acronyms = {}

    # Get project root directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent  # src/agent -> src -> project_root

    # Load from synonyms.json
    synonyms_path = project_root / "data" / "synonyms.json"
    if synonyms_path.exists():
        with open(synonyms_path, "r") as f:
            synonyms = json.load(f)
            # Convert all keys to uppercase for consistent matching
            for key, value in synonyms.items():
                acronyms[key.upper()] = value

    # Load from swagger_keyword.json for additional mappings
    swagger_path = project_root / "data" / "swagger_keyword.json"
    if swagger_path.exists():
        with open(swagger_path, "r") as f:
            swagger_data = json.load(f)
            if (
                "Product List" in swagger_data
                and "Utilities" in swagger_data["Product List"]
            ):
                for apg in swagger_data["Product List"]["Utilities"]:
                    if "Acronyms" in apg and "APG_Name" in apg:
                        acronym = apg["Acronyms"].upper()
                        full_name = apg["APG_Name"]
                        # Don't override if already exists from synonyms.json
                        if acronym not in acronyms:
                            acronyms[acronym] = full_name

    _ACRONYM_CACHE = acronyms
    return acronyms


def _load_api_data() -> Dict:
    """Load full API data for detailed lookups."""
    global _API_DATA_CACHE

    if _API_DATA_CACHE is not None:
        return _API_DATA_CACHE

    api_data = {"utilities": {}, "products": []}

    # Get project root directory
    current_file = Path(__file__)
    project_root = current_file.parent.parent.parent

    # Load swagger_keyword.json for API listings
    swagger_path = project_root / "data" / "swagger_keyword.json"
    if swagger_path.exists():
        with open(swagger_path, "r") as f:
            swagger_data = json.load(f)
            api_data = swagger_data

    _API_DATA_CACHE = api_data
    return api_data


# Dynamically loaded acronym map
UTILITY_ACRONYMS = _load_acronym_data()


def expand_acronym(query: str) -> tuple[str, list[str]]:
    """
    Expand acronyms in query to their full names.

    Args:
        query: Raw user query

    Returns:
        Tuple of (expanded_query, list_of_expansions)

    Example:
        "what is CIU" -> ("what is Customer Interaction Utility CIU", ["Customer Interaction Utility"])
    """

    # Normalize query for matching
    query_upper = query.upper()

    expansions = []
    expanded_parts = []

    for word in query.split():
        word_upper = word.upper()
        # Check if word is a known acronym
        if word_upper in UTILITY_ACRONYMS:
            expansion = UTILITY_ACRONYMS[word_upper]
            expansions.append(expansion)
            # Keep both acronym and expansion for better matching
            expanded_parts.append(f"{expansion} {word_upper}")
        else:
            expanded_parts.append(word)

    expanded_query = " ".join(expanded_parts)

    return expanded_query, expansions


def is_short_acronym_query(query: str) -> bool:
    """
    Check if query is a short acronym query (≤3 tokens, mostly uppercase).

    These queries need special handling with title boosting.
    """
    if not query:
        return False

    tokens = query.strip().split()

    # Check if query is ≤3 tokens
    if len(tokens) > 3:
        return False

    # Reload acronyms if needed
    acronyms = _load_acronym_data()

    # Check if any token is an uppercase acronym
    has_acronym = any(token.upper() in acronyms for token in tokens)

    return has_acronym


def get_apis_for_acronym(acronym: str) -> List[str]:
    """
    Get list of API names associated with an acronym.

    Args:
        acronym: The acronym to lookup (e.g., "CIU", "ETU")

    Returns:
        List of API names for that utility
    """
    api_data = _load_api_data()
    acronym_upper = acronym.upper()

    if "Product List" in api_data and "Utilities" in api_data["Product List"]:
        for apg in api_data["Product List"]["Utilities"]:
            if apg.get("Acronyms", "").upper() == acronym_upper:
                return apg.get("API-Names-List", [])

    return []


def get_all_utility_apis() -> Dict[str, List[str]]:
    """
    Get all utility APIs grouped by their APG name.

    Returns:
        Dictionary mapping APG names to lists of API names
    """
    api_data = _load_api_data()
    result = {}

    if "Product List" in api_data and "Utilities" in api_data["Product List"]:
        for apg in api_data["Product List"]["Utilities"]:
            apg_name = apg.get("APG_Name", "Unknown")
            api_list = apg.get("API-Names-List", [])
            result[apg_name] = api_list

    return result
