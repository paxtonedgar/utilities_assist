# src/agent/acronym_map.py
"""
Acronym expansion mapping for Utilities domain.

This map helps disambiguate common acronyms to their full utility names,
preventing incorrect matches (e.g., medical "CIU" instead of Customer Interaction Utility).
"""

# Core utility acronyms and their expansions
UTILITY_ACRONYMS = {
    # Customer-related utilities
    "CIU": "Customer Interaction Utility",
    "CSU": "Customer Summary Utility", 
    "CAU": "Customer Account Utility",
    "CTU": "Customer Transaction Utility",
    
    # Transaction and finance utilities
    "ETU": "Enhanced Transaction Utility",
    "TSU": "Transaction Summary Utility",
    "PTU": "Payment Transaction Utility",
    "FTU": "Financial Transaction Utility",
    
    # Account and balance utilities  
    "ABU": "Account Balance Utility",
    "ASU": "Account Summary Utility",
    "AAU": "Account Activity Utility",
    
    # Other common utilities
    "APU": "API Platform Utility",
    "DSU": "Data Service Utility",
    "RSU": "Reference Service Utility",
    "NSU": "Notification Service Utility",
    
    # API Groups (APG)
    "CAPG": "Customer API Group",
    "TAPG": "Transaction API Group", 
    "AAPG": "Account API Group",
    "PAPG": "Payment API Group"
}

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
    import re
    
    # Normalize query for matching
    query_upper = query.upper()
    words = query_upper.split()
    
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
    
    # Check if any token is an uppercase acronym
    has_acronym = any(
        token.upper() in UTILITY_ACRONYMS 
        for token in tokens
    )
    
    return has_acronym