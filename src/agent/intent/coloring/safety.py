"""Safety detector - PII/credential detection with false-positive guards."""

import logging
from typing import Set

logger = logging.getLogger(__name__)


class SafetyDetector:
    """
    Single responsibility: detect PII/security concerns with false-positive guards.
    
    Conservative detection to avoid over-masking:
    - cred/secrets: keyword + plausible token pattern (≥16 chars) co-occur
    - pii: email patterns, SSN patterns
    - Never log raw matched substrings (only flag names)
    """
    
    def __init__(self, thresholds: dict):
        """Initialize with thresholds from centralized config."""
        self.thresholds = thresholds
    
    def detect(self, text: str) -> Set[str]:
        """
        Detect safety concerns in text.
        
        Args:
            text: User query text
            
        Returns:
            Set of safety flags: {\"pii\", \"cred\", \"secrets\"}
            
        Examples:
            >>> detector = SafetyDetector({\"cred_min_token_length\": 16})
            >>> flags = detector.detect(\"user@example.com\")
            >>> \"pii\" in flags
            True
        """
        flags = set()
        
        # PII detection
        if self._has_email_patterns(text):
            flags.add(\"pii\")
        
        if self._has_ssn_patterns(text):
            flags.add(\"pii\")
        
        # Credential detection (with false-positive guard)
        if self._has_credential_patterns(text):
            flags.add(\"cred\")
        
        # Secret detection (keyword + token co-occurrence)
        if self._has_secret_patterns(text):
            flags.add(\"secrets\")
        
        # Log only flag names, never raw content
        if flags:
            logger.info(f\"Safety flags detected: {flags} (content not logged)\")
        
        return flags
    
    def _has_email_patterns(self, text: str) -> bool:
        \"\"\"Detect email patterns.\"\"\"
        from .patterns import has_pattern_match
        return has_pattern_match(text, \"email\")
    
    def _has_ssn_patterns(self, text: str) -> bool:
        \"\"\"Detect SSN patterns (with basic validation).\"\"\"
        from .patterns import extract_pattern_matches
        
        ssn_matches = extract_pattern_matches(text, \"ssn\")
        
        # Basic validation to reduce false positives
        for ssn in ssn_matches:
            # Remove hyphens and check if it's not obviously fake
            digits = ssn.replace(\"-\", \"\")
            if len(digits) == 9 and not digits.startswith((\"000\", \"666\", \"9\")):
                return True
        
        return False
    
    def _has_credential_patterns(self, text: str) -> bool:
        \"\"\"Detect credential patterns with false-positive guard.\"\"\"
        from .patterns import has_pattern_match, extract_pattern_matches
        
        # Must have both keyword and plausible token
        if not has_pattern_match(text, \"credential_keywords\"):
            return False
        
        # Extract token patterns and validate length
        token_matches = extract_pattern_matches(text, \"token_pattern\")
        
        for token_match in token_matches:
            # Extract the token part after the keyword
            token_part = token_match.split(\"=\", 1)[-1].split(\":\", 1)[-1].strip()
            
            # Check minimum length threshold
            if len(token_part) >= self.thresholds[\"cred_min_token_length\"]:
                return True
        
        return False
    
    def _has_secret_patterns(self, text: str) -> bool:
        \"\"\"Detect secret patterns (simple keyword-based).\"\"\"
        text_lower = text.lower()
        
        # Look for obvious secret indicators
        secret_indicators = [\"password=\", \"token=\", \"secret=\", \"key=\", \"auth=\"]
        
        for indicator in secret_indicators:
            if indicator in text_lower:
                # Check if there's content after the indicator
                idx = text_lower.find(indicator)
                after_indicator = text[idx + len(indicator):].strip()
                
                # Must have some content that looks like a token
                if len(after_indicator) >= 8 and not after_indicator.isdigit():
                    return True
        
        return False