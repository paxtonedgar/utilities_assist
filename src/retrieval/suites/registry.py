# src/retrieval/suites/registry.py
"""
Suite registry for loading and managing tool suite manifests.
Provides centralized detection of actionable spans across multiple tool suites.
"""

import yaml
import logging
from pathlib import Path
from typing import List, Set

from .schemas import (
    SuiteManifest,
    ExtractorConfig,
    Span,
    CAPABILITY_TO_TYPE,
    CAPABILITY_WEIGHTS,
)
from src.services.models import SearchResult as Passage

logger = logging.getLogger(__name__)


class SuiteRegistry:
    """Registry for managing tool suite manifests and span detection."""

    def __init__(self, manifest_dir: Path = None):
        self.manifest_dir = manifest_dir or Path(__file__).parent / "manifests"
        self.manifests: List[SuiteManifest] = []
        self._loaded = False

    def load_manifests(self) -> List[SuiteManifest]:
        """Load all YAML manifests from the manifest directory."""
        if self._loaded:
            return self.manifests

        self.manifests = []

        if not self.manifest_dir.exists():
            logger.warning(f"Manifest directory not found: {self.manifest_dir}")
            return self.manifests

        for yaml_file in self.manifest_dir.glob("*.yml"):
            try:
                manifest = self._load_single_manifest(yaml_file)
                if manifest:
                    self.manifests.append(manifest)
                    logger.debug(f"Loaded suite manifest: {manifest.suite}")
            except Exception as e:
                logger.error(f"Failed to load manifest {yaml_file}: {e}")

        logger.info(f"Loaded {len(self.manifests)} suite manifests")
        self._loaded = True
        return self.manifests

    def _load_single_manifest(self, yaml_file: Path) -> SuiteManifest:
        """Load and validate a single YAML manifest file."""
        with open(yaml_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Convert extractor configs
        extractors = []
        for ext_data in data.get("extractors", []):
            extractor = ExtractorConfig(**ext_data)
            extractors.append(extractor)

        # Create manifest
        manifest = SuiteManifest(
            suite=data["suite"], hints=data.get("hints", []), extractors=extractors
        )

        return manifest

    def detect_spans(self, passages: List[Passage]) -> List[Span]:
        """
        Detect actionable spans across all loaded suites.

        Args:
            passages: List of document passages to analyze

        Returns:
            List of detected actionable spans, deduplicated and ranked
        """
        if not self._loaded:
            self.load_manifests()

        spans = []

        # Run suite-specific detection
        for suite_manifest in self.manifests:
            suite_confidence = suite_manifest.score_hints(passages)

            if suite_confidence < 0.1:  # Skip suites with very low confidence
                continue

            for extractor in suite_manifest.extractors:
                suite_spans = self._extract_with_extractor(
                    passages, extractor, suite_manifest.suite, suite_confidence
                )
                spans.extend(suite_spans)

        # Add global patterns (steps, tables, generic contacts)
        global_spans = self._extract_global_patterns(passages)
        spans.extend(global_spans)

        # Deduplicate and rank
        final_spans = self._dedupe_and_rank(spans)

        logger.debug(
            f"Detected {len(final_spans)} actionable spans from {len(passages)} passages"
        )

        return final_spans

    def _extract_with_extractor(
        self,
        passages: List[Passage],
        extractor: ExtractorConfig,
        suite: str,
        suite_confidence: float,
    ) -> List[Span]:
        """Extract spans using a single extractor configuration."""
        spans = []

        for passage in passages:
            text = f"{passage.title}\n{passage.text}"

            # Try each compiled pattern
            for pattern in extractor._compiled_patterns:
                for match in pattern.finditer(text):
                    # Extract attributes from named groups
                    attrs = {}
                    for attr_name, attr_pattern in extractor.attrs.items():
                        attr_match = pattern.search(match.group(0))
                        if attr_match and attr_match.groups():
                            attrs[attr_name] = attr_match.group(1).strip()

                    # Determine best URL (extracted or passage URL)
                    best_url = self._extract_best_url(passage, match, extractor)

                    # Calculate confidence
                    confidence = min(1.0, suite_confidence * extractor.weight)

                    # Create span
                    span = Span(
                        type=CAPABILITY_TO_TYPE.get(extractor.capability, "endpoint"),
                        suite=suite,
                        capability=extractor.capability,
                        text=match.group(0).strip(),
                        url=best_url,
                        doc_id=passage.doc_id,
                        offset=match.start(),
                        confidence=confidence,
                        attrs=attrs,
                    )

                    spans.append(span)

        return spans

    def _extract_best_url(
        self, passage: Passage, match, extractor: ExtractorConfig
    ) -> str:
        """Extract the best URL for a span (from URL patterns or passage URL)."""
        # Try URL extraction patterns first
        for url_pattern in extractor._compiled_url_patterns:
            url_match = url_pattern.search(match.group(0))
            if url_match:
                return url_match.group(1)

        # Check if match contains a URL
        import re

        url_in_match = re.search(r"https?://[^\s]+", match.group(0))
        if url_in_match:
            return url_in_match.group(0)

        # Fall back to passage URL
        return passage.url

    def _extract_global_patterns(self, passages: List[Passage]) -> List[Span]:
        """Extract spans using global patterns (steps, tables, generic contacts)."""
        import re

        spans = []

        # Global pattern definitions
        GLOBAL_PATTERNS = {
            "step": [
                re.compile(r"(?m)^\s*(\d+[\.\)]|\(\d+\)|[-–•*])\s+\S.+", re.MULTILINE),
                re.compile(r"(?i)^\s*step\s*\d+\s*:", re.MULTILINE),
            ],
            "table": [
                re.compile(r"(?m)^\|.+\|.+\|", re.MULTILINE),
            ],
            "owner": [
                re.compile(
                    r"(?i)\b(owner|team|contact|dl|distribution\s*list|support):\s*([^\n]+)",
                    re.IGNORECASE,
                ),
                re.compile(r"(?i)\b(email|contact):\s*([^\n]+)", re.IGNORECASE),
            ],
        }

        for passage in passages:
            text = f"{passage.title}\n{passage.text}"

            for span_type, patterns in GLOBAL_PATTERNS.items():
                for pattern in patterns:
                    for match in pattern.finditer(text):
                        span = Span(
                            type=span_type,
                            suite="global",
                            capability=span_type,
                            text=match.group(0).strip(),
                            url=passage.url,
                            doc_id=passage.doc_id,
                            offset=match.start(),
                            confidence=0.6,  # Moderate confidence for global patterns
                            attrs={},
                        )
                        spans.append(span)

        return spans

    def _dedupe_and_rank(self, spans: List[Span]) -> List[Span]:
        """Deduplicate overlapping spans and rank by confidence."""
        if not spans:
            return spans

        # Sort by position for overlap detection
        sorted_spans = sorted(spans, key=lambda s: (s.doc_id, s.offset))

        # Remove near-duplicates (overlapping spans from same doc)
        deduped = []
        for span in sorted_spans:
            # Check if this span significantly overlaps with recent spans
            overlaps = False
            for existing in deduped[-3:]:  # Check last few spans
                if (
                    span.doc_id == existing.doc_id
                    and abs(span.offset - existing.offset) < 50  # Within 50 chars
                    and span.suite == existing.suite
                ):
                    overlaps = True
                    break

            if not overlaps:
                deduped.append(span)

        # Sort by confidence (descending)
        ranked = sorted(deduped, key=lambda s: s.confidence, reverse=True)

        return ranked


def actionable_score(spans: List[Span]) -> float:
    """
    Calculate actionability score from detected spans.

    Args:
        spans: List of detected actionable spans

    Returns:
        Weighted actionability score (higher = more actionable)
    """
    if not spans:
        return 0.0

    total_score = 0.0
    seen_capabilities: Set[str] = set()

    for span in spans:
        # Get base weight for capability
        base_weight = CAPABILITY_WEIGHTS.get(span.capability, 0.3)

        # Apply confidence multiplier
        weighted_score = base_weight * span.confidence

        # Reduce weight for duplicate capabilities (diminishing returns)
        if span.capability in seen_capabilities:
            weighted_score *= 0.5

        total_score += weighted_score
        seen_capabilities.add(span.capability)

    return round(total_score, 2)


def actionable_count(spans: List[Span]) -> int:
    """Count actionable spans excluding tables (for backward compatibility)."""
    return len([s for s in spans if s.type != "table"])


# Global registry instance
_registry = SuiteRegistry()


def detect_spans(passages: List[Passage]) -> List[Span]:
    """Convenience function for span detection."""
    return _registry.detect_spans(passages)


def get_suite_registry() -> dict:
    """Get suite registry as dict for coloring system."""
    _registry.load_manifests()
    registry_dict = {}
    for manifest in _registry.manifests:
        # Extract keywords from extractor patterns (simplified)
        keywords = []
        url_patterns = []
        
        for extractor in manifest.extractors:
            # Basic keyword extraction from patterns - very simplified
            for pattern in extractor.patterns:
                # Extract simple word patterns (not full regex parsing)
                if not any(char in pattern for char in r'[](){}*+?|^$\\'): 
                    keywords.append(pattern.lower())
            url_patterns.extend(extractor.url_patterns)
        
        registry_dict[manifest.suite] = {
            "keywords": keywords,
            "url_patterns": url_patterns,
            "hints": {"any": manifest.hints}
        }
    return registry_dict


def get_suite_names() -> List[str]:
    """Get names of all loaded suites."""
    _registry.load_manifests()
    return [m.suite for m in _registry.manifests]
