from __future__ import annotations

"""Build gazetteer entries from scored taxonomy terms."""

import hashlib
import logging
from typing import List

try:  # pragma: no cover - library optional during unit tests
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - fallback if rapidfuzz missing
    fuzz = None  # type: ignore

from .config import TaxonomyConfig
from .schemas import GazetteerEntry, ScoredTerm

logger = logging.getLogger(__name__)


def _make_entity_id(surface: str, class_id: str) -> str:
    digest = hashlib.sha1(f"{class_id}:{surface}".encode("utf-8")).hexdigest()
    return f"TERM:{digest[:12]}"


def _candidate_aliases(surface: str) -> List[str]:
    variants = {surface}
    variants.add(surface.lower())
    variants.add(surface.title())
    if surface.endswith("s"):
        variants.add(surface[:-1])
    else:
        variants.add(f"{surface}s")
    return [v for v in variants if v]


def build_gazetteer(
    scored_terms: List[ScoredTerm], config: TaxonomyConfig
) -> List[GazetteerEntry]:
    """Select high-confidence terms and convert them into gazetteer entries."""

    if not scored_terms:
        return []

    selected_terms = [term for term in scored_terms if term.selected]
    if not selected_terms:
        logger.info("No selected terms available for gazetteer build")
        return []

    entries: List[GazetteerEntry] = []
    seen: List[str] = []
    for term in sorted(
        selected_terms,
        key=lambda t: (
            -t.scores.get("contrastive_margin", 0.0),
            -t.scores.get("specificity_margin", 0.0),
            t.surface,
        ),
    ):
        surface = term.surface.strip()
        if not surface:
            continue

        if fuzz is not None:
            duplicate = False
            for prior in seen:
                if fuzz.token_set_ratio(surface, prior) >= config.dedupe_ratio:
                    duplicate = True
                    break
            if duplicate:
                continue
        seen.append(surface)

        aliases = _candidate_aliases(surface)
        entry = GazetteerEntry(
            entity_id=_make_entity_id(surface, term.class_id),
            class_id=term.class_id,
            preferred_name=surface,
            aliases=aliases,
            evidence_contexts=term.evidence[: config.contexts_export],
            source=f"contrastive_miner@{config.taxonomy_version}",
            status="active" if term.scores.get("contrastive_margin", 0.0) >= config.margin_min else "candidate",
            scores={k: float(v) for k, v in term.scores.items()},
        )
        entries.append(entry)

    logger.info("Built %d gazetteer entries", len(entries))
    return entries

