from __future__ import annotations

"""Contrastive scoring and MMR selection for taxonomy term candidates."""

import logging
import math
import random
from collections import defaultdict
from statistics import median
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .config import TaxonomyConfig
from .schemas import ContrastiveMetrics, LabelPrototype, ScoredTerm, TermCandidate

logger = logging.getLogger(__name__)


def _normalize(vec: Iterable[float]) -> np.ndarray:
    arr = np.array(list(vec), dtype=float)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr
    return arr / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _build_parent_map(labels: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for label in labels:
        if "." in label:
            mapping[label] = label.rsplit(".", 1)[0]
    return mapping


def _mmr_select(
    vectors: np.ndarray,
    relevance: np.ndarray,
    k: int,
    lam: float,
) -> List[int]:
    if len(vectors) == 0:
        return []
    selected: List[int] = []
    remaining = list(range(len(vectors)))
    sims = np.clip(vectors @ vectors.T, 0.0, 1.0)
    while remaining and len(selected) < k:
        if not selected:
            idx = max(remaining, key=lambda j: (relevance[j], -j))
        else:
            def objective(j: int) -> float:
                redundancy = float(np.max(sims[j, selected])) if selected else 0.0
                return lam * relevance[j] - (1.0 - lam) * redundancy

            idx = max(remaining, key=objective)
        selected.append(idx)
        remaining.remove(idx)
    return selected


def score_candidates(
    prototypes: List[LabelPrototype],
    candidates: List[TermCandidate],
    config: TaxonomyConfig,
) -> Tuple[List[ScoredTerm], ContrastiveMetrics]:
    """Score term candidates against label prototypes and apply selection logic."""

    if not prototypes or not candidates:
        return [], ContrastiveMetrics()

    random.seed(config.random_seed)

    proto_vectors = {proto.class_id: _normalize(proto.embedding) for proto in prototypes}
    parent_map = _build_parent_map(list(proto_vectors.keys()))
    children_by_parent: Dict[str, List[str]] = defaultdict(list)
    for label, parent in parent_map.items():
        children_by_parent[parent].append(label)

    scored_terms: List[ScoredTerm] = []
    margins: List[float] = []
    specifics: List[float] = []
    parent_deltas: List[float] = []

    # Organize candidates per label
    by_label: Dict[str, List[TermCandidate]] = defaultdict(list)
    for cand in candidates:
        if cand.class_id in proto_vectors:
            by_label[cand.class_id].append(cand)

    for label in sorted(by_label.keys()):
        label_vec = proto_vectors.get(label)
        if label_vec is None or label_vec.size == 0:
            continue
        sibs = [proto_vectors[sib] for sib in children_by_parent.get(parent_map.get(label, ""), []) if sib != label]
        parent_vec = proto_vectors.get(parent_map.get(label, "") or "")

        candidates_for_label = by_label[label]
        if not candidates_for_label:
            continue

        cand_vectors = np.vstack([_normalize(c.embedding) for c in candidates_for_label])
        relevance = cand_vectors @ label_vec

        sibling_means: List[float] = []
        for row in cand_vectors:
            if sibs:
                sib_scores = [float(np.dot(row, sib_vec)) for sib_vec in sibs]
                sibling_means.append(float(np.mean(sib_scores)))
            else:
                sibling_means.append(0.0)

        selected_idx = _mmr_select(
            cand_vectors,
            relevance,
            min(config.mmr_k, len(candidates_for_label)),
            config.mmr_lambda,
        )

        for idx, candidate in enumerate(candidates_for_label):
            own = float(relevance[idx])
            sibling_mean = sibling_means[idx]
            parent_sim = float(np.dot(cand_vectors[idx], parent_vec)) if parent_vec is not None else 0.0
            margin = own - sibling_mean
            specificity = own - parent_sim
            parent_delta = parent_sim - sibling_mean

            selected = (
                idx in selected_idx
                and margin >= config.margin_min
                and (specificity >= config.specificity_min or parent_sim >= config.parent_similarity_min)
            )

            assignment = "child"
            if parent_vec is not None and specificity < config.specificity_min and parent_sim >= config.parent_similarity_min:
                assignment = "parent"

            scores = {
                "similarity_self": round(own, 4),
                "similarity_parent": round(parent_sim, 4),
                "similarity_siblings": round(sibling_mean, 4),
                "contrastive_margin": round(margin, 4),
                "specificity_margin": round(specificity, 4),
                "parent_delta": round(parent_delta, 4),
            }

            scored_terms.append(
                ScoredTerm(
                    term_id=candidate.term_id,
                    surface=candidate.surface,
                    class_id=label if assignment == "child" else parent_map.get(label, label),
                    scores=scores,
                    selected=selected,
                    assignment=assignment,
                    evidence=candidate.contexts,
                    frequency=candidate.frequency,
                    doc_ids=candidate.doc_ids,
                )
            )

            margins.append(margin)
            specifics.append(specificity)
            parent_deltas.append(parent_delta)

    if not scored_terms:
        return [], ContrastiveMetrics()

    def _safe_median(values: List[float]) -> float:
        if not values:
            return 0.0
        return float(median(values))

    margins_sorted = sorted(margins)
    p10_index = max(0, math.floor(0.1 * (len(margins_sorted) - 1)))
    metrics = ContrastiveMetrics(
        total_terms=len(scored_terms),
        selected_terms=sum(1 for term in scored_terms if term.selected),
        margin_median=_safe_median(margins),
        margin_p10=float(margins_sorted[p10_index]) if margins_sorted else 0.0,
        specificity_median=_safe_median(specifics),
        parent_delta_median=_safe_median(parent_deltas),
    )

    logger.info(
        "Contrastive scoring produced %d selected terms (median margin %.3f)",
        metrics.selected_terms,
        metrics.margin_median,
    )

    return scored_terms, metrics
