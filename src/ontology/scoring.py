"""
Calibrated scoring for link and entity decisions.

Defines feature extraction and scoring stubs for edges/entities with
relation-specific thresholds and explainable signal breakdowns.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# ---------------------
# Feature Vector Schema
# ---------------------

EdgeFeatures = Dict[str, float]


@dataclass
class EdgeScore:
    score: float
    relation: str
    threshold: float
    accepted: bool
    signals: EdgeFeatures = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ---------------------
# Thresholds & Weights
# ---------------------

DEFAULT_WEIGHTS: Dict[str, float] = {
    # step/imperative detection
    "regex_step": 0.15,
    "spacy_imperative": 0.20,
    # ordering & cues
    "bullet_order_consistency": 0.15,
    "cue_word": 0.15,
    # semantic/entity overlap
    "same_entity_overlap": 0.10,
    "embed_sim": 0.15,
    # entailment (optional)
    "nli_entailment": 0.08,
    # context
    "section_header_match": 0.01,
    "same_space": 0.01,
    # domain relations
    "ownership_cue": 0.35,
    "platform_cue": 0.35,
    "diagram_cue": 0.25,
    "header_cue": 0.10,
}

RELATION_THRESHOLDS: Dict[str, float] = {
    # Different relations require different confidence cutoffs
    "NEXT": 0.55,
    "REQUIRES": 0.65,
    "SAME_AS": 0.70,
    "IN_DIVISION": 0.55,
    "OWNS": 0.60,
    "RUNS_ON": 0.60,
    "DOCUMENTED_BY": 0.50,
}


# ---------------------
# Feature Extraction (stubs)
# ---------------------

def extract_edge_features(pair: Dict[str, Any]) -> EdgeFeatures:
    """Build an edge feature vector for a candidate relation.

    pair: {
        'a': Step-like dict with attributes
        'b': Step-like dict with attributes
        'evidence': { 'a_snippet': str, 'b_snippet': str, 'doc_rel': 'intra'|'inter', ... }
        'signals': optional precomputed booleans/scores
    }
    """
    s = pair.get("signals", {})
    features: EdgeFeatures = {
        "regex_step": float(s.get("regex_step", False)),
        "spacy_imperative": float(s.get("spacy_imperative", False)),
        "bullet_order_consistency": float(s.get("bullet_order_consistency", 0.0)),
        "cue_word": float(s.get("cue_word", False)),
        "same_entity_overlap": float(s.get("same_entity_overlap", 0.0)),
        "embed_sim": float(s.get("embed_sim", 0.0)),
        "nli_entailment": float(s.get("nli_entailment", 0.0)),
        "section_header_match": float(s.get("section_header_match", 0.0)),
        "same_space": float(s.get("same_space", False)),
    }
    return features


# ---------------------
# Scoring
# ---------------------

def _linear_score(features: EdgeFeatures, weights: Optional[Dict[str, float]] = None) -> float:
    w = weights or DEFAULT_WEIGHTS
    score = 0.0
    for k, v in features.items():
        score += w.get(k, 0.0) * v
    # Clip to [0, 1] for interpretability
    return max(0.0, min(1.0, score))


def score_edge(
    relation: str,
    pair: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    thresholds: Optional[Dict[str, float]] = None,
) -> EdgeScore:
    """Compute an explainable score for an edge candidate and decide accept/reject.

    - Uses a linear blend (placeholder) that can be replaced by a calibrated model.
    - Enforces minimal-signal rules per relation (NEXT vs REQUIRES).
    """
    features = extract_edge_features(pair)
    score = _linear_score(features, weights)
    th = (thresholds or RELATION_THRESHOLDS).get(relation, 0.6)
    notes: List[str] = []

    # Minimal-signal rules (conservative defaults)
    if relation == "NEXT":
        # Must be intra-doc and have ordering consistency
        if pair.get("doc_id_a") != pair.get("doc_id_b"):
            notes.append("NEXT must be intra-doc; doc_id mismatch")
            accept = False
        else:
            accept = score >= th and features.get("bullet_order_consistency", 0.0) > 0
    elif relation == "REQUIRES":
        # Require cue OR (high entailment + entity overlap)
        cue = features.get("cue_word", 0.0) > 0
        entail = features.get("nli_entailment", 0.0) >= 0.5
        overlap = features.get("same_entity_overlap", 0.0) >= 0.5
        if not (cue or (entail and overlap)):
            notes.append("REQUIRES needs cue or entailment+overlap")
        accept = score >= th and (cue or (entail and overlap))
    elif relation in ("IN_DIVISION", "OWNS", "RUNS_ON", "DOCUMENTED_BY"):
        # rely on domain-specific cues
        accept = score >= th
    else:
        accept = score >= th

    return EdgeScore(
        score=score,
        relation=relation,
        threshold=th,
        accepted=accept,
        signals=features,
        notes=notes,
    )


# ---------------------
# Entity Linking Scoring (stub)
# ---------------------

def score_same_as(mention_a: str, mention_b: str, signals: Dict[str, float]) -> EdgeScore:
    """Score SAME_AS relation between two surface entities.

    Signals may include: embed_sim, lev_ratio, dict_synonym, cooccur, type_match.
    """
    feats: EdgeFeatures = {
        "embed_sim": float(signals.get("embed_sim", 0.0)),
        "lev_ratio": float(signals.get("lev_ratio", 0.0)),
        "dict_synonym": float(signals.get("dict_synonym", 0.0)),
        "type_match": float(signals.get("type_match", 0.0)),
    }
    # Reuse linear model with relation-specific threshold
    score = _linear_score(feats, weights={
        "embed_sim": 0.5,
        "lev_ratio": 0.2,
        "dict_synonym": 0.25,
        "type_match": 0.05,
    })
    th = RELATION_THRESHOLDS.get("SAME_AS", 0.7)
    return EdgeScore(
        score=score,
        relation="SAME_AS",
        threshold=th,
        accepted=score >= th,
        signals=feats,
        notes=[],
    )


# ---------------------
# Calibration (placeholder)
# ---------------------

class Calibrator:
    """Stub for probability calibration (e.g., isotonic or Platt scaling)."""

    def __init__(self):
        self.fitted = False

    def fit(self, scores: List[float], labels: List[int]) -> None:
        # TODO: implement real calibration
        self.fitted = True

    def predict_proba(self, scores: List[float]) -> List[float]:
        if not self.fitted:
            return scores  # identity fallback
        # TODO: apply learned calibration
        return scores
