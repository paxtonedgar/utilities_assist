# src/quality/coverage.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math, regex as re, numpy as np
from sentence_transformers import CrossEncoder

_ACTION_PATTERNS = {
    "steps": re.compile(r"(?m)^\s*(?:\d+\.|[-*])\s+\S+"),
    "endpoint": re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\s+\/[^\s#]+"),
    "jira": re.compile(r"\b(jira|servicenow|request\s+form|intake|project\s*key)\b", re.I),
    "owner": re.compile(r"\b(owner|team|contact|dl|email|slack|channel)\b", re.I),
    "table": re.compile(r"\n\|.+\|\n\|[- :|]+\|\n", re.M),  # Markdown table-ish
}

def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

@dataclass
class Passage:
    text: str
    meta: Dict  # should include at least {"rank": int, "url": str, "title": str, "heading": str}

class CoverageGate:
    """
    Computes per-(subquery, passage) answerability and aggregates:
      - Aspect Recall (coverage of sub-queries)
      - alpha-nDCG (diversity-aware gain)
    Then selects passages for composition.
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None,
        weights: Dict[str, float] = None,
        tau: float = 0.45,
        alpha: float = 0.5,
        min_actionable_spans: int = 3,
        gate_ar: float = 0.60,
        gate_andcg: float = 0.40,
    ):
        self.ce = CrossEncoder(model_name, device=device or ("cuda" if self._has_cuda() else "cpu"))
        self.tau = tau
        self.alpha = alpha
        self.min_actionable_spans = min_actionable_spans
        self.gate_ar = gate_ar
        self.gate_andcg = gate_andcg
        self.w = {"w0": 1.0, "steps": 0.20, "endpoint": 0.15, "jira": 0.15, "owner": 0.10, "table": 0.05}
        if weights:
            self.w.update(weights)

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    # ---------- feature helpers ----------
    def _features(self, text: str) -> Dict[str, int]:
        return {k: 1 if p.search(text) else 0 for k, p in _ACTION_PATTERNS.items()}

    # ---------- main API ----------
    def score_matrix(self, subqs: List[str], passages: List[Passage]) -> np.ndarray:
        """Return a matrix [len(subqs) x len(passages)] of a'(q_i, p_j) in [0,1]."""
        if not subqs or not passages:
            return np.zeros((len(subqs), len(passages)), dtype="float32")

        pairs = [(q, p.text) for q in subqs for p in passages]
        # CE returns 0..1 for some models; for others it's a logit-like score. Normalize via sigmoid on-demand.
        raw = np.array(self.ce.predict(pairs), dtype="float32")
        # Ensure shape
        raw = raw.reshape(len(subqs), len(passages))

        # augment with features
        feats = [self._features(p.text) for p in passages]  # list of dicts per passage
        mat = np.zeros_like(raw, dtype="float32")
        for i in range(len(subqs)):
            for j in range(len(passages)):
                ce = float(raw[i, j])
                # If model returns big numbers (logits), squash
                if ce > 1.0 or ce < 0.0:
                    ce = _sigmoid(ce)
                s = self.w["w0"] * ce
                for name in ("steps", "endpoint", "jira", "owner", "table"):
                    s += self.w[name] * feats[j][name]
                mat[i, j] = _sigmoid(s)  # calibrated into 0..1
        return mat

    def aspect_recall(self, mat: np.ndarray) -> float:
        """Fraction of subqs for which some passage crosses tau."""
        if mat.size == 0:
            return 0.0
        covered = (mat.max(axis=1) >= self.tau).sum()
        return float(covered) / mat.shape[0]

    def alpha_ndcg(self, mat: np.ndarray, ranks: List[int], alpha: float = None) -> float:
        """
        Diversity-aware nDCG using per-aspect gains from the score matrix and a ranking.
        ranks: lower is better (1-based). Provide len(passages) ranks consistent with your fusion order.
        """
        if mat.size == 0:
            return 0.0
        alpha = alpha or self.alpha
        # order passages by rank
        order = np.argsort(np.array(ranks))
        gains = []
        # ideal gains (per-aspect sorted)
        ideal = []
        m, n = mat.shape  # m subqs, n passages
        seen = np.zeros(m, dtype="int32")
        for r_idx, j in enumerate(order, start=1):
            # gain for each aspect decreases if already covered
            g_i = ( (1 - alpha) ** seen ) * mat[:, j]
            gains.append(g_i.sum() / math.log2(1 + r_idx))
            # update "seen" as if we pick the max aspect per step
            seen = seen + (mat[:, j] > 0).astype("int32")
        dcg = float(np.sum(gains))

        # Ideal DCG: pick best per step (upper bound by sorting per-aspect maxima)
        ideal_per_aspect = np.sort(mat.max(axis=1))[::-1]
        seen = np.zeros_like(ideal_per_aspect, dtype="int32")
        ideal_terms = []
        for r_idx in range(1, len(order)+1):
            # pick next best remaining aspect (cyclic approx): this upper bounds the real ideal DCG
            i = (r_idx - 1) % len(ideal_per_aspect)
            g = ((1 - alpha) ** seen[i]) * ideal_per_aspect[i]
            ideal_terms.append(g / math.log2(1 + r_idx))
            seen[i] += 1
        idcg = float(np.sum(ideal_terms)) or 1.0
        return dcg / idcg

    def select_passages(self, mat: np.ndarray, subqs: List[str], passages: List[Passage], top_n_per_aspect: int = 1) -> Dict[int, List[int]]:
        """
        For each aspect i, return indices of top passages j (<= top_n_per_aspect).
        """
        pick = {}
        for i in range(len(subqs)):
            order = np.argsort(-mat[i, :])[:top_n_per_aspect]
            pick[i] = [int(j) for j in order if mat[i, j] >= self.tau]
        return pick

    def count_actionable_spans(self, passages: List[Passage]) -> int:
        return sum(any(v for v in self._features(p.text).values()) for p in passages)

    def evaluate(self, user_query: str, subqs: List[str], passages: List[Passage]) -> Dict:
        mat = self.score_matrix(subqs, passages)
        ranks = [int(p.meta.get("rank", idx+1)) for idx, p in enumerate(passages)]
        C_AR = self.aspect_recall(mat)
        C_aNDCG = self.alpha_ndcg(mat, ranks)
        actionable = self.count_actionable_spans(passages)

        decision = (C_AR >= self.gate_ar) or (C_aNDCG >= self.gate_andcg)
        decision = decision and (actionable >= self.min_actionable_spans)

        picks = self.select_passages(mat, subqs, passages, top_n_per_aspect=1)

        return {
            "answerability_matrix": mat,
            "aspect_recall": C_AR,
            "alpha_ndcg": C_aNDCG,
            "actionable_spans": actionable,
            "gate_pass": decision,
            "picks": picks,  # indices per subq
        }