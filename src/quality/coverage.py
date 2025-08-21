# src/quality/coverage.py
from dataclasses import dataclass
from typing import List, Dict
import math
import regex as re
import numpy as np

# Use our BGE reranker instead of sentence_transformers
from src.services.reranker import get_reranker, is_reranker_available

_ACTION_PATTERNS = {
    "steps": re.compile(r"(?m)^\s*(?:\d+\.|[-*])\s+\S+"),
    "endpoint": re.compile(r"\b(GET|POST|PUT|DELETE|PATCH)\s+\/[^\s#]+"),
    "jira": re.compile(
        r"\b(jira|servicenow|request\s+form|intake|project\s*key)\b", re.I
    ),
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
    Computes per-(subquery, passage) answerability using BGE v2-m3 reranker and aggregates:
      - Aspect Recall (coverage of sub-queries)
      - alpha-nDCG (diversity-aware gain)
    Then selects passages for composition.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        weights: Dict[str, float] = None,
        tau: float = 0.25,  # Updated for BGE reranker scores
        alpha: float = 0.5,
        min_actionable_spans: int = 3,
        gate_ar: float = 0.60,
        gate_andcg: float = 0.40,
    ):
        self.model_name = model_name
        self.device = device
        self.tau = tau
        self.alpha = alpha
        self.min_actionable_spans = min_actionable_spans
        self.gate_ar = gate_ar
        self.gate_andcg = gate_andcg
        self.w = {
            "w0": 1.0,
            "steps": 0.20,
            "endpoint": 0.15,
            "jira": 0.15,
            "owner": 0.10,
            "table": 0.05,
        }
        if weights:
            self.w.update(weights)

        # Initialize BGE reranker
        self.reranker = None
        self._init_reranker()

    def _init_reranker(self):
        """Initialize BGE reranker if available."""
        if is_reranker_available():
            try:
                self.reranker = get_reranker(
                    model_id=self.model_name,
                    device=self.device,
                    batch_size=32,  # Can be tuned
                )
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize BGE reranker: {e}")
                self.reranker = None
        else:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                "BGE reranker not available, coverage evaluation will be limited"
            )
            self.reranker = None

    # ---------- feature helpers ----------
    def _features(self, text: str) -> Dict[str, int]:
        return {k: 1 if p.search(text) else 0 for k, p in _ACTION_PATTERNS.items()}

    # ---------- main API ----------
    def score_matrix(self, subqs: List[str], passages: List[Passage]) -> np.ndarray:
        """Return a matrix [len(subqs) x len(passages)] of a'(q_i, p_j) in [0,1] using BGE reranker."""
        if not subqs or not passages:
            return np.zeros((len(subqs), len(passages)), dtype="float32")

        passage_texts = [p.text for p in passages]
        mat = np.zeros((len(subqs), len(passages)), dtype="float32")

        if self.reranker is not None:
            # Use BGE reranker for cross-encoder scores
            for i, subq in enumerate(subqs):
                try:
                    # Get BGE scores for this subquery against all passages
                    scores = self.reranker.score(subq, passage_texts)

                    # Convert BGE scores to [0,1] range and apply feature weighting
                    feats = [self._features(p.text) for p in passages]
                    for j, (score, feat_dict) in enumerate(zip(scores, feats)):
                        # BGE scores are typically in range [0, 1], but may exceed
                        # Normalize and apply sigmoid for calibration
                        ce_score = max(0.0, min(1.0, float(score)))  # Clamp to [0,1]

                        # Apply feature weighting
                        s = self.w["w0"] * ce_score
                        for name in ("steps", "endpoint", "jira", "owner", "table"):
                            s += self.w[name] * feat_dict[name]

                        mat[i, j] = _sigmoid(s)  # calibrated into 0..1

                except Exception as e:
                    import logging

                    logger = logging.getLogger(__name__)
                    logger.warning(f"BGE scoring failed for subquery {i}: {e}")
                    # Fallback to zero scores for this subquery
                    mat[i, :] = 0.0
        else:
            # Fallback to basic feature-only scoring if BGE not available
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("BGE reranker not available, using feature-only scoring")

            feats = [self._features(p.text) for p in passages]
            for i in range(len(subqs)):
                for j, feat_dict in enumerate(feats):
                    # No cross-encoder score, just features
                    s = sum(
                        self.w[name] * feat_dict[name]
                        for name in ("steps", "endpoint", "jira", "owner", "table")
                    )
                    mat[i, j] = _sigmoid(s * 0.5)  # Scale down since no CE score

        return mat

    def aspect_recall(self, mat: np.ndarray) -> float:
        """Fraction of subqs for which some passage crosses tau."""
        if mat.size == 0:
            return 0.0
        covered = (mat.max(axis=1) >= self.tau).sum()
        return float(covered) / mat.shape[0]

    def alpha_ndcg(
        self, mat: np.ndarray, ranks: List[int], alpha: float = None
    ) -> float:
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
            g_i = ((1 - alpha) ** seen) * mat[:, j]
            gains.append(g_i.sum() / math.log2(1 + r_idx))
            # update "seen" as if we pick the max aspect per step
            seen = seen + (mat[:, j] > 0).astype("int32")
        dcg = float(np.sum(gains))

        # Ideal DCG: pick best per step (upper bound by sorting per-aspect maxima)
        ideal_per_aspect = np.sort(mat.max(axis=1))[::-1]
        seen = np.zeros_like(ideal_per_aspect, dtype="int32")
        ideal_terms = []
        for r_idx in range(1, len(order) + 1):
            # pick next best remaining aspect (cyclic approx): this upper bounds the real ideal DCG
            i = (r_idx - 1) % len(ideal_per_aspect)
            g = ((1 - alpha) ** seen[i]) * ideal_per_aspect[i]
            ideal_terms.append(g / math.log2(1 + r_idx))
            seen[i] += 1
        idcg = float(np.sum(ideal_terms)) or 1.0
        return dcg / idcg

    def select_passages(
        self,
        mat: np.ndarray,
        subqs: List[str],
        passages: List[Passage],
        top_n_per_aspect: int = 1,
    ) -> Dict[int, List[int]]:
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

    def evaluate(
        self, user_query: str, subqs: List[str], passages: List[Passage]
    ) -> Dict:
        mat = self.score_matrix(subqs, passages)
        ranks = [int(p.meta.get("rank", idx + 1)) for idx, p in enumerate(passages)]
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
