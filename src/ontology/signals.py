"""
Signal computation for edge scoring.

Provides lightweight, dependency-minimal signals that feed into scoring.
"""

from __future__ import annotations
from typing import Dict, Any
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


_CUE_WORDS = re.compile(r"\b(before|after|prereq|prerequisite|requires|depend(s|encies)?|must|first|then|next)\b", re.IGNORECASE)
_STOP = set(
    "the,a,an,and,or,of,to,in,on,for,with,by,from,as,be,is,are,was,were,this,that,these,those,please,note".split(",")
)


def _tokenize(text: str) -> set[str]:
    toks = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {t for t in toks if t not in _STOP}


def compute_signals_for_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    """Compute signals used by the scoring model for a candidate edge.

    a, b are step dicts {label, order, source, evidence{...}, ...}
    """
    la = a.get("label", "")
    lb = b.get("label", "")
    # regex_step if both steps came from regex extractor
    regex_step = float(a.get("source") == "regex" and b.get("source") == "regex")

    # bullet order consistency (1.0 if consecutive order integers)
    ord_a, ord_b = a.get("order"), b.get("order")
    bullet_order_consistency = 1.0 if (isinstance(ord_a, int) and isinstance(ord_b, int) and ord_b == ord_a + 1) else 0.0

    # cue words presence across evidence snippets or labels
    text_all = " ".join(
        [
            la,
            lb,
            (a.get("evidence", {}) or {}).get("text_snippet", ""),
            (b.get("evidence", {}) or {}).get("text_snippet", ""),
        ]
    )
    cue_word = 1.0 if _CUE_WORDS.search(text_all) else 0.0

    # same entity overlap via token Jaccard (proxy until entity linking lands)
    ta, tb = _tokenize(la), _tokenize(lb)
    inter = len(ta & tb)
    union = max(1, len(ta | tb))
    same_entity_overlap = inter / union

    # TF-IDF cosine similarity between labels as embed_sim proxy
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    X = vec.fit_transform([la, lb])
    sim = float(cosine_similarity(X[0:1], X[1:2])[0, 0])

    return {
        "regex_step": regex_step,
        "bullet_order_consistency": bullet_order_consistency,
        "cue_word": cue_word,
        "same_entity_overlap": same_entity_overlap,
        "embed_sim": sim,
        # Placeholders for future signals
        "spacy_imperative": 0.0,
        "nli_entailment": 0.0,
        "section_header_match": 0.0,
        "same_space": 0.0,
    }

