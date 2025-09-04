"""
Entity normalization and linking scaffolding.

Combines dictionary/synonyms, embedding similarity, and string similarity to
canonize surface forms and link mentions across documents.
"""

from typing import List, Dict, Any, Tuple


def normalize_surface_form(text: str, synonyms: Dict[str, List[str]] | None = None) -> str:
    """Map a surface form to a canonical entity name using a synonyms table.

    - Case-folding and simple token normalization as baseline
    - Apply domain synonyms where available
    """
    # TODO: implement lowercasing, punctuation stripping, and synonyms lookup
    return text


def cluster_entities(
    mentions: List[str],
    embed_fn,  # callable(list[str]) -> list[vector]
    sim_threshold: float = 0.82,
) -> Tuple[List[str], Dict[str, str]]:
    """Cluster mentions by embedding similarity to derive canonical IDs.

    Returns:
        canonicals: list of canonical IDs/names
        mapping: mention -> canonical
    """
    # TODO: implement embeddings, pairwise similarity, and greedy clustering
    return [], {}

