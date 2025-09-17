from __future__ import annotations

"""Candidate extraction and prototype embedding for taxonomy mining."""

import hashlib
import json
import logging
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from pydantic import BaseModel

from src.infra.resource_manager import initialize_resources
from src.infra.search_config import OpenSearchConfig
from src.infra.settings import get_settings

from .config import TaxonomyConfig
from .schemas import LabelPrototype, TermCandidate

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-']+")


class _SemanticArtifacts(BaseModel):
    topics: Dict[str, Dict[str, object]]
    doc_map: List[Dict[str, object]]


def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not path.exists():
        return rows
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return rows


def _load_semantic_artifacts(semantic_dir: Path) -> _SemanticArtifacts:
    topics = _read_json(semantic_dir / "topics.json")
    doc_map = _read_jsonl(semantic_dir / "doc_map.jsonl")
    return _SemanticArtifacts(topics=topics or {}, doc_map=doc_map)


def _collect_segment_text(segment: Dict[str, object]) -> str:
    text = segment.get("text") or segment.get("content")
    if isinstance(text, str) and text.strip():
        return text
    features = segment.get("features")
    if not isinstance(features, dict):
        return ""
    seg_type = segment.get("type")
    if seg_type == "Table":
        headers = " ".join(map(str, (features.get("headers") or [])[:20]))
        rows = [
            " ".join(map(str, row))
            for row in (features.get("rows") or [])[:5]
            if isinstance(row, (list, tuple))
        ]
        return " ".join([headers] + rows)
    if seg_type in {"StepBlock", "ListBlock"}:
        items = features.get("items")
        if isinstance(items, list):
            return " ".join(map(str, items[:12]))
    sample = features.get("sample")
    if isinstance(sample, str):
        return sample
    return ""


def _gather_label_text(
    docs: List[Dict[str, object]], config: TaxonomyConfig
) -> Tuple[str, List[str]]:
    collected: List[str] = []
    doc_ids: List[str] = []
    char_budget = config.max_chars_per_label
    for doc in docs:
        doc_id = str(doc.get("doc_id") or "")
        if doc_id:
            doc_ids.append(doc_id)
        segments = (doc.get("segments") or [])[: config.max_segments_per_doc]
        for seg in segments:
            snippet = _collect_segment_text(seg)
            if snippet:
                collected.append(snippet.strip())
                char_budget -= len(snippet)
                if char_budget <= 0:
                    break
        if char_budget <= 0:
            break
    merged = " ".join(collected)
    return merged[: config.max_chars_per_label], doc_ids


def _token_spans(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def _term_hash(surface: str, class_id: str) -> str:
    digest = hashlib.sha1(f"{class_id}:{surface.lower()}".encode("utf-8")).hexdigest()
    return digest[:16]


def _embed_texts(resources, texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    client = resources.embed_client
    if not client:
        raise RuntimeError("Embed client not configured")
    model = resources.settings.embed.model
    dims = OpenSearchConfig.EMBEDDING_DIMENSIONS
    vectors: List[List[float]] = []
    for idx in range(0, len(texts), 32):
        chunk = texts[idx : idx + 32]
        resp = client.embeddings.create(
            model=model,
            input=chunk,
            dimensions=dims,
        )
        for item in resp.data:
            vec = getattr(item, "embedding", None)
            if vec:
                vectors.append(vec)
    return vectors


def _select_contexts(contexts: List[str], limit: int) -> List[str]:
    unique = []
    seen = set()
    for ctx in contexts:
        key = ctx.strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(key)
        if len(unique) >= limit:
            break
    return unique


def generate_candidates(
    semantic_dir: Path,
    config: TaxonomyConfig,
    resources=None,
) -> Tuple[List[LabelPrototype], List[TermCandidate]]:
    """Generate label prototypes and term candidates with embedded contexts."""

    artifacts = _load_semantic_artifacts(semantic_dir)
    if not artifacts.doc_map:
        logger.warning("No documents found in semantic outputs: %s", semantic_dir)
        return [], []

    if resources is None:
        settings = get_settings()
        resources = initialize_resources(settings)

    random.seed(config.random_seed)

    label_docs: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for doc in sorted(artifacts.doc_map, key=lambda d: (str(d.get("doc_type")), str(d.get("doc_id")))):
        label = str(doc.get("doc_type") or "").strip()
        if not label or label == "abstain":
            continue
        label_docs[label].append(doc)

    if not label_docs:
        logger.warning("No labeled documents available for taxonomy extraction")
        return [], []

    proto_records: List[Tuple[str, str, List[str]]] = []
    for label in sorted(label_docs.keys()):
        merged, doc_ids = _gather_label_text(label_docs[label], config)
        proto_records.append((label, merged, doc_ids))

    prototype_vectors = _embed_texts(resources, [text for _, text, _ in proto_records])

    prototypes: List[LabelPrototype] = []
    for (label, text, doc_ids), vector in zip(proto_records, prototype_vectors):
        prototypes.append(
            LabelPrototype(
                class_id=label,
                embedding=vector,
                document_ids=doc_ids,
                text_length=len(text),
            )
        )

    # Candidate extraction
    candidates_by_label: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for label, docs in label_docs.items():
        for doc in docs:
            doc_id = str(doc.get("doc_id") or "")
            for segment in (doc.get("segments") or []):
                snippet = _collect_segment_text(segment)
                if not snippet:
                    continue
                spans = _token_spans(snippet)
                if not spans:
                    continue
                for n in range(1, config.max_ngram + 1):
                    if len(spans) < n:
                        break
                    for idx in range(0, len(spans) - n + 1):
                        span = spans[idx : idx + n]
                        words = [w for w, _, _ in span]
                        surface = " ".join(words)
                        normalized = " ".join(w.lower() for w in words)
                        if len(normalized) < 3:
                            continue
                        if normalized in candidates_by_label[label]:
                            candidate = candidates_by_label[label][normalized]
                        else:
                            candidate = {
                                "surface": surface,
                                "contexts": [],
                                "frequency": 0,
                                "doc_ids": set(),
                                "length": len(normalized.split()),
                            }
                            candidates_by_label[label][normalized] = candidate
                        candidate["frequency"] += 1
                        candidate["doc_ids"].add(doc_id)
                        start, end = span[0][1], span[-1][2]
                        ctx_start = max(0, start - 240)
                        ctx_end = min(len(snippet), end + 240)
                        candidate["contexts"].append(snippet[ctx_start:ctx_end])

    term_candidates: List[TermCandidate] = []
    for label in sorted(candidates_by_label.keys()):
        for key, data in sorted(candidates_by_label[label].items()):
            if data["frequency"] < config.min_frequency:
                continue
            contexts = _select_contexts(data["contexts"], config.contexts_per_term)
            if not contexts:
                continue
            term_candidates.append(
                TermCandidate(
                    term_id=_term_hash(data["surface"], label),
                    surface=data["surface"],
                    class_id=label,
                    frequency=int(data["frequency"]),
                    contexts=contexts,
                    embedding=[],  # placeholder, filled after embedding
                    doc_ids=sorted(d for d in data["doc_ids"] if d),
                    stats={"ngram": float(data["length"])},
                )
            )

    if not term_candidates:
        return prototypes, []

    # Embed term contexts
    all_contexts: List[str] = [cand.contexts[0] for cand in term_candidates]
    embeddings = _embed_texts(resources, all_contexts)
    for candidate, vec in zip(term_candidates, embeddings):
        candidate.embedding = vec

    logger.info(
        "Extracted %d prototypes and %d term candidates", len(prototypes), len(term_candidates)
    )

    return prototypes, term_candidates
