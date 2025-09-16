from __future__ import annotations

"""
Clustering and labeling for Stage 1A.

Uses embeddings (Azure/OpenAI) if available; falls back to TF‑IDF + KMeans.
Optionally integrates BERTopic if installed.
"""

from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


TAXONOMY_LABELS = [
    "process_runbook",
    "api_reference",
    "system_design",
    "troubleshooting",
    "meeting_notes",
    "org_chart",
    "faq",
    "misc",
    "abstain",
]

ABSTAIN_LABEL = "abstain"
ABSTAIN_CONFIDENCE_THRESHOLD = 0.65
MAX_LABEL_RETRIES = 2
RETRY_TEMPERATURE = 0.2
RETRY_CANDIDATE_LIMIT = 3


def _try_import_bertopic():
    try:
        from bertopic import BERTopic  # type: ignore
        return BERTopic
    except Exception:
        return None


def embed_tokens(resources, docs: List[Dict[str, Any]]) -> List[List[float]]:
    """Embed each fingerprint tokens string using the shared embed client.

    resources: infra.resource_manager.RAGResources
    """
    strings = [" ".join(d["tokens"]) for d in docs]
    if not strings:
        return []
    if not resources.embed_client:
        logger.warning("Embed client not configured; returning empty embeddings")
        return []
    try:
        client = resources.embed_client
        # OpenAI/Azure compatible embeddings API
        resp = client.embeddings.create(model=resources.settings.embed.model, input=strings)
        # OpenAI/Azure client returns objects with .embedding attribute
        vectors = [getattr(d, "embedding", None) for d in resp.data]  # type: ignore
        vectors = [v for v in vectors if v is not None]
        return vectors
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []


def cluster_with_bertopic(vectors: List[List[float]], docs: List[Dict[str, Any]]):
    BERTopic = _try_import_bertopic()
    if not BERTopic:
        return None
    try:
        model = BERTopic(verbose=False)
        topics, probs = model.fit_transform([" ".join(d["tokens"]) for d in docs], embeddings=vectors)
        return model, topics, probs
    except Exception as e:
        logger.error(f"BERTopic failed: {e}")
        return None


def cluster_with_tfidf(docs: List[Dict[str, Any]], k: int = 12):
    """Fallback: TF‑IDF + KMeans on token strings.
    Returns (labels, centers) minimal info.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.cluster import KMeans  # type: ignore
    except Exception:
        logger.error("scikit-learn not available")
        return None
    texts = [" ".join(d["tokens"]) for d in docs]
    vec = TfidfVectorizer(max_features=4096)
    X = vec.fit_transform(texts)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return {"labels": labels, "vocab": vec.get_feature_names_out().tolist()}


def build_topic_cards(docs: List[Dict[str, Any]], labels: List[int]) -> Dict[int, Dict[str, Any]]:
    """Aggregate simple topic/cluster summaries for LLM labeling.
    Returns: cluster_id -> {size, keywords, mean_patterns}
    """
    from collections import defaultdict
    clusters: Dict[int, Dict[str, Any]] = {}
    acc = defaultdict(lambda: {"n": 0, "kw": {}, "pat": {}})
    for d, lab in zip(docs, labels):
        a = acc[lab]
        a["n"] += 1
        for t in d.get("tokens", []):
            if t.startswith("keyword:"):
                k = t.split(":", 1)[1]
                a["kw"][k] = a["kw"].get(k, 0) + 1
        for k, v in (d.get("features", {}).get("patterns", {}) or {}).items():
            a["pat"][k] = a["pat"].get(k, 0) + v

    for lab, a in acc.items():
        n = max(a["n"], 1)
        top_kw = sorted(a["kw"].items(), key=lambda kv: kv[1], reverse=True)[:12]
        mean_pat = {k: round(v / n, 3) for k, v in a["pat"].items()}
        clusters[lab] = {
            "size": a["n"],
            "keywords": [k for k, _ in top_kw],
            "mean_patterns": mean_pat,
        }
    return clusters


def llm_label_topics(resources, topic_cards: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """Use the shared chat client to label topics with our taxonomy via few-shot.
    Returns: topic_id -> {label, confidence}
    """
    if not resources.chat_client:
        logger.warning("Chat client not configured; returning empty labels")
        return {}
    client = resources.chat_client

    def _call_labeler(
        cards_payload: List[Dict[str, Any]],
        attempt: int,
        temperature: float,
        hints: Dict[int, Dict[str, Any]] | None = None,
    ) -> Dict[int, Dict[str, Any]]:
        system = (
            "You are a taxonomy labeler. Map topic cards to exactly one label from the list "
            f"{', '.join(TAXONOMY_LABELS)}. Choose '{ABSTAIN_LABEL}' when no label applies or confidence is below 0.65. "
            "Return strictly JSON: {labels: [{topic_id, label, confidence}]}."
        )
        examples = [
            {
                "keywords": ["install", "configure", "run", "click"],
                "mean_patterns": {"ordered_lists": 0.7, "proc_cues": 0.6},
                "label": "process_runbook",
            },
            {
                "keywords": ["parameter", "type", "default", "required"],
                "mean_patterns": {"tables": 0.8, "headers": 0.5},
                "label": "api_reference",
            },
            {
                "keywords": ["class", "method", "code", "build"],
                "mean_patterns": {"code_blocks": 0.6, "inline_backticks": 0.4},
                "label": "system_design",
            },
        ]

        prompt: Dict[str, Any] = {
            "task": "label_topics",
            "taxonomy": TAXONOMY_LABELS,
            "examples": examples,
            "topics": cards_payload,
        }
        if hints:
            prompt["hints"] = {
                str(tid): {k: v for k, v in hint.items() if v}
                for tid, hint in hints.items()
            }

        logger.debug(
            "LLM topic labeling request",
            extra={"topic_count": len(cards_payload), "attempt": attempt},
        )
        resp = client.chat.completions.create(
            model=resources.settings.chat.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json_dumps(prompt)},
            ],
            temperature=temperature,
        )
        msg = resp.choices[0].message  # type: ignore
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        content = content or "{}"
        logger.debug(
            "LLM topic label raw response",
            extra={"preview": str(content)[:300], "attempt": attempt},
        )
        data = json_loads_safe(str(content))
        allowed_labels = {lbl.lower(): lbl for lbl in TAXONOMY_LABELS}
        result: Dict[int, Dict[str, Any]] = {}
        for item in (data.get("labels") or []):
            topic_id = int(item.get("topic_id") or 0)
            label_raw = str(item.get("label") or "").strip()
            conf = float(item.get("confidence") or 0.0)
            label_key = label_raw.lower()
            label = allowed_labels.get(label_key, ABSTAIN_LABEL)
            if conf < ABSTAIN_CONFIDENCE_THRESHOLD:
                label = ABSTAIN_LABEL
            conf = max(0.0, min(conf, 1.0))
            result[topic_id] = {
                "label": label,
                "confidence": conf if label != ABSTAIN_LABEL else 0.0,
                "raw_confidence": conf,
                "attempt": attempt,
            }
        return result

    def _build_retry_hints(
        assigned: Dict[int, Dict[str, Any]],
        cards_map: Dict[int, Dict[str, Any]],
        pending_ids: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        label_keyword_map: Dict[str, set[str]] = {}
        for tid, info in assigned.items():
            if info.get("label") in (None, ABSTAIN_LABEL):
                continue
            label = info["label"]
            label_keyword_map.setdefault(label, set()).update(
                cards_map.get(tid, {}).get("keywords", [])
            )

        ranked_labels = [
            label
            for label, _ in sorted(
                label_keyword_map.items(), key=lambda kv: len(kv[1]), reverse=True
            )
        ]
        if not ranked_labels:
            ranked_labels = [lbl for lbl in TAXONOMY_LABELS if lbl != ABSTAIN_LABEL]

        hints: Dict[int, Dict[str, Any]] = {}
        for tid in pending_ids:
            keywords = set(cards_map.get(tid, {}).get("keywords", []))
            candidates = [
                label
                for label, kw in label_keyword_map.items()
                if keywords.intersection(kw)
            ]
            if not candidates:
                candidates = ranked_labels[:RETRY_CANDIDATE_LIMIT]
            hints[tid] = {
                "candidate_labels": candidates[:RETRY_CANDIDATE_LIMIT],
                "keywords": list(keywords)[:8],
            }
        return hints

    try:
        cards_payload = [{"topic_id": tid, **card} for tid, card in topic_cards.items()]
        initial = _call_labeler(cards_payload, attempt=1, temperature=0.0)
        results: Dict[int, Dict[str, Any]] = {}
        results.update(initial)
        for tid, info in results.items():
            info.setdefault("resolved_by", "attempt_1")
            info.setdefault("attempt_history", []).append(
                {
                    "attempt": 1,
                    "label": info.get("label"),
                    "confidence": info.get("confidence"),
                }
            )

        for tid in topic_cards:
            results.setdefault(
                tid,
                {
                    "label": ABSTAIN_LABEL,
                    "confidence": 0.0,
                    "raw_confidence": 0.0,
                    "attempt": 1,
                    "resolved_by": "attempt_1",
                    "attempt_history": [
                        {
                            "attempt": 1,
                            "label": ABSTAIN_LABEL,
                            "confidence": 0.0,
                        }
                    ],
                },
            )

        recovered = 0
        pending = [tid for tid, info in results.items() if info.get("label") == ABSTAIN_LABEL]
        attempt = 2
        while pending and attempt <= MAX_LABEL_RETRIES + 1:
            hints = _build_retry_hints(results, topic_cards, pending)
            subset_cards = [{"topic_id": tid, **topic_cards[tid]} for tid in pending]
            retry_results = _call_labeler(
                subset_cards,
                attempt=attempt,
                temperature=RETRY_TEMPERATURE,
                hints=hints,
            )
            for tid in pending:
                if hints.get(tid):
                    results.setdefault(tid, {}).setdefault(
                        "candidate_labels", hints[tid].get("candidate_labels", [])
                    )
            for tid, info in retry_results.items():
                current = results.get(tid, {})
                current.setdefault("attempt_history", []).append(
                    {
                        "attempt": attempt,
                        "label": info.get("label"),
                        "confidence": info.get("confidence"),
                    }
                )
                if info.get("label") != ABSTAIN_LABEL:
                    if current.get("label") == ABSTAIN_LABEL:
                        recovered += 1
                    merged = {**current, **info, "resolved_by": f"attempt_{attempt}"}
                    # Preserve candidate hints if present
                    if current.get("candidate_labels"):
                        merged["candidate_labels"] = current["candidate_labels"]
                    results[tid] = merged
                else:
                    current["resolved_by"] = f"attempt_{attempt}"
                    current["label"] = ABSTAIN_LABEL
                    current["confidence"] = 0.0
                    current["raw_confidence"] = 0.0
                    results[tid] = current
            pending = [tid for tid in pending if results.get(tid, {}).get("label") == ABSTAIN_LABEL]
            attempt += 1

        abstained = sum(1 for v in results.values() if v.get("label") == ABSTAIN_LABEL)
        logger.info(
            "LLM topic labeling completed",
            extra={
                "topic_count": len(topic_cards),
                "labels_returned": len(results),
                "abstained": abstained,
                "recovered_from_retries": recovered,
                "attempts": attempt - 1,
            },
        )
        return results
    except Exception as e:
        logger.exception(
            "LLM topic labeling failed",
            extra={"topic_count": len(topic_cards), "error": str(e)},
        )
        return {}


def json_dumps(x: Any) -> str:
    import json
    return json.dumps(x, ensure_ascii=False)


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```") and s.endswith("```"):
        # remove first and last fence line
        lines = s.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1])
    return s


def json_loads_safe(s: str) -> Dict[str, Any]:
    import json
    import re
    try:
        s = _strip_code_fences(s or "{}")
        # If the LLM prepended a language tag like "json\n", drop it
        if s.lower().startswith("json\n"):
            s = s.split("\n", 1)[1]
        try:
            return json.loads(s)
        except Exception:
            # Try to extract the first JSON object substring
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                chunk = s[start : end + 1]
                try:
                    return json.loads(chunk)
                except Exception:
                    pass
        # Fallback: accept Python-like dicts with single quotes via ast.literal_eval
        try:
            import ast
            obj = ast.literal_eval(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        return {}
    except Exception:
        return {}
