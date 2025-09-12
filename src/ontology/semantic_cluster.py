from __future__ import annotations

"""
Clustering and labeling for Stage 1A.

Uses embeddings (Azure/OpenAI) if available; falls back to TF‑IDF + KMeans.
Optionally integrates BERTopic if installed.
"""

from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


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

    system = (
        "You are a taxonomy labeler. Map topic cards to one label: "
        "process_runbook, api_reference, system_design, troubleshooting, meeting_notes, org_chart, faq, misc. "
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

    cards = []
    for tid, card in topic_cards.items():
        cards.append({"topic_id": tid, **card})

    prompt = {
        "task": "label_topics",
        "taxonomy": [
            "process_runbook",
            "api_reference",
            "system_design",
            "troubleshooting",
            "meeting_notes",
            "org_chart",
            "faq",
            "misc",
        ],
        "examples": examples,
        "topics": cards,
    }

    try:
        resp = client.chat.completions.create(
            model=resources.settings.chat.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": json_dumps(prompt)}],
            temperature=0.0,
        )
        # Support SDK object or dict response shapes
        msg = resp.choices[0].message  # type: ignore
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        content = content or "{}"
        logger.info(f"LLM topic label raw response (first 300 chars): {str(content)[:300]}")
        data = json_loads_safe(str(content))
        result = {}
        for item in (data.get("labels") or []):
            result[int(item.get("topic_id") or 0)] = {
                "label": item.get("label", "misc"),
                "confidence": float(item.get("confidence") or 0.5),
            }
        return result
    except Exception as e:
        logger.error(f"LLM labeling failed: {e}")
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
