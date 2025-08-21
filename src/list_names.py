import os
import sys
from typing import List, Tuple

# Choose one here (or pass via CLI)
MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
QUERY = "How do I onboard to ETU in our environment?"
DOCS = [
    "ETU: Enhanced Transaction Utility. To onboard, open a Jira in project ETUONB and attach the service form.",
    "Energy Transfer Unit is a physics concept unrelated to onboarding.",
    "Contact the ETU platform team on Slack #etu-onboarding. Prereqs: AppID, VPC egress, ServiceNOW change.",
    "This is a glossary page with a definition only.",
]


def use_flagembedding(model_id: str) -> bool:
    return (
        model_id.lower().startswith("baai/bge-reranker")
        or "bge-reranker" in model_id.lower()
    )


def rerank_flagembedding(
    model_id: str, query: str, docs: List[str]
) -> List[Tuple[str, float]]:
    from FlagEmbedding import FlagReranker

    rr = FlagReranker(
        model_id,
        use_fp16=True,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu",
    )
    scores = rr.compute_score([(query, d) for d in docs], normalize=True)
    return sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)


def rerank_sentence_transformers(
    model_id: str, query: str, docs: List[str]
) -> List[Tuple[str, float]]:
    from sentence_transformers import CrossEncoder

    rr = CrossEncoder(
        model_id, device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") else "cpu"
    )
    scores = rr.predict([[query, d] for d in docs])
    return sorted(
        zip(docs, [float(s) for s in scores]), key=lambda x: x[1], reverse=True
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MODEL = sys.argv[1]
    if use_flagembedding(MODEL):
        ranked = rerank_flagembedding(MODEL, QUERY, DOCS)
    else:
        ranked = rerank_sentence_transformers(MODEL, QUERY, DOCS)

    print(f"\nModel: {MODEL}")
    for i, (d, s) in enumerate(ranked, 1):
        print(f"{i:>2}. {s:6.3f}  {d[:120]}")
