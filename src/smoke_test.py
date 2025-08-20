import os, json
import cdao

# ---- pick one from your catalog UI ----
MODEL_ID = "HF__BAAI__bge-reranker-v2-m3"
# MODEL_ID = "HF__cross-encoder__ms-marco-MiniLM-L-12-v2"

print(f"Downloading (or locating cached) model: {MODEL_ID}")
local_path = cdao.public_models_download_all_id_or_name(MODEL_ID)
print("Local model path:", local_path)

# ---- tiny inference: cross-encoder scoring for (query, passage) pairs ----
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained(local_path, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(local_path)
model.eval()

def score_pair(query: str, passage: str) -> float:
    # Works for pair-classification cross-encoders
    inputs = tok(query, passage, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
        # handle either single-logit (sigmoid) or 2-class (softmax) heads
        if logits.shape[-1] == 1:
            score = torch.sigmoid(logits[0])[0].item()
        else:
            score = torch.softmax(logits, dim=-1)[0, 1].item()
    return float(score)

query = "what is ETU"
candidates = [
    "ETU is the Enterprise Technology Utility used for onboarding internal apps.",
    "This paragraph describes the cafeteria menu and parking hours.",
]

scores = [score_pair(query, p) for p in candidates]
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

print("\nScores (higher = more relevant):")
for p, s in ranked:
    print(f"{s:0.4f}  |  {p}")
