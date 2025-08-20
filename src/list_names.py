# list_models.py
import cdao

# dump every public model that looks like reranker / ms-marco / bge
def rows_like(substrs):
    # Try a couple of list/search entry points (SDKs differ slightly by version)
    for fn_name in [
        "public_models_list_all",            # preferred
        "public_models_list_models",         # alt
        "public_models_search",              # alt (may take q=)
    ]:
        fn = getattr(cdao, fn_name, None)
        if fn is None:
            continue
        try:
            rows = fn() if fn_name != "public_models_search" else fn(q="")
        except TypeError:
            # some versions take no args for search
            rows = fn()
        for r in rows:
            text = " ".join(str(r.get(k, "")) for k in ["model_unique_name","name","publisher","repo"])
            if all(s.lower() in text.lower() for s in substrs):
                yield r

hits = list(rows_like(["mini", "lm"])) + list(rows_like(["bge", "reranker"]))
seen = set()
print("\nPossible matches:\n")
for r in hits:
    mu = r.get("model_unique_name") or r.get("unique_name") or r.get("name")
    if mu and mu not in seen:
        seen.add(mu)
        print(" -", mu)
