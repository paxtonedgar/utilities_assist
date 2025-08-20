# src/list_models.py
import sys, cdao

# terms to search for (defaults target rerankers / MiniLM / BGE)
TERMS = [t.lower() for t in (sys.argv[1:] or ["mini", "lm", "bge", "reranker"])]

def call(fn_name, *args, **kwargs):
    fn = getattr(cdao, fn_name, None)
    if not fn:
        return []
    try:
        return fn(*args, **kwargs)
    except TypeError:
        try:
            return fn()
        except Exception:
            return []
    except Exception:
        return []

def yield_rows():
    # Try a few possible list/search entry points; SDKs vary by version
    for fn in ("public_models_list_all",
               "public_models_list_models",
               "public_models_search",
               "models_list",
               "list_models"):
        for r in call(fn, q=""):
            yield r

def name_of(r):
    if isinstance(r, str):
        return r
    if isinstance(r, dict):
        for k in ("model_unique_name","unique_name","name","id","model_name"):
            if r.get(k):
                return r[k]
    for k in ("model_unique_name","unique_name","name","id","model_name"):
        if hasattr(r, k):
            v = getattr(r, k)
            if v:
                return v
    return None

def matches(name):
    s = name.lower()
    return all(t in s for t in TERMS)

# collect candidates
all_names = set()
for r in yield_rows():
    n = name_of(r)
    if n:
        all_names.add(n)

cands = sorted(n for n in all_names if matches(n))

print("\nCandidates (may include 3- or 4-segment names):")
for n in cands:
    print(" -", n)

# If we saw 3-segment prefixes, show all 4-segment variants that exist
prefixes = [n for n in cands if len(n.split("__")) == 3]
if prefixes:
    print("\nVariants for 3-segment prefixes (choose one of these):")
    for p in prefixes:
        vs = sorted(x for x in all_names if x.startswith(p + "__"))
        for v in vs:
            print(" -", v)
