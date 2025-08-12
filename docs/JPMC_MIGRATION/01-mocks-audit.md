# JPMC Migration: Mocks Deep Audit

## Overview
This document provides a precise audit of all mock symbols, their wiring, and how they're controlled by environment flags in the `feat/phase1-foundations` branch.

## Mock Symbol Classification

### Core Mock Classes

#### MockSearchClient (`src/mocks/search_client.py`)
**Purpose**: BM25 text search using `rank-bm25` library  
**Trigger**: Imported directly in test files  
**Data Source**: `data/mock_docs/doc-001.json` to `doc-050.json` (50 synthetic documents)

```python
# Entry point - module-level search function
from mocks.search_client import search

def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    global _search_client
    if '_search_client' not in globals():
        _search_client = MockSearchClient()
    return _search_client.search(query, top_k)
```

**Usage Sites**:
- `tests/probe_tests/test_interfaces.py:30` - Direct import for testing
- `scripts/verify_setup.py:60` - Direct import for local development verification
- **No runtime usage in production code paths**

**Recommendation**: **DELETE** - Only used in dev/test, no ENV flag gating

#### MockVectorSearch (`src/mocks/vector_search.py`)  
**Purpose**: FAISS vector similarity search with Azure OpenAI embeddings  
**Trigger**: Imported directly in test files  
**Dependencies**: FAISS index file `data/mock_faiss.index` + metadata

```python
# Entry point - get client singleton
from mocks.vector_search import get_vector_search_client

def get_vector_search_client() -> MockVectorSearch:
    global _vector_search_client
    if _vector_search_client is None:
        _vector_search_client = MockVectorSearch()
    return _vector_search_client
```

**Usage Sites**:
- `scripts/test_vector_search.py:18` - Direct import for dev testing
- **No runtime usage in production code paths**

**Recommendation**: **DELETE** - Only used in dev/test, requires FAISS data files

### Mock Data Corpus

#### Mock Documents (`data/mock_docs/`)
**Purpose**: 50 synthetic utility API documents for local testing  
**Structure**: Each `doc-NNN.json` contains:
- `_id`, `api_name`, `utility_name`
- `page_url`, `sections` array with `heading`/`content`

**Usage Sites**:
- `src/mocks/search_client.py:33` - Loads doc-*.json files for BM25 search
- `scripts/embed_mock_docs.py:79` - Generates embeddings from documents  
- `scripts/index_mock_docs.py` - Indexes documents into local OpenSearch

**Recommendation**: **DELETE** - Synthetic test data, not needed in JPMC

#### Mock Evaluation Corpus (`eval/mock_corpus/utilities_docs.jsonl`)
**Purpose**: JSONL format corpus for evaluation runs  
**Content**: Same mock documents in different format

**Usage Sites**:
- `eval/run_eval.py` - Loads corpus for evaluation benchmarks
- `Makefile:75` - Referenced in evaluation targets

**Recommendation**: **MOVE** to `tests/` - Keep for testing only

### Environment Flag Controls

#### USE_MOCK_SEARCH Flag
**Purpose**: Controls whether to use mock search clients vs real OpenSearch  
**Default**: Not set (false)  
**Enabled in**: Local development only

```bash
# Makefile targets that enable mock search
run-local:    USE_MOCK_SEARCH=true USE_LOCAL_AZURE=true streamlit run src/main.py  
test-local:   USE_MOCK_SEARCH=true USE_LOCAL_AZURE=true python -m pytest
run-mock:     USE_MOCK_SEARCH=true streamlit run src/main.py
```

**Code Detection Sites**:
```python
# tests/probe_tests/test_interfaces.py:58
print(f"USE_MOCK_SEARCH: {search_and_rerank.USE_MOCK_SEARCH}")
if search_and_rerank.USE_MOCK_SEARCH:
    assert hasattr(search_and_rerank, '_mock_search_client')
```

**Critical Issue**: The `search_and_rerank` module referenced in tests **does not exist** in current codebase. This indicates **stale test code** that should be removed.

#### USE_LOCAL_AZURE Flag  
**Purpose**: Controls Azure vs OpenAI client configuration  
**Enabled in**: Local development with Azure configs

**Usage Sites**:
```python
# src/mocks/vector_search.py:95
os.environ["USE_LOCAL_AZURE"] = "true"

# scripts/embed_mock_docs.py:51  
os.environ["USE_LOCAL_AZURE"] = "true"

# tests/probe_tests/test_interfaces.py:83
if not os.getenv("USE_LOCAL_AZURE") and not os.getenv("USE_MOCK_SEARCH"):
    pytest.skip("Requires local Azure config or mock mode")
```

**Recommendation**: **REPLACE** with `CLOUD_PROFILE=local` check

#### use_mock_corpus Controller Flag  
**Purpose**: Runtime flag to switch between production and mock indices  
**Location**: `src/controllers/turn_controller.py:32`

```python
async def handle_turn(
    user_input: str,
    settings: Settings,
    use_mock_corpus: bool = False  # <-- This flag
) -> AsyncGenerator[Dict[str, Any], None]:
```

**Runtime Logic**:
```python
# src/controllers/turn_controller.py:270-275
if use_mock_corpus:
    index_name = "confluence_mock"  # Use mock corpus for evaluation
elif intent.intent == "swagger":
    index_name = "khub-opensearch-swagger-index"
else:
    index_name = "confluence_current"  # Use alias for blue/green deployment
```

**UI Integration**:
```python
# src/app/chat_interface.py:111-115
if "use_mock_corpus" not in st.session_state:
    st.session_state.use_mock_corpus = True

# Toggle in UI
corpus_label = "Mock Data" if st.session_state.use_mock_corpus else "Production"
st.session_state.use_mock_corpus = not st.session_state.use_mock_corpus
```

**Recommendation**: **GUARD** with ENV check - disable in JPMC profile

### Mock Integration Points

#### No Direct Mock Wiring in Services
**Analysis**: Unlike expected, the main services (`retrieve.py`, `respond.py`, `rerank.py`) **do not** contain direct mock integration. Mocks are only used in:
1. Test files that import mock modules directly
2. Dev utility scripts  
3. Streamlit UI toggle for corpus selection

**This is good news**: The core business logic is clean and doesn't need mock removal surgery.

#### Stale Test References
**Problem**: `tests/probe_tests/test_interfaces.py` references `search_and_rerank` module that doesn't exist:

```python
# STALE CODE - module doesn't exist
import search_and_rerank
print(f"USE_MOCK_SEARCH: {search_and_rerank.USE_MOCK_SEARCH}")
if search_and_rerank.USE_MOCK_SEARCH:
    assert hasattr(search_and_rerank, '_mock_search_client')
```

**Recommendation**: **DELETE** stale test code

### JPMC Production Compatibility

#### Blocked Mock Usage  
These patterns **must be blocked** in JPMC profile:

```python
# src/app/chat_interface.py - Block mock corpus toggle
if settings.profile == "jpmc_azure":
    # Force production corpus
    st.session_state.use_mock_corpus = False
    # Hide toggle from UI
```

#### Environment Flag Validation
Add startup validation in `src/infra/config.py`:

```python
def validate_jpmc_profile():
    """Ensure no mock flags enabled in JPMC profile."""
    if os.getenv("CLOUD_PROFILE") == "jpmc_azure":
        blocked_flags = [
            ("USE_MOCK_SEARCH", "Mock search not allowed in JPMC"),
            ("USE_LOCAL_AZURE", "Local Azure config not allowed in JPMC")
        ]
        for flag, message in blocked_flags:
            if os.getenv(flag, "").lower() == "true":
                raise ValueError(f"{message}. Set CLOUD_PROFILE=local for development")
```

## Mock Removal Plan

### Phase 1: Safe Deletions (No Code Changes)
**Target**: Files that have zero runtime dependencies
- `data/mock_docs/` - All 50+ mock document files  
- `scripts/embed_mock_docs.py` - Dev utility only
- `scripts/index_mock_docs.py` - Dev utility only  
- `src/mocks/search_client.py` - Only imported by dev/test code
- `src/mocks/vector_search.py` - Only imported by dev/test code

**Risk**: **LOW** - These files are not imported by production code paths

### Phase 2: Test Code Cleanup (Test Changes Only)  
**Target**: Clean up stale and mock-dependent test code
- Delete `tests/probe_tests/test_interfaces.py:50-62` - References non-existent `search_and_rerank`
- Replace `USE_MOCK_SEARCH` / `USE_LOCAL_AZURE` checks with `CLOUD_PROFILE` checks
- Move evaluation code from `eval/` to `tests/eval/`

**Risk**: **LOW** - Test-only changes

### Phase 3: Runtime Mock Guards (Code Changes)
**Target**: Block mock usage in JPMC profile
- Add `validate_jpmc_profile()` call to config loader
- Guard `use_mock_corpus` toggle in Streamlit UI  
- Remove mock corpus index references from turn controller

**Risk**: **MEDIUM** - Changes production code paths, needs testing

### Phase 4: Environment Flag Deprecation  
**Target**: Remove deprecated environment flags
- Remove `USE_MOCK_SEARCH` from all documentation
- Remove `USE_LOCAL_AZURE` in favor of `CLOUD_PROFILE=local`
- Update Makefile targets to use `CLOUD_PROFILE` only

**Risk**: **LOW** - Documentation and build system changes

## High-Risk Mock Areas

### 🔍 Index Name Resolution  
**Risk**: Using wrong index in production
- **Current**: `use_mock_corpus=True` → `confluence_mock` index  
- **JPMC**: Must always use `confluence_current` (blue/green alias)
- **Mitigation**: Assert `confluence_mock` index doesn't exist in production

### 🎯 UI Toggle Exposure
**Risk**: Users accidentally enabling mock mode  
- **Current**: Streamlit UI shows "Mock Data" toggle
- **JPMC**: Toggle must be hidden/disabled  
- **Mitigation**: Profile-based UI conditional rendering

### 🔄 Fallback Behavior
**Risk**: Automatic fallback to mocks on auth failures
- **Current**: No automatic fallbacks detected ✅
- **JPMC**: Ensure auth failures fail fast, don't fall back to mocks  
- **Mitigation**: Explicit error handling without fallback logic

### 📊 Evaluation Code in Production
**Risk**: Evaluation harness accidentally running in production
- **Current**: Evaluation code in `eval/` directory
- **JPMC**: Move to `tests/` or exclude from production builds
- **Mitigation**: Build system exclusion patterns

---

**Next Steps**:
1. Execute Phase 1 deletions first - lowest risk, biggest cleanup
2. Validate no import errors after mock file removal  
3. Add environment flag validation before JPMC deployment
4. Test `CLOUD_PROFILE=jpmc_azure` end-to-end with mock guards active