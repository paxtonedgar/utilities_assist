# JPMC Migration: Inventory & Branch Diff Analysis

## Overview
This document provides a bird's-eye view of mocks, adapters, config flags, and areas where `feat/phase1-foundations` diverges from `main` branch, identifying what needs to be removed/replaced/kept for JPMC production deployment.

## Mock/Test/Adapter Code Inventory

### Core Mock Infrastructure

| Path (phase1) | Purpose | Runtime Usage? | JPMC Parity? | Recommendation |
|---------------|---------|----------------|--------------|----------------|
| `src/mocks/search_client.py` | BM25 mock search using rank-bm25 | Yes - via `USE_MOCK_SEARCH=true` | No analog in main | **REMOVE** - JPMC uses real OpenSearch |
| `src/mocks/vector_search.py` | FAISS vector search mock | Yes - via `USE_MOCK_SEARCH=true` | No analog in main | **REMOVE** - JPMC uses real vector search |
| `scripts/embed_mock_docs.py` | Generate embeddings for mock docs + FAISS index | No - dev utility | No analog in main | **REMOVE** - dev-only tool |
| `scripts/index_mock_docs.py` | Index mock docs into local OpenSearch | No - dev utility | No analog in main | **REMOVE** - dev-only tool |

### Mock Data & Test Documents  

| Path (phase1) | Purpose | Runtime Usage? | JPMC Parity? | Recommendation |
|---------------|---------|----------------|--------------|----------------|
| `data/mock_docs/doc-001.json` to `doc-050.json` | 50 synthetic utility docs | No - mock data only | No analog in main | **REMOVE** - synthetic test data |
| `data/mock_docs/index_template.json` | OpenSearch mapping template | No - dev utility | No analog in main | **REMOVE** - dev-only config |
| `eval/mock_corpus/utilities_docs.jsonl` | Evaluation corpus | No - evaluation only | No analog in main | **KEEP** in tests/ only |
| `eval/golden_set.yaml` | Ground truth for evaluation | No - evaluation only | No analog in main | **KEEP** in tests/ only |

### Demo & Evaluation Scripts

| Path (phase1) | Purpose | Runtime Usage? | JPMC Parity? | Recommendation |
|---------------|---------|----------------|--------------|----------------|
| `demo_ui.py` | Standalone demo UI | No - demo only | No analog in main | **REMOVE** - demo code |
| `demo_embedding_creation.py` | Demo embedding generation | No - demo only | No analog in main | **REMOVE** - demo code |
| `modern_demo.py` | Modern demo interface | No - demo only | No analog in main | **REMOVE** - demo code |
| `eval/run_eval.py` | Evaluation runner | No - evaluation only | No analog in main | **MOVE** to tests/ |
| `eval/hybrid_eval_client.py` | Hybrid retrieval evaluation | No - evaluation only | No analog in main | **MOVE** to tests/ |

### Local Development Infrastructure

| Path (phase1) | Purpose | Runtime Usage? | JPMC Parity? | Recommendation |
|---------------|---------|----------------|--------------|----------------|
| `config.local.ini` | Local dev config | No - dev only | No analog in main | **REMOVE** - dev config |
| `scripts/start_opensearch_local.sh` | Start local OpenSearch container | No - dev utility | No analog in main | **REMOVE** - dev script |
| `scripts/stop_opensearch_local.sh` | Stop local OpenSearch container | No - dev utility | No analog in main | **REMOVE** - dev script |
| `.env.local.example` | Local dev environment template | No - dev template | No analog in main | **KEEP** for local dev |
| `.env.jpmc.example` | JPMC environment template | No - config template | No analog in main | **KEEP** - essential for JPMC |

### Test Infrastructure

| Path (phase1) | Purpose | Runtime Usage? | JPMC Parity? | Recommendation |
|---------------|---------|----------------|--------------|----------------|
| `tests/probe_tests/test_mock_flow.py` | Test mock search flow | No - tests only | No analog in main | **KEEP** - guard with env flags |
| `tests/probe_tests/test_interfaces.py` | Test client interfaces | No - tests only | No analog in main | **KEEP** - production tests |
| `tests/e2e-openai.spec.js` | End-to-end Playwright tests | No - tests only | No analog in main | **KEEP** - production tests |

## Branch Differences Analysis (main → feat/phase1)

### Major Structural Changes
- **Complete src/ reorganization**: `src/data_pipeline/` → `src/infra/`, `src/services/`, `src/app/`
- **Configuration refactor**: Legacy `config.ini` → Modern `src/infra/config.py` with profile-based switching
- **Client architecture**: Singleton pattern → Request-scoped clients with LRU caching
- **Mock integration**: No mocks in main → Extensive mock system in phase1

### Configuration Changes
- **Profile-based config**: `CLOUD_PROFILE=jpmc_azure|local|tests` drives all client selection
- **Azure OpenAI integration**: New Azure endpoints, AAD auth, deployment names
- **OpenSearch enterprise**: AWS4Auth, JPMC proxy settings, enterprise endpoints
- **Environment variable standardization**: `USE_MOCK_SEARCH`, `USE_LOCAL_AZURE` flags

### Retrieval System Changes
- **Hybrid retrieval**: BM25 + vector search combination
- **Reranking pipeline**: Multi-stage ranking with tuned parameters  
- **ACL filtering**: Time-decay factors, user-based filtering
- **Blue/Green indexing**: Production reindexing support

### Infrastructure & Client Changes
- **JPMC proxy support**: `proxy.jpmchase.net:10443` configuration
- **AWS authentication**: `requests-aws4auth` for OpenSearch
- **Enterprise headers**: `user_sid` JPMC header injection
- **Retry/timeout tuning**: Production-grade error handling

### OpenSearch Query Changes
- **RRF (Rank Fusion)**: Combines BM25 and vector scores
- **Time decay**: Recent documents get score boosts
- **ACL queries**: Security filtering integration
- **Index aliasing**: `confluence_current` vs hardcoded indices

### Caching Architecture
- **LRU caching**: Client connection pooling only
- **No in-memory caches**: Removed for enterprise compliance
- **Session pooling**: HTTP keep-alive for OpenSearch connections

## Blocking Deltas for JPMC Deployment

### Critical Blockers (Must Change)
1. **Mock system removal**: All `USE_MOCK_SEARCH=true` code paths must be disabled/removed
2. **Local OpenSearch**: `localhost:9200` → Enterprise OpenSearch endpoints
3. **OpenAI API keys**: Public OpenAI → Azure OpenAI with AAD
4. **Proxy configuration**: Local direct access → JPMC proxy routing
5. **Authentication**: API keys → Azure AAD token flow

### Configuration Blockers  
1. **Environment switching**: Must set `CLOUD_PROFILE=jpmc_azure` everywhere
2. **Secret management**: Hardcoded tokens → Enterprise secret injection
3. **Endpoint URLs**: Dev endpoints → Production JPMC URLs
4. **Index names**: Mock indices → Real JPMC OpenSearch indices

### Runtime Blockers
1. **Mock data dependencies**: 50 mock docs → Real Confluence content
2. **FAISS index files**: Local files → Production embeddings
3. **Evaluation harness**: Test queries → Production query patterns
4. **Demo UIs**: Streamlit demos → Production web interface

## High-Risk Areas Checklist

### 🔐 Authentication & Authorization
- [ ] **Azure AAD token flow** - Verify token refresh and expiration handling
- [ ] **JPMC proxy authentication** - Test proxy credentials and routing  
- [ ] **OpenSearch AWS4Auth** - Validate AWS role/policy permissions
- [ ] **Enterprise headers** - Confirm `user_sid` header requirement
- [ ] **Secret rotation** - Handle client secret updates without restart

### 🔍 OpenSearch Integration
- [ ] **Enterprise endpoints** - Validate production OpenSearch cluster access
- [ ] **Index aliases** - Confirm `confluence_current` vs hardcoded index names
- [ ] **ACL query filters** - Test user-based document access controls
- [ ] **Connection pooling** - Monitor connection limits and cleanup
- [ ] **Query timeouts** - Tune for production data volume and network latency

### 📊 Caching & Performance  
- [ ] **LRU cache limits** - Prevent memory leaks in long-running processes
- [ ] **HTTP session reuse** - Verify keep-alive connections work through proxy
- [ ] **Token caching** - Balance security vs performance for AAD tokens
- [ ] **Retry exponential backoff** - Avoid thundering herd on service failures
- [ ] **Circuit breaker patterns** - Handle downstream service degradation

### 🔒 Secrets & Configuration
- [ ] **Environment variable injection** - Verify K8s/container secret mounting
- [ ] **Config validation** - Fail fast on missing/invalid JPMC credentials  
- [ ] **Logging sanitization** - Prevent secret leakage in logs/traces
- [ ] **Profile isolation** - Ensure test/dev profiles can't access prod resources
- [ ] **Runtime config changes** - Handle config updates without restart

### 🚨 Mock System Cleanup
- [ ] **Mock detection at startup** - Assert no mocks enabled in production
- [ ] **Environment flag validation** - Block `USE_MOCK_SEARCH=true` in JPMC profile
- [ ] **Mock file removal** - Ensure no mock data bundled in production images
- [ ] **Test isolation** - Mock tests only run in test environments
- [ ] **Fallback prevention** - No automatic fallback to mocks on auth failures

---

**Next Steps**: 
1. Review [01-mocks-audit.md](01-mocks-audit.md) for detailed mock wiring analysis
2. Review [02-config-switch.md](02-config-switch.md) for exact configuration changes needed
3. Execute production deployment with `CLOUD_PROFILE=jpmc_azure` and validate all high-risk areas