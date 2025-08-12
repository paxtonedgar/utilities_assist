# JPMC Migration: Configuration Switch Analysis

## Overview
This document details the exact configuration knobs to flip for JPMC deployment, confirming that **no code edits are required** - only environment variable changes.

## Configuration Architecture Evolution

### Main Branch (Legacy)
**File**: `src/data_pipeline/config.ini` - Hardcoded JPMC production values
- ❌ **Monolithic INI file** with placeholder tokens `<AZURE_TENANT_ID>`, `<API_KEY>`
- ❌ **No environment-based switching** - single configuration for all environments
- ❌ **Hardcoded JPMC endpoints** - no local development support

```ini
# main branch config.ini (legacy)
[azure_openai]
azure_openai_endpoint = https://llm-multitenancy-exp.jpmchase.net/ver2/
deployment_name = gpt-4o-2024-08-06
api_version = 2024-10-21

[aws_info]
opensearch_endpoint = https://utilitiesassist.dev.aws.jpmchase.net
index_name = khub-opensearch-index
```

### Phase1 Branch (Modern)
**File**: `src/infra/config.py` - Profile-based switching with environment injection

✅ **Profile-driven architecture** using `CLOUD_PROFILE` environment variable  
✅ **Type-safe Pydantic models** with validation  
✅ **Zero hardcoded secrets** - all injected via environment  
✅ **Local development support** with OpenAI fallback

```python
# phase1 branch config.py (modern)
@lru_cache(1)
def load_settings() -> Settings:
    profile = os.getenv("CLOUD_PROFILE", "local").lower()
    
    if profile == "jpmc_azure":
        return _jpmc()      # Production JPMC
    elif profile == "tests":
        return _tests()     # Test configuration  
    else:
        return _local()     # Local development (default)
```

## Environment Variable Matrix

### Local Development Profile (`CLOUD_PROFILE=local`)

**Required Environment Variables**:
```bash
# .env.local (based on .env.local.example)
CLOUD_PROFILE=local
OPENAI_API_KEY=sk-your-openai-api-key-here
OS_HOST=http://localhost:9200                    # Optional - defaults to localhost:9200
```

**Configuration Resolution**:
- **Chat Provider**: OpenAI public API
- **Embedding Provider**: OpenAI public API (`text-embedding-3-small`)
- **Search Backend**: Local OpenSearch container
- **Authentication**: OpenAI API key
- **Proxy**: None (direct internet access)

### JPMC Azure Profile (`CLOUD_PROFILE=jpmc_azure`)

**Required Environment Variables**:
```bash
# .env.jpmc (based on .env.jpmc.example)
CLOUD_PROFILE=jpmc_azure

# Azure OpenAI Authentication
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com
AZURE_TENANT_ID=your-azure-tenant-id
AZURE_CLIENT_ID=your-azure-client-id
AZURE_CLIENT_SECRET=your-azure-client-secret

# Azure Deployment Names
AZURE_CHAT_DEPLOYMENT=gpt-4o
AZURE_EMBED_DEPLOYMENT=text-embedding-3-small

# Enterprise OpenSearch
OPENSEARCH_ENDPOINT=https://utilitiesassist.dev.aws.jpmchase.net
OPENSEARCH_INDEX=khub-opensearch-index
OPENSEARCH_TIMEOUT=2.5

# Optional AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key

# Optional JPMC-specific headers
JPMC_USER_SID=user-session-id
```

**Configuration Resolution**:
- **Chat Provider**: Azure OpenAI with AAD authentication
- **Embedding Provider**: Azure OpenAI with AAD authentication  
- **Search Backend**: Enterprise OpenSearch with AWS4Auth
- **Authentication**: Azure AAD certificate + AWS role-based
- **Proxy**: JPMC corporate proxy (`proxy.jpmchase.net:10443`)

### Test Profile (`CLOUD_PROFILE=tests`)

**Required Environment Variables**:
```bash
CLOUD_PROFILE=tests
OPENAI_API_KEY=sk-test-key  # Can be dummy for unit tests
```

**Configuration Resolution**:
- **Chat Provider**: OpenAI with minimal model (`gpt-3.5-turbo`)
- **Embedding Provider**: OpenAI with standard embedding model
- **Search Backend**: Local OpenSearch (localhost:9200)
- **Authentication**: API key (can be mocked)
- **Proxy**: None

## Decision Tree Logic

The configuration system uses a **single decision point** with **no code edits required**:

```python
# src/infra/config.py:97-104 (the only decision logic)
profile = os.getenv("CLOUD_PROFILE", "local").lower()

if profile == "jpmc_azure":
    return _jpmc()      # → Azure OpenAI + Enterprise OpenSearch + AWS4Auth + JPMC Proxy
elif profile == "tests":  
    return _tests()     # → OpenAI + Local OpenSearch + Minimal Config
else:
    return _local()     # → OpenAI + Local OpenSearch + Direct Access (DEFAULT)
```

**Key Benefits**:
- ✅ **Zero code changes** between environments
- ✅ **Fail-safe defaults** - missing `CLOUD_PROFILE` → local development
- ✅ **Type safety** - Pydantic validates all configurations at startup
- ✅ **Hot swapping** - Change `CLOUD_PROFILE` and restart, no deployments

## Configuration Templates

### .env.local.example (Local Development)
```bash
# .env.local.example - Local development configuration
# Copy to .env and fill in your OpenAI API key

CLOUD_PROFILE=local

# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here

# Local OpenSearch (optional - defaults to localhost:9200)
OS_HOST=http://localhost:9200

# Optional: Override default models
# OPENAI_CHAT_MODEL=gpt-4o-mini
# OPENAI_EMBED_MODEL=text-embedding-3-small
```

### .env.jpmc.example (JPMC Production)
```bash
# .env.jpmc.example - JPMC enterprise configuration  
# Copy to .env and fill in your Azure/enterprise credentials

CLOUD_PROFILE=jpmc_azure

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://llm-multitenancy-exp.jpmchase.net/ver2/
AZURE_TENANT_ID=<AZURE_TENANT_ID>
AZURE_CLIENT_ID=<AZURE_CLIENT_ID>
AZURE_CLIENT_SECRET=<AZURE_CLIENT_SECRET>

# Azure OpenAI Deployment Names  
AZURE_CHAT_DEPLOYMENT=gpt-4o-2024-08-06
AZURE_EMBED_DEPLOYMENT=text-embedding-3-small-1
AZURE_API_VERSION=2024-10-21

# Enterprise OpenSearch Configuration
OPENSEARCH_ENDPOINT=https://utilitiesassist.dev.aws.jpmchase.net
OPENSEARCH_INDEX=khub-opensearch-index
OPENSEARCH_TIMEOUT=2.5

# AWS Configuration for OpenSearch Authentication
AWS_REGION=us-east-1

# Optional JPMC Enterprise Headers
JPMC_USER_SID=<USER_SESSION_ID>
```

## Deployment Verification Checklist

### Startup Configuration Verification
Add this validation to application startup (`src/infra/config.py`):

```python
def validate_jpmc_production():
    """Validate JPMC production configuration at startup."""
    settings = load_settings()
    
    print(f"🔧 Active Profile: {settings.profile}")
    print(f"💬 Chat Provider: {settings.chat.provider} ({settings.chat.model})")
    print(f"🔗 Chat Endpoint: {settings.chat.api_base or 'default'}")
    print(f"🧠 Embed Provider: {settings.embed.provider} ({settings.embed.model})")
    print(f"🔍 Search Host: {settings.search.host}")
    print(f"📚 Search Index: {settings.search.index_alias}")
    
    # Assert critical production requirements
    if settings.profile == "jpmc_azure":
        assert settings.chat.provider == "azure", f"Expected Azure chat, got {settings.chat.provider}"
        assert settings.embed.provider == "azure", f"Expected Azure embeddings, got {settings.embed.provider}"  
        assert settings.embed.dims == 1536, f"Expected 1536 embedding dims, got {settings.embed.dims}"
        assert "jpmchase.net" in settings.search.host, f"Expected JPMC OpenSearch host, got {settings.search.host}"
        assert settings.search.index_alias == "confluence_current", f"Expected production index alias, got {settings.search.index_alias}"
        print("✅ JPMC production configuration validated")
    
    elif settings.profile == "local":
        assert settings.chat.provider == "openai", f"Expected OpenAI chat, got {settings.chat.provider}"
        assert settings.embed.provider == "openai", f"Expected OpenAI embeddings, got {settings.embed.provider}"
        print("✅ Local development configuration validated")
    
    return settings
```

### Runtime Environment Assertion
Add environment flag blocking for production (`src/infra/config.py`):

```python
def assert_no_dev_flags():
    """Assert no development flags are enabled in JPMC production."""
    if os.getenv("CLOUD_PROFILE") == "jpmc_azure":
        blocked_flags = [
            ("USE_MOCK_SEARCH", "Mock search disabled in production"),
            ("USE_LOCAL_AZURE", "Local Azure config disabled in production")  
        ]
        
        for flag, message in blocked_flags:
            if os.getenv(flag, "").lower() == "true":
                raise EnvironmentError(f"🚫 {message}: {flag}=true not allowed with CLOUD_PROFILE=jpmc_azure")
        
        print("🛡️  Production environment flags validated")
```

### Embedding Dimensions Validation
Ensure embedding compatibility at startup:

```python
def validate_embedding_dims():
    """Validate embedding dimensions match expected values."""  
    settings = load_settings()
    expected_dims = 1536  # Standard for text-embedding-3-small
    
    if settings.embed.dims != expected_dims:
        raise ValueError(f"❌ Embedding dimension mismatch: expected {expected_dims}, got {settings.embed.dims}")
    
    print(f"🔢 Embedding dimensions validated: {settings.embed.dims}")
```

## Migration Execution Plan

### Step 1: Environment Preparation
```bash
# 1. Copy JPMC template
cp .env.jpmc.example .env

# 2. Fill in actual secrets (Azure tenant, client ID, etc.)
vi .env

# 3. Set JPMC profile
export CLOUD_PROFILE=jpmc_azure

# 4. Validate no dev flags
unset USE_MOCK_SEARCH
unset USE_LOCAL_AZURE
```

### Step 2: Service Startup  
```bash
# 1. Validate configuration
python -c "from src.infra.config import validate_jpmc_production; validate_jpmc_production()"

# 2. Start application
streamlit run streamlit_app.py
# OR
python -m src.app.chat_interface
```

### Step 3: Runtime Verification
```bash
# 1. Check active profile in logs
grep "Active Profile: jpmc_azure" logs/application.log

# 2. Verify Azure OpenAI connectivity  
grep "Azure OpenAI clients initialized successfully" logs/application.log

# 3. Verify OpenSearch connectivity
grep "OpenSearch connection established" logs/application.log

# 4. Verify no mocks loaded
grep -v "Mock.*loaded" logs/application.log

# 5. Test end-to-end query
curl -X POST localhost:8501/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Customer Summary Utility?"}'
```

## Troubleshooting Guide

### Common Configuration Errors

#### ❌ Missing CLOUD_PROFILE
**Symptom**: Application uses local OpenAI instead of Azure  
**Cause**: `CLOUD_PROFILE` environment variable not set  
**Fix**: `export CLOUD_PROFILE=jpmc_azure`

#### ❌ Azure Authentication Failure  
**Symptom**: `Azure authentication failed: invalid_client`  
**Cause**: Missing or incorrect `AZURE_CLIENT_SECRET`  
**Fix**: Verify Azure credentials in `.env` file

#### ❌ OpenSearch Connection Timeout
**Symptom**: `OpenSearch connection timeout after 2.5s`  
**Cause**: Network connectivity or wrong `OPENSEARCH_ENDPOINT`  
**Fix**: Verify JPMC network access and endpoint URL

#### ❌ Mock Flags in Production
**Symptom**: `Mock search disabled in production: USE_MOCK_SEARCH=true`  
**Cause**: Development flags still set  
**Fix**: `unset USE_MOCK_SEARCH USE_LOCAL_AZURE`

#### ❌ Wrong Index Name
**Symptom**: `Index khub-opensearch-swagger-index does not exist`  
**Cause**: Incorrect `OPENSEARCH_INDEX` in environment  
**Fix**: Set `OPENSEARCH_INDEX=confluence_current` (or actual production index)

### Configuration Debugging Commands

```bash
# Print active configuration (without secrets)
python -c "
from src.infra.config import load_settings
settings = load_settings()
print(f'Profile: {settings.profile}')
print(f'Chat: {settings.chat.provider}/{settings.chat.model}')  
print(f'Embed: {settings.embed.provider}/{settings.embed.model}')
print(f'Search: {settings.search.host}')
"

# Test OpenSearch connectivity
curl -X GET "${OPENSEARCH_ENDPOINT}/_cluster/health" \
  -H "Authorization: AWS4-HMAC-SHA256 Credential=..."

# Test Azure OpenAI connectivity  
curl -X POST "${AZURE_OPENAI_ENDPOINT}/openai/deployments/${AZURE_CHAT_DEPLOYMENT}/chat/completions?api-version=${AZURE_API_VERSION}" \
  -H "Authorization: Bearer ${azure_token}" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"test"}],"max_tokens":1}'
```

---

## Summary

✅ **Configuration switching requires ZERO code changes**  
✅ **Single decision point**: `CLOUD_PROFILE` environment variable  
✅ **Type-safe validation**: Pydantic models prevent configuration errors  
✅ **Fail-safe defaults**: Missing config → local development mode  
✅ **Runtime verification**: Startup validation catches misconfigurations early  

**To deploy to JPMC**: Set `CLOUD_PROFILE=jpmc_azure` + populate Azure/AWS secrets → Done.

**Next Steps**: 
1. Test configuration switching in staging environment
2. Validate Azure AAD token flow end-to-end  
3. Confirm OpenSearch ACL filters work with production data
4. Execute mock removal plan from [01-mocks-audit.md](01-mocks-audit.md)