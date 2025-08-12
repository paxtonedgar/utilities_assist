# src/infra/config.py
# Switching between local and JPMC only requires CLOUD_PROFILE + env secrets. No code changes.

from pydantic import BaseModel, Field
from functools import lru_cache
import os


class ChatCfg(BaseModel):
    """Configuration for chat/LLM providers"""
    provider: str           # "openai" | "azure"
    model: str
    api_base: str | None = None  # for azure
    api_version: str | None = None


class EmbedCfg(BaseModel):
    """Configuration for embedding providers"""
    provider: str
    model: str
    dims: int


class SearchCfg(BaseModel):
    """Configuration for OpenSearch/Elasticsearch"""
    host: str
    index_alias: str = "confluence_current"
    username: str | None = None
    password: str | None = None
    timeout_s: float = 2.5


class Settings(BaseModel):
    """Unified configuration with profile-based switching"""
    profile: str = Field(default=os.getenv("CLOUD_PROFILE", "local"))
    chat: ChatCfg
    embed: EmbedCfg
    search: SearchCfg


def _local() -> Settings:
    """Local development profile using OpenAI"""
    return Settings(
        profile="local",
        chat=ChatCfg(provider="openai", model="gpt-4o-mini"),
        embed=EmbedCfg(provider="openai", model="text-embedding-3-small", dims=1536),
        search=SearchCfg(host=os.getenv("OS_HOST", "http://localhost:9200"))
    )


def _jpmc() -> Settings:
    """JPMC production profile using Azure OpenAI with AAD and enterprise endpoints"""
    return Settings(
        profile="jpmc_azure", 
        chat=ChatCfg(
            provider="azure", 
            model=os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-2024-08-06"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT", "https://llm-multitenancy-exp.jpmchase.net/ver2/"), 
            api_version=os.getenv("AZURE_API_VERSION", "2024-10-21")
        ),
        embed=EmbedCfg(
            provider="azure", 
            model=os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small-1"), 
            dims=1536
        ),
        search=SearchCfg(
            host=os.getenv("OPENSEARCH_ENDPOINT", "https://utilitiesassist.dev.aws.jpmchase.net"), 
            index_alias=os.getenv("OPENSEARCH_INDEX", "khub-opensearch-index"),
            # No username/password - uses AWS4Auth
            username=None, 
            password=None,
            timeout_s=float(os.getenv("OPENSEARCH_TIMEOUT", "2.5"))
        )
    )


def _tests() -> Settings:
    """Test profile with minimal configuration"""
    return Settings(
        profile="tests",
        chat=ChatCfg(provider="openai", model="gpt-3.5-turbo"),
        embed=EmbedCfg(provider="openai", model="text-embedding-3-small", dims=1536),
        search=SearchCfg(host="http://localhost:9200")
    )


# Optional minimal config caching - can be disabled if config changes frequently
@lru_cache(1)  # Single config instance only
def load_settings() -> Settings:
    """Load settings based on CLOUD_PROFILE environment variable.
    
    Profiles:
    - local: OpenAI with local OpenSearch (default)
    - jpmc_azure: Azure OpenAI with AAD and enterprise OpenSearch  
    - tests: Minimal test configuration
    """
    profile = os.getenv("CLOUD_PROFILE", "local").lower()
    
    if profile == "jpmc_azure":
        return _jpmc()
    elif profile == "tests":
        return _tests()
    else:
        return _local()


# Convenience function for quick access
def get_settings() -> Settings:
    """Get current settings - cached singleton"""
    return load_settings()