# src/infra/clients.py
"""Client factory for creating LLM, embedding, and search clients.

No singletons - provides request-scoped clients with efficient LRU caching.
Supports both OpenAI and Azure providers with flexible token management.
"""

from functools import lru_cache
from typing import Callable, Any
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import ChatCfg, EmbedCfg, SearchCfg


def _token_fingerprint(token_provider: Callable[[], str] | None) -> str:
    """Generate a cache key fingerprint for token provider."""
    if token_provider is None:
        return "none"
    
    # Use function name and id for fingerprinting
    # This ensures different token providers get different cache entries
    func_id = f"{token_provider.__name__}:{id(token_provider)}"
    return hashlib.md5(func_id.encode()).hexdigest()[:8]


@lru_cache(maxsize=8)
def _cached_chat_client(
    provider: str, 
    model: str, 
    api_base: str | None, 
    api_version: str | None,
    token_fingerprint: str
) -> Any:
    """Internal cached chat client creation."""
    if provider == "openai":
        import os
        from openai import OpenAI
        
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=5.0
        )
    
    elif provider == "azure":
        import os
        from openai import AzureOpenAI
        
        # For Azure, we'll handle token in the wrapper
        return AzureOpenAI(
            api_key=os.getenv("AZURE_CLIENT_SECRET", "dummy"),
            api_version=api_version,
            azure_endpoint=api_base,
            timeout=5.0
        )
    
    else:
        raise ValueError(f"Unsupported chat provider: {provider}")


@lru_cache(maxsize=8)
def _cached_embed_client(
    provider: str,
    model: str, 
    dims: int,
    token_fingerprint: str
) -> Any:
    """Internal cached embedding client creation."""
    if provider == "openai":
        import os
        from openai import OpenAI
        
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=5.0
        )
    
    elif provider == "azure":
        import os
        from openai import AzureOpenAI
        
        return AzureOpenAI(
            api_key=os.getenv("AZURE_CLIENT_SECRET", "dummy"),
            api_version="2024-06-01",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            timeout=5.0
        )
    
    else:
        raise ValueError(f"Unsupported embed provider: {provider}")


@lru_cache(maxsize=8)
def _cached_search_session(
    host: str,
    index_alias: str,
    username: str | None,
    password: str | None, 
    timeout_s: float
) -> requests.Session:
    """Internal cached search session creation."""
    session = requests.Session()
    
    # Configure retries with exponential backoff
    retry_strategy = Retry(
        total=2,
        backoff_factor=0.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
    )
    
    # HTTP adapter with retry strategy
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,
        pool_maxsize=20
    )
    
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Configure timeouts
    session.timeout = (2.0, timeout_s)  # (connect, read)
    
    # Configure auth if provided
    if username and password:
        session.auth = (username, password)
    
    # Keep-alive headers
    session.headers.update({
        'Connection': 'keep-alive',
        'Content-Type': 'application/json',
        'User-Agent': 'utilities-assist/1.0'
    })
    
    return session


def make_chat_client(cfg: ChatCfg, token_provider: Callable[[], str] | None = None) -> Any:
    """Create a chat/LLM client with LRU caching.
    
    Args:
        cfg: Chat configuration (provider, model, api_base, etc.)
        token_provider: Optional callable returning Bearer token for Azure
        
    Returns:
        OpenAI or Azure OpenAI client instance
        
    Cache key: (provider, model, api_base, api_version, token_fingerprint)
    """
    client = _cached_chat_client(
        cfg.provider,
        cfg.model,
        cfg.api_base,
        cfg.api_version,
        _token_fingerprint(token_provider)
    )
    
    # Add token to headers if Azure and token provider available
    if cfg.provider == "azure" and token_provider:
        token = token_provider()
        # Create a new client with token headers for Azure
        if not hasattr(client, '_token_headers_set'):
            client.default_headers = client.default_headers or {}
            client.default_headers["Authorization"] = f"Bearer {token}"
            client._token_headers_set = True
    
    return client


def make_embed_client(cfg: EmbedCfg, token_provider: Callable[[], str] | None = None) -> Any:
    """Create an embedding client with LRU caching.
    
    Args:
        cfg: Embedding configuration (provider, model, dimensions)
        token_provider: Optional callable returning Bearer token for Azure
        
    Returns:
        OpenAI or Azure OpenAI client instance for embeddings
        
    Cache key: (provider, model, dims, token_fingerprint)
    """
    client = _cached_embed_client(
        cfg.provider,
        cfg.model,
        cfg.dims,
        _token_fingerprint(token_provider)
    )
    
    # Add token to headers if Azure and token provider available
    if cfg.provider == "azure" and token_provider:
        token = token_provider()
        if not hasattr(client, '_token_headers_set'):
            client.default_headers = client.default_headers or {}
            client.default_headers["Authorization"] = f"Bearer {token}"
            client._token_headers_set = True
    
    return client


def make_search_session(cfg: SearchCfg) -> requests.Session:
    """Create a pooled requests session for OpenSearch/Elasticsearch.
    
    Args:
        cfg: Search configuration (host, auth, timeout)
        
    Returns:
        Configured requests.Session with retries, timeouts, and keep-alive
        
    Features:
    - HTTP retries: 2 total, backoff 0.2-0.5s
    - Read timeout: configurable, Connect timeout: 2.0s
    - Keep-alive enabled
    - Basic auth if credentials provided
    """
    return _cached_search_session(
        cfg.host,
        cfg.index_alias,
        cfg.username,
        cfg.password,
        cfg.timeout_s
    )


def clear_client_cache():
    """Clear all client caches. Useful for testing or token refresh scenarios."""
    _cached_chat_client.cache_clear()
    _cached_embed_client.cache_clear()
    _cached_search_session.cache_clear()


def get_cache_info() -> dict:
    """Get cache statistics for monitoring and debugging."""
    return {
        "chat_client": _cached_chat_client.cache_info(),
        "embed_client": _cached_embed_client.cache_info(),
        "search_session": _cached_search_session.cache_info()
    }