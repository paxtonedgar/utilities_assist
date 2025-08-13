# src/infra/clients.py
"""Client factory for creating LLM, embedding, and search clients.

No singletons - provides request-scoped clients with efficient LRU caching.
Supports both OpenAI and Azure providers with flexible token management.
Includes JPMC proxy configuration and AWS authentication support.
"""

from functools import lru_cache
from typing import Callable, Any
import hashlib
import requests
import os
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.infra.config import ChatCfg, EmbedCfg, SearchCfg

logger = logging.getLogger(__name__)


def _token_fingerprint(token_provider: Callable[[], str] | None) -> str:
    """Generate a cache key fingerprint for token provider."""
    if token_provider is None:
        return "none"
    
    # Use function name and id for fingerprinting
    # This ensures different token providers get different cache entries
    func_id = f"{token_provider.__name__}:{id(token_provider)}"
    return hashlib.md5(func_id.encode()).hexdigest()[:8]


def _setup_jpmc_proxy():
    """Configure JPMC proxy settings if in JPMC environment."""
    profile = os.getenv("CLOUD_PROFILE", "local").lower()
    if profile == "jpmc_azure":
        os.environ["http_proxy"] = "proxy.jpmchase.net:10443"
        os.environ["https_proxy"] = "proxy.jpmchase.net:10443"
        if 'no_proxy' in os.environ:
            os.environ['no_proxy'] = os.environ['no_proxy'] + ",jpmchase.net,openai.azure.com"
        else:
            os.environ['no_proxy'] = 'localhost,127.0.0.1,jpmchase.net,openai.azure.com'
        logger.info("JPMC proxy configuration applied")


# Optional minimal LRU cache for HTTP connection pooling only
@lru_cache(maxsize=2)  # Keep minimal - just current and previous client
def _cached_chat_client(
    provider: str, 
    model: str, 
    api_base: str | None, 
    api_version: str | None,
    token_fingerprint: str
) -> Any:
    """Internal cached chat client creation."""
    # Apply JPMC proxy if needed
    _setup_jpmc_proxy()
    
    if provider == "openai":
        from openai import OpenAI
        
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=5.0
        )
    
    elif provider == "azure":
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


# Optional minimal LRU cache for HTTP connection pooling only  
@lru_cache(maxsize=2)  # Keep minimal - just current and previous client
def _cached_embed_client(
    provider: str,
    model: str, 
    dims: int,
    token_fingerprint: str
) -> Any:
    """Internal cached embedding client creation."""
    # Apply JPMC proxy if needed
    _setup_jpmc_proxy()
    
    if provider == "openai":
        from openai import OpenAI
        
        return OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=5.0
        )
    
    elif provider == "azure":
        from openai import AzureOpenAI
        
        # Get the endpoint from environment or use fallback
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            try:
                from src.infra.config import get_settings
                settings = get_settings()
                azure_endpoint = settings.chat.api_base
            except:
                azure_endpoint = "https://llm-multitenancy-exp.jpmchase.net/ver2/"
        
        return AzureOpenAI(
            api_key=os.getenv("AZURE_CLIENT_SECRET", "dummy"),
            api_version="2024-06-01",
            azure_endpoint=azure_endpoint,
            timeout=5.0
        )
    
    else:
        raise ValueError(f"Unsupported embed provider: {provider}")


def _get_aws_auth():
    """Get AWS4Auth for OpenSearch authentication in JPMC environment."""
    try:
        import boto3
        from requests_aws4auth import AWS4Auth
        
        # Copy exact code from working main branch
        session = boto3.Session()
        logger.info(session)
        region = 'us-east-1'
        credentials = session.get_credentials()
        logger.info(credentials)
        return AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
        
    except ImportError:
        logger.warning("AWS4Auth not available - install requests-aws4auth for JPMC OpenSearch")
        return None
    except Exception as e:
        logger.error(f"Failed to configure AWS authentication: {e}")
        return None


# Essential LRU cache for HTTP connection pooling to OpenSearch/ElasticSearch
@lru_cache(maxsize=2)  # Keep minimal - just current and previous session
def _cached_search_session(
    host: str,
    index_alias: str,
    username: str | None,
    password: str | None, 
    timeout_s: float,
    use_aws_auth: bool = False
) -> requests.Session:
    """Internal cached search session creation."""
    # Apply JPMC proxy if needed
    _setup_jpmc_proxy()
    
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
    
    # Configure authentication
    if use_aws_auth:
        # Use AWS4Auth for JPMC OpenSearch
        aws_auth = _get_aws_auth()
        if aws_auth:
            session.auth = aws_auth
            logger.info("AWS4Auth configured for OpenSearch")
    elif username and password:
        # Use basic auth
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
    # For Azure, always create a fresh client to handle token provider or direct API key
    # instead of using the cached version
    if cfg.provider == "azure":
        from openai import AzureOpenAI
        
        # Get API key from config (required base authentication)
        api_key = None
        headers = {"user_sid": os.getenv("JPMC_USER_SID", "REPLACE")}
        
        try:
            from utils import load_config
            config = load_config()
            # Get API key from config
            api_key = config.get('azure_openai', 'api_key', fallback=None)
            if not api_key:
                # Debug: show what keys are actually available
                if config.has_section('azure_openai'):
                    available_keys = list(config['azure_openai'].keys())
                    logger.error(f"No API key found in [azure_openai]. Available keys: {available_keys}")
                    raise ValueError("No API key found in config - this is required as base authentication")
                else:
                    logger.error("No [azure_openai] section found in config")
                    raise ValueError("No [azure_openai] section found in config")
            
            logger.info("Got API key from config.ini for chat client")
        except Exception as config_error:
            logger.error(f"Failed to get API key from config: {config_error}")
            raise ValueError("API key from config is required for Azure authentication")
        
        # Additionally, try to get Bearer token from certificate (like main branch does)
        if token_provider:
            try:
                bearer_token = token_provider()
                headers["Authorization"] = f"Bearer {bearer_token}"
                logger.info("Added Bearer token from certificate for chat client")
            except Exception as e:
                logger.warning(f"Bearer token failed ({e}), using API key only for chat client")
        else:
            logger.info("No certificate token provider - using API key only for chat client")
        
        # Apply JPMC proxy if needed
        _setup_jpmc_proxy()
        
        return AzureOpenAI(
            api_key=api_key,
            api_version=cfg.api_version,
            azure_endpoint=cfg.api_base,
            timeout=5.0,
            default_headers=headers
        )
    
    # For other cases, use cached client
    return _cached_chat_client(
        cfg.provider,
        cfg.model,
        cfg.api_base,
        cfg.api_version,
        _token_fingerprint(token_provider)
    )


def make_embed_client(cfg: EmbedCfg, token_provider: Callable[[], str] | None = None) -> Any:
    """Create an embedding client with LRU caching.
    
    Args:
        cfg: Embedding configuration (provider, model, dimensions)
        token_provider: Optional callable returning Bearer token for Azure
        
    Returns:
        OpenAI or Azure OpenAI client instance for embeddings
        
    Cache key: (provider, model, dims, token_fingerprint)
    """
    # For Azure, always create a fresh client to handle token provider or direct API key
    # instead of using the cached version
    if cfg.provider == "azure":
        from openai import AzureOpenAI
        
        # Get API key from config (required base authentication)
        api_key = None
        headers = {"user_sid": os.getenv("JPMC_USER_SID", "REPLACE")}
        
        try:
            from utils import load_config
            config = load_config()
            # Get API key from config
            api_key = config.get('azure_openai', 'api_key', fallback=None)
            if not api_key:
                # Debug: show what keys are actually available
                if config.has_section('azure_openai'):
                    available_keys = list(config['azure_openai'].keys())
                    logger.error(f"No API key found in [azure_openai]. Available keys: {available_keys}")
                    raise ValueError("No API key found in config - this is required as base authentication")
                else:
                    logger.error("No [azure_openai] section found in config")
                    raise ValueError("No [azure_openai] section found in config")
            
            logger.info("Got API key from config.ini for embed client")
        except Exception as config_error:
            logger.error(f"Failed to get API key from config: {config_error}")
            raise ValueError("API key from config is required for Azure authentication")
        
        # Additionally, try to get Bearer token from certificate (like main branch does)
        if token_provider:
            try:
                bearer_token = token_provider()
                headers["Authorization"] = f"Bearer {bearer_token}"
                logger.info("Added Bearer token from certificate for embed client")
            except Exception as e:
                logger.warning(f"Bearer token failed ({e}), using API key only for embed client")
        else:
            logger.info("No certificate token provider - using API key only for embed client")
        
        # Apply JPMC proxy if needed
        _setup_jpmc_proxy()
        
        # Get the endpoint from environment or use the one from chat config if available
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            # Try to get settings and use chat endpoint as fallback
            try:
                from src.infra.config import get_settings
                settings = get_settings()
                azure_endpoint = settings.chat.api_base
            except:
                azure_endpoint = "https://llm-multitenancy-exp.jpmchase.net/ver2/"
        
        return AzureOpenAI(
            api_key=api_key,
            api_version="2024-06-01",
            azure_endpoint=azure_endpoint,
            timeout=5.0,
            default_headers=headers
        )
    
    # For other cases, use cached client
    return _cached_embed_client(
        cfg.provider,
        cfg.model,
        cfg.dims,
        _token_fingerprint(token_provider)
    )


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
    - Basic auth or AWS4Auth for JPMC environment
    """
    # Use AWS auth for JPMC environment
    profile = os.getenv("CLOUD_PROFILE", "local").lower()
    use_aws_auth = profile == "jpmc_azure"
    
    return _cached_search_session(
        cfg.host,
        cfg.index_alias,
        cfg.username,
        cfg.password,
        cfg.timeout_s,
        use_aws_auth
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