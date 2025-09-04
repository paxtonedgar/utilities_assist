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

# Third-party imports (moved to top to avoid repeated dynamic imports)
try:
    from openai import OpenAI, AzureOpenAI
except ImportError:
    OpenAI = AzureOpenAI = None
    logging.warning("OpenAI library not available - client creation will fail")

try:
    import boto3
    from requests_aws4auth import AWS4Auth
except ImportError:
    boto3 = AWS4Auth = None
    logging.debug("AWS libraries not available - AWS auth will be disabled")

# Internal imports
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


# Cache with environment-aware key to handle CLOUD_PROFILE changes
@lru_cache(maxsize=4)
def _setup_jpmc_proxy_cached(cloud_profile: str):
    """Internal cached proxy setup with environment key."""
    if cloud_profile == "jpmc_azure":
        os.environ["http_proxy"] = "proxy.jpmchase.net:10443"
        os.environ["https_proxy"] = "proxy.jpmchase.net:10443"
        if "no_proxy" in os.environ:
            os.environ["no_proxy"] = (
                os.environ["no_proxy"] + ",jpmchase.net,openai.azure.com"
            )
        else:
            os.environ["no_proxy"] = "localhost,127.0.0.1,jpmchase.net,openai.azure.com"
        logger.info(f"JPMC proxy configuration applied for {cloud_profile} (cached)")


def _setup_jpmc_proxy():
    """Configure JPMC proxy settings if in JPMC environment. Handles environment changes properly."""
    profile = os.getenv("CLOUD_PROFILE", "local").lower()
    _setup_jpmc_proxy_cached(profile)


# Optional minimal LRU cache for HTTP connection pooling only
@lru_cache(maxsize=2)  # Keep minimal - just current and previous client
def _cached_chat_client(
    provider: str,
    model: str,
    api_base: str | None,
    api_version: str | None,
    token_fingerprint: str,  # Used for cache key
) -> Any:
    """Internal cached chat client creation."""
    if not OpenAI or not AzureOpenAI:
        raise ImportError("OpenAI library not available")

    # Apply JPMC proxy if needed
    _setup_jpmc_proxy()

    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=5.0)

    elif provider == "azure":
        # NOTE: This cached function should not be used for Azure in production
        # Azure clients use the direct authentication path instead
        return AzureOpenAI(
            api_key="dummy-not-used-for-azure",  # Cached Azure clients are bypassed
            api_version=api_version,
            azure_endpoint=api_base,
            timeout=5.0,
        )

    else:
        raise ValueError(f"Unsupported chat provider: {provider}")


# Optional minimal LRU cache for HTTP connection pooling only
@lru_cache(maxsize=2)  # Keep minimal - just current and previous client
def _cached_embed_client(
    provider: str,
    model: str,
    dims: int,
    token_fingerprint: str,  # Used for cache key
) -> Any:
    """Internal cached embedding client creation."""
    if not OpenAI or not AzureOpenAI:
        raise ImportError("OpenAI library not available")

    # Apply JPMC proxy if needed
    _setup_jpmc_proxy()

    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=5.0)

    elif provider == "azure":
        # Get the endpoint from environment or use fallback
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not azure_endpoint:
            try:
                from src.infra.config import get_settings

                settings = get_settings()
                azure_endpoint = settings.chat.api_base
            except:
                azure_endpoint = "https://llm-multitenancy-exp.jpmchase.net/ver2/"

        # NOTE: This cached function should not be used for Azure in production
        # Azure clients use the direct authentication path instead
        return AzureOpenAI(
            api_key="dummy-not-used-for-azure",  # Cached Azure clients are bypassed
            api_version="2024-06-01",
            azure_endpoint=azure_endpoint,
            timeout=5.0,
        )

    else:
        raise ValueError(f"Unsupported embed provider: {provider}")


@lru_cache(maxsize=1)
def _get_aws_auth():
    """Get AWS4Auth for OpenSearch authentication in JPMC environment.

    Cached to avoid repeated authentication setup.
    """
    if not boto3 or not AWS4Auth:
        logger.warning(
            "AWS libraries not available - install boto3 and requests-aws4auth for JPMC OpenSearch"
        )
        return None

    try:
        # Resolve region from env or settings, fallback to us-east-1
        region = os.getenv("OPENSEARCH_REGION")
        if not region:
            try:
                from src.infra.settings import get_settings

                st = get_settings()
                if st.aws_info and getattr(st.aws_info, "aws_region", None):
                    region = st.aws_info.aws_region
            except Exception:
                region = None
        if not region:
            region = "us-east-1"

        session = boto3.Session()
        logger.debug(f"AWS session: {session}")
        credentials = session.get_credentials()
        logger.debug(f"AWS credentials loaded for region {region}")
        return AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            region,
            "es",
            session_token=credentials.token,
        )

    except Exception as e:
        logger.error(f"Failed to configure AWS authentication: {e}")
        return None




def _create_azure_client(
    client_type: str,
    api_version: str,
    azure_endpoint: str,
    token_provider: Callable[[], str] | None = None,
) -> Any:
    """Consolidated Azure client creation with shared authentication logic.

    Args:
        client_type: "chat" or "embed" for logging
        api_version: Azure API version
        azure_endpoint: Azure OpenAI endpoint
        token_provider: Optional callable returning Bearer token

    Returns:
        Configured AzureOpenAI client
    """
    if not AzureOpenAI:
        raise ImportError("OpenAI library not available")

    # Get API key from config (required base authentication)
    api_key = None
    headers = {"user_sid": os.getenv("JPMC_USER_SID", "REPLACE")}

    try:
        from src.infra.settings import get_config_value

        api_key = get_config_value("azure_openai", "api_key")
        if not api_key:
            logger.error("No API key found in [azure_openai] config section")
            raise ValueError(
                "No API key found in config - this is required as base authentication"
            )

        logger.info(f"Got API key from shared config for {client_type} client")
    except Exception as config_error:
        logger.error(f"Failed to get API key from shared config: {config_error}")
        raise ValueError("API key from config is required for Azure authentication")

    # Try to get Bearer token from certificate
    if token_provider:
        try:
            bearer_token = token_provider()
            headers["Authorization"] = f"Bearer {bearer_token}"
            logger.info(f"Added Bearer token from certificate for {client_type} client")
        except Exception as e:
            logger.warning(
                f"Bearer token failed ({e}), using API key only for {client_type} client"
            )
    else:
        logger.info(
            f"No certificate token provider - using API key only for {client_type} client"
        )

    # Apply JPMC proxy if needed
    _setup_jpmc_proxy()

    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        timeout=5.0,
        default_headers=headers,
    )


def make_chat_client(
    cfg: ChatCfg, token_provider: Callable[[], str] | None = None
) -> Any:
    """Create a chat/LLM client with LRU caching.

    Args:
        cfg: Chat configuration (provider, model, api_base, etc.)
        token_provider: Optional callable returning Bearer token for Azure

    Returns:
        OpenAI or Azure OpenAI client instance
    """
    if cfg.provider == "azure":
        return _create_azure_client(
            "chat", cfg.api_version, cfg.api_base, token_provider
        )

    # For other cases, use cached client
    return _cached_chat_client(
        cfg.provider,
        cfg.model,
        cfg.api_base,
        cfg.api_version,
        _token_fingerprint(token_provider),
    )


def make_embed_client(
    cfg: EmbedCfg, token_provider: Callable[[], str] | None = None
) -> Any:
    """Create an embedding client with LRU caching.

    Args:
        cfg: Embedding configuration (provider, model, dimensions)
        token_provider: Optional callable returning Bearer token for Azure

    Returns:
        OpenAI or Azure OpenAI client instance for embeddings
    """
    if cfg.provider == "azure":
        # Get the endpoint and API version from config (avoiding repeated dynamic imports)
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = "2024-10-21"  # Default

        try:
            from src.infra.config import get_settings

            settings = get_settings()
            if not azure_endpoint:
                azure_endpoint = settings.chat.api_base
            api_version = settings.chat.api_version
        except:
            if not azure_endpoint:
                azure_endpoint = "https://llm-multitenancy-exp.jpmchase.net/ver2/"

        return _create_azure_client(
            "embed", api_version, azure_endpoint, token_provider
        )

    # For other cases, use cached client
    return _cached_embed_client(
        cfg.provider, cfg.model, cfg.dims, _token_fingerprint(token_provider)
    )



