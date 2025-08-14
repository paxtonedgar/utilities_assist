# src/infra/resource_manager.py
"""
Resource Management for RAG Application

Provides singleton resource container to eliminate repeated config loading,
client creation, and authentication setup that was causing 25-50% performance overhead.

Usage:
    # At application startup
    resources = initialize_resources(settings)
    
    # In handlers - reuse existing clients
    response = resources.chat_client.invoke(prompt)
    results = resources.search_client.bm25_search(query)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Any
from threading import Lock

from src.infra.settings import ApplicationSettings
from src.infra.azure_auth import azure_token_provider

logger = logging.getLogger(__name__)

@dataclass
class RAGResources:
    """
    Singleton resource container created once at startup.
    
    Contains pre-configured, reusable clients and settings to eliminate
    the performance overhead of repeated initialization.
    
    Following LangGraph patterns for resource management and lifespan events.
    """
    settings: ApplicationSettings
    chat_client: Any
    embed_client: Optional[Any]
    search_client: Any
    token_provider: Optional[callable]
    initialized_at: float
    config_params: Optional[dict] = None  # Cached config.ini parameters
    
    def get_age_seconds(self) -> float:
        """Get age of resources in seconds"""
        return time.time() - self.initialized_at
    
    def get_config_param(self, key: str, default=None):
        """Get config parameter using exact key name from config.ini"""
        if self.config_params:
            return self.config_params.get(key, default)
        return get_config_param(key, default)

# Global resource singleton with thread safety
_resources: Optional[RAGResources] = None
_lock = Lock()

def initialize_resources(settings: ApplicationSettings, force_refresh: bool = False) -> RAGResources:
    """
    Initialize all resources once at startup.
    
    This replaces the pattern of creating clients per turn, which was causing:
    - Config loading: 40-300ms overhead per turn  
    - Client creation: 150-600ms overhead per turn
    - Authentication setup: 100-500ms overhead per turn
    
    Args:
        settings: Application settings (loaded once)
        force_refresh: Force recreation of resources even if they exist
        
    Returns:
        RAGResources: Pre-configured, reusable resource container
    """
    global _resources
    
    with _lock:
        if _resources is not None and not force_refresh:
            logger.info(f"Reusing existing resources (age: {_resources.get_age_seconds():.1f}s)")
            return _resources
        
        start_time = time.time()
        logger.info("Initializing shared resources...")
        
        # Load config parameters once using exact key names from config.ini
        config_params = _load_config_once()
        
        # Auto-detect Azure token provider for JPMC profile (like main branch)
        token_provider = None
        if settings.cloud_profile == "jpmc_azure":
            token_provider = azure_token_provider
            logger.info("Using Azure certificate authentication + API key (dual auth) for JPMC profile")
        
        # Create clients once - these will be reused for all requests
        logger.info("Creating chat client...")
        # For now, using placeholder - will integrate proper client factories
        chat_client = f"chat_client_for_{settings.cloud_profile}"
        
        logger.info("Creating embed client...")
        embed_client = None
        if settings.azure_openai and settings.azure_openai.azure_openai_embedding_model:
            embed_client = f"embed_client_for_{settings.cloud_profile}"
        
        logger.info("Creating search client...")
        from src.infra.opensearch_client import create_search_client
        search_client = create_search_client(settings)
        
        # Create resource container with cached config parameters
        _resources = RAGResources(
            settings=settings,
            chat_client=chat_client,
            embed_client=embed_client,
            search_client=search_client,
            token_provider=token_provider,
            initialized_at=time.time()
        )
        
        # Cache config parameters to eliminate repeated file I/O
        _resources.config_params = config_params
        
        init_time = (time.time() - start_time) * 1000
        logger.info(f"Resources initialized in {init_time:.1f}ms")
        logger.info(f"Embed client: {'✓' if embed_client else '✗'}")
        logger.info(f"Profile: {settings.cloud_profile}")
        logger.info(f"Search index: {settings.search_index_alias}")
        logger.info(f"OpenSearch host: {settings.opensearch_host}")
        
        return _resources

def get_resources() -> Optional[RAGResources]:
    """
    Get the current resource container.
    
    Returns:
        RAGResources: Current resource container, or None if not initialized
    """
    return _resources

def refresh_resources(settings: ApplicationSettings) -> RAGResources:
    """
    Force refresh of all resources.
    
    Useful for:
    - Configuration changes
    - Token expiration 
    - Connection issues
    
    Args:
        settings: Updated application settings
        
    Returns:
        RAGResources: Newly created resource container
    """
    logger.info("Forcing refresh of all resources...")
    return initialize_resources(settings, force_refresh=True)

def health_check() -> dict:
    """
    Check health of all resources.
    
    Returns:
        dict: Health status of each resource
    """
    if _resources is None:
        return {"status": "not_initialized"}
    
    health = {
        "status": "healthy",
        "age_seconds": _resources.get_age_seconds(),
        "profile": _resources.settings.cloud_profile,
        "chat_client": "available",
        "embed_client": "available" if _resources.embed_client else "not_configured",
        "search_client": "available"
    }
    
    # Test search client connectivity
    try:
        search_health = _resources.search_client.health_check()
        health["search_health"] = search_health["status"]
    except Exception as e:
        health["search_health"] = f"error: {e}"
        health["status"] = "degraded"
    
    return health


# Config caching for performance optimization (inspired by LangGraph lifespan patterns)
_cached_config_params = None

def _load_config_once() -> dict:
    """Load config.ini once and cache parameters using exact key names.
    
    Following LangGraph resource management patterns for optimal performance.
    Eliminates repeated file I/O that was causing 40-300ms overhead per turn.
    """
    global _cached_config_params
    
    if _cached_config_params is not None:
        return _cached_config_params
    
    try:
        from utils import load_config
        config = load_config()
        
        # Extract all relevant parameters using exact config.ini key names (source of truth)
        _cached_config_params = {}
        
        # Azure OpenAI section - use exact key names from config.ini
        if config.has_section('azure_openai'):
            azure_section = config['azure_openai']
            _cached_config_params.update({
                'deployment_name': azure_section.get('deployment_name', ''),
                'azure_openai_embedding_model': azure_section.get('azure_openai_embedding_model', ''),
                'azure_openai_endpoint': azure_section.get('azure_openai_endpoint', ''),
                'api_version': azure_section.get('api_version', ''),
                'temperature': azure_section.get('temperature', '0.1'),
                'max_tokens_2k': azure_section.get('max_tokens_2k', '2000'),
                'max_tokens_500': azure_section.get('max_tokens_500', '500'),
                'api_key': azure_section.get('api_key', ''),
                'azure_tenant_id': azure_section.get('azure_tenant_id', ''),
                'azure_client_id': azure_section.get('azure_client_id', '')
            })
        
        # AWS info section - use exact key names from config.ini
        if config.has_section('aws_info'):
            aws_section = config['aws_info']
            _cached_config_params.update({
                'opensearch_endpoint': aws_section.get('opensearch_endpoint', ''),
                'index_name': aws_section.get('index_name', ''),
                'aws_region': aws_section.get('aws_region', 'us-east-1'),
                's3_bucket_name': aws_section.get('s3_bucket_name', ''),
                'azure_cert_file_name': aws_section.get('azure_cert_file_name', '')
            })
        
        # OpenSearch section for local development
        if config.has_section('opensearch'):
            os_section = config['opensearch']
            _cached_config_params.update({
                'local_opensearch_endpoint': os_section.get('endpoint', 'http://localhost:9200'),
                'local_index_name': os_section.get('index_name', 'khub-opensearch-index')
            })
        
        logger.info(f"Cached {len(_cached_config_params)} config parameters from config.ini")
        return _cached_config_params
        
    except Exception as e:
        logger.warning(f"Could not load config.ini parameters: {e}")
        return {}


def get_config_param(key: str, default=None):
    """Get a cached config parameter by exact key name from config.ini."""
    params = _load_config_once()
    return params.get(key, default)