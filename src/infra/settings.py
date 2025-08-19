# src/infra/settings.py
"""
Centralized configuration management using pydantic-settings.

Loads configuration with proper priority:
1. Environment variables (highest priority)
2. .env file 
3. config.ini file (current system - lowest priority)

Profile-aware configuration for different environments (local, jpmc_azure, dev).
Uses aliases instead of hardcoded indices for production flexibility.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import configparser
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_shared_config():
    """Shared configuration loading utility with caching.
    
    Loads config.ini once and caches the result to eliminate repeated file I/O
    across different modules (config.py, azure_auth.py, resource_manager.py).
    
    Returns:
        configparser.ConfigParser: Loaded configuration or empty config on error
    """
    try:
        from utils import load_config
        config = load_config()
        logger.debug("Loaded shared config.ini successfully")
        return config
    except Exception as e:
        logger.warning(f"Failed to load shared config.ini: {e}")
        # Return empty config to prevent crashes
        return configparser.ConfigParser()


def get_config_value(section: str, key: str, fallback=None):
    """Get a configuration value from the shared config.
    
    Args:
        section: Config section name (e.g., 'azure_openai')
        key: Config key name
        fallback: Default value if not found
        
    Returns:
        Configuration value or fallback
    """
    config = _load_shared_config()
    return config.get(section, key, fallback=fallback)


class AzureOpenAIConfig(BaseModel):
    """Azure OpenAI configuration with exact config.ini field mappings."""
    azure_tenant_id: str = ""
    azure_client_id: str = ""
    scope: str = "https://cognitiveservices.azure.com/.default"
    azure_openai_endpoint: str = ""
    azure_openai_embedding_model: str = "text-embedding-3-small"
    api_key: str = ""
    deployment_name: str = ""
    api_version: str = "2024-10-21"
    max_tokens_2k: int = 2000
    max_tokens_500: int = 500
    temperature: float = 0.1
    openai_api_type: str = "azure"


class AWSConfig(BaseModel):
    """AWS configuration for OpenSearch and S3."""
    aws_region: str = "us-east-1"
    s3_bucket_name: Optional[str] = None
    azure_cert_file_name: str = "UtilitiesAssist.pem"
    osa_file_name: Optional[str] = None
    opensearch_endpoint: Optional[str] = None
    index_name: Optional[str] = None


class OpenSearchConfig(BaseModel):
    """Local OpenSearch configuration."""
    endpoint: str = "http://localhost:9200"
    index_name: str = "confluence_current"


class ApplicationSettings(BaseSettings):
    """Main application settings with profile-aware configuration."""
    
    model_config = SettingsConfigDict(
        env_file=('.env', '.env.local'),
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'
    )
    
    # Profile configuration with environment override
    cloud_profile: str = Field(
        default="local",
        validation_alias="CLOUD_PROFILE", 
        description="Cloud profile (local, jpmc_azure, dev)"
    )
    
    # Feature flags with environment override
    enable_knn: bool = Field(
        default=True,
        validation_alias="ENABLE_KNN",
        description="Enable k-NN vector search"
    )
    
    enable_langgraph_persistence: bool = Field(
        default=True,
        validation_alias="ENABLE_LANGGRAPH_PERSISTENCE", 
        description="Enable LangGraph conversation persistence"
    )
    
    # Logging configuration
    log_level: str = Field(
        default="INFO",
        validation_alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    
    enable_structured_logging: bool = Field(
        default=True,
        validation_alias="ENABLE_STRUCTURED_LOGGING",
        description="Enable structured JSON logging"  
    )
    
    # Production verbosity control
    reduce_log_verbosity: bool = Field(
        default=True,  # Enable by default to reduce log spam
        validation_alias="REDUCE_LOG_VERBOSITY",
        description="Reduce log verbosity for production environments"
    )
    
    # Configuration sections (loaded from config.ini)
    azure_openai: Optional[AzureOpenAIConfig] = None
    aws_info: Optional[AWSConfig] = None
    opensearch: Optional[OpenSearchConfig] = None
    
    # File paths
    synonyms_file_path: str = "data/synonyms.json"
    api_file_path: str = "data/api_description.json"
    questions_intent_path: str = "data/questions_intent.csv"
    
    # Confluence
    confluence_directory: str = "data"
    
    def __init__(self, **kwargs):
        # Load config.ini data using simplified approach
        config_ini_data = self._load_config_ini_simplified()
        
        # Merge with kwargs, giving kwargs precedence
        merged_data = {**config_ini_data, **kwargs}
        
        super().__init__(**merged_data)
        
        # Log effective configuration
        self._log_effective_config()
    
    def _load_config_ini_simplified(self) -> Dict[str, Any]:
        """Simplified config.ini loading using pydantic patterns."""
        try:
            config_file = os.getenv('UTILITIES_CONFIG', 'config.local.ini')
            config_path = self._resolve_config_path(config_file)
            
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return {}
            
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Use pydantic model validation for cleaner loading
            config_data = {}
            
            # Load sections using model defaults and validation
            for section_name, model_class in [
                ('azure_openai', AzureOpenAIConfig),
                ('aws_info', AWSConfig), 
                ('opensearch', OpenSearchConfig)
            ]:
                if config.has_section(section_name):
                    section_data = dict(config[section_name])
                    config_data[section_name] = model_class(**section_data)
            
            # Load simple settings
            self._load_simple_settings(config, config_data)
            
            logger.debug(f"Loaded config from: {config_path}")
            return config_data
            
        except Exception as e:
            logger.warning(f"Failed to load config.ini: {e}")
            return {}
    
    def _resolve_config_path(self, config_file: str) -> Path:
        """Resolve config file path using standard patterns."""
        if os.path.isabs(config_file):
            return Path(config_file)
        
        # Try current dir, then src/ directory
        candidates = [Path(config_file), Path('src') / config_file]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return Path(config_file)  # Return original for error handling
    
    def _load_simple_settings(self, config: configparser.ConfigParser, config_data: dict):
        """Load simple non-nested settings from config sections."""
        if config.has_section('file_paths'):
            section = config['file_paths']
            config_data.update({
                'synonyms_file_path': section.get('synonyms_file_path', 'data/synonyms.json'),
                'api_file_path': section.get('api_file_path', 'data/api_description.json'),
                'questions_intent_path': section.get('questions_intent', 'data/questions_intent.csv')
            })
        
        if config.has_section('confluence_info'):
            section = config['confluence_info']
            config_data['confluence_directory'] = section.get('directory', 'data')
    
    @property
    def search_index_alias(self) -> str:
        """Get the appropriate search index alias for the current profile."""
        profile = self.cloud_profile.lower()
        
        if profile == "jpmc_azure":
            # Production JPMC uses configured index from AWS section or fallback to alias
            if self.aws_info and self.aws_info.index_name:
                return self.aws_info.index_name
            return "confluence_current"  # Production alias
        
        elif profile in ["local", "dev"]:
            # Local development uses OpenSearch section or fallback  
            if self.opensearch and self.opensearch.index_name:
                return self.opensearch.index_name
            return "confluence_current"  # Local alias
        
        else:
            # Unknown profile - use safe default alias
            return "confluence_current"
    
    @property  
    def opensearch_host(self) -> str:
        """Get the appropriate OpenSearch host for the current profile."""
        profile = self.cloud_profile.lower()
        
        if profile == "jpmc_azure":
            # Production JPMC uses AWS OpenSearch
            if self.aws_info and self.aws_info.opensearch_endpoint:
                return self.aws_info.opensearch_endpoint
            return "https://utilitiesassist.dev.aws.jpmchase.net"  # JPMC default
        
        elif profile in ["local", "dev"]:
            # Local development uses local OpenSearch
            if self.opensearch:
                return self.opensearch.endpoint
            return "http://localhost:9200"  # Local default
            
        else:
            # Unknown profile - use local default
            return "http://localhost:9200"
    
    @property
    def embedding_index_alias(self) -> str:
        """Get the embedding index alias (could be same or different from main index)."""
        # For now, same as search index. Can be made configurable later.
        return self.search_index_alias
    
    @property
    def requires_aws_auth(self) -> bool:
        """Check if AWS authentication is required for the current profile."""
        return self.cloud_profile.lower() == "jpmc_azure"
    
    @property
    def requires_azure_auth(self) -> bool:
        """Check if Azure authentication is required for the current profile."""
        return self.cloud_profile.lower() == "jpmc_azure" and self.azure_openai is not None
    
    def _log_effective_config(self):
        """Log the effective configuration without secrets."""
        config_summary = {
            "cloud_profile": self.cloud_profile,
            "search_index_alias": self.search_index_alias, 
            "opensearch_host": self._mask_url(self.opensearch_host),
            "requires_aws_auth": self.requires_aws_auth,
            "requires_azure_auth": self.requires_azure_auth,
            "enable_knn": self.enable_knn,
            "enable_langgraph_persistence": self.enable_langgraph_persistence,
            "log_level": self.log_level,
            "enable_structured_logging": self.enable_structured_logging,
            "sections_loaded": {
                "azure_openai": self.azure_openai is not None,
                "aws_info": self.aws_info is not None,
                "opensearch": self.opensearch is not None
            }
        }
        
        print("✅ Loaded configs successfully")
        logger.info(f"Effective configuration: {config_summary}")
    
    def _mask_url(self, url: str) -> str:
        """Mask sensitive parts of URLs for logging."""
        if not url:
            return url
        # Simple masking - could be enhanced  
        if "localhost" in url or "127.0.0.1" in url:
            return url  # Local URLs are safe
        else:
            # Mask domain but keep protocol and path structure
            parts = url.split("/")
            if len(parts) >= 3:
                parts[2] = "***masked***"
                return "/".join(parts)
        return url
    
    def _create_config_wrapper(self, config_type: str):
        """Create configuration wrapper with shared logic.
        
        Args:
            config_type: Type of config ("chat", "embed", "search")
            
        Returns:
            Configuration wrapper object or None if azure_openai not available
        """
        if not self.azure_openai:
            return None
        
        class ConfigWrapper:
            def __init__(self, azure_config, settings, config_type):
                # Common attributes for all config types
                self.api_version = azure_config.api_version
                self.api_base = azure_config.azure_openai_endpoint
                self.api_key = azure_config.api_key
                
                # Type-specific attributes
                if config_type == "chat":
                    self.model = azure_config.deployment_name
                    self.temperature = azure_config.temperature
                    self.max_tokens_2k = azure_config.max_tokens_2k
                    self.max_tokens_500 = azure_config.max_tokens_500
                elif config_type == "embed":
                    self.model = azure_config.azure_openai_embedding_model
                elif config_type == "search":
                    self.model = azure_config.deployment_name
                    self.index_alias = settings.search_index_alias
                    self.opensearch_host = settings.opensearch_host
        
        return ConfigWrapper(self.azure_openai, self, config_type)
    
    @property
    def chat(self):
        """Compatibility property for LangGraph nodes expecting settings.chat.*"""
        return self._create_config_wrapper("chat")
    
    @property 
    def embed(self):
        """Compatibility property for LangGraph nodes expecting settings.embed.*"""
        return self._create_config_wrapper("embed")
    
    @property
    def search(self):
        """Compatibility property for graph search configuration."""
        return self._create_config_wrapper("search")


# Singleton instance
_settings: Optional[ApplicationSettings] = None


def get_settings() -> ApplicationSettings:
    """Get singleton application settings instance."""
    global _settings
    if _settings is None:
        _settings = ApplicationSettings()
    return _settings


def refresh_settings() -> ApplicationSettings:
    """Force refresh of application settings."""
    global _settings
    _settings = None
    return get_settings()