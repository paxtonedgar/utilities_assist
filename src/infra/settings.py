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

logger = logging.getLogger(__name__)


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
        default=False,
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
        # Load config.ini data first
        config_ini_data = self._load_config_ini()
        
        # Merge with kwargs, giving kwargs precedence
        merged_data = {**config_ini_data, **kwargs}
        
        super().__init__(**merged_data)
        
        # Log effective configuration
        self._log_effective_config()
    
    def _load_config_ini(self) -> Dict[str, Any]:
        """Load configuration from config.ini using existing system."""
        try:
            config_file = os.getenv('UTILITIES_CONFIG', 'config.local.ini')
            
            # Use existing file resolution logic
            if not os.path.isabs(config_file):
                if os.path.exists(config_file):
                    file_path = config_file
                elif os.path.exists(f'src/{config_file}'):
                    file_path = f'src/{config_file}'
                else:
                    file_path = config_file
            else:
                file_path = config_file
            
            config = configparser.ConfigParser()
            config.read(file_path)
            
            # Convert config sections to nested dict
            config_data = {}
            
            # Azure OpenAI section
            if config.has_section('azure_openai'):
                section = config['azure_openai']
                config_data['azure_openai'] = AzureOpenAIConfig(
                    azure_tenant_id=section.get('azure_tenant_id', ''),
                    azure_client_id=section.get('azure_client_id', ''),
                    scope=section.get('scope', 'https://cognitiveservices.azure.com/.default'),
                    azure_openai_endpoint=section.get('azure_openai_endpoint', ''),
                    azure_openai_embedding_model=section.get('azure_openai_embedding_model', 'text-embedding-3-small'),
                    api_key=section.get('api_key', ''),
                    deployment_name=section.get('deployment_name', ''),
                    api_version=section.get('api_version', '2024-10-21'),
                    max_tokens_2k=section.getint('max_tokens_2k', 2000),
                    max_tokens_500=section.getint('max_tokens_500', 500),
                    temperature=section.getfloat('temperature', 0.1),
                    openai_api_type=section.get('openai_api_type', 'azure')
                )
            
            # AWS section  
            if config.has_section('aws_info'):
                section = config['aws_info']
                config_data['aws_info'] = AWSConfig(
                    aws_region=section.get('aws_region', 'us-east-1'),
                    s3_bucket_name=section.get('s3_bucket_name', None),
                    azure_cert_file_name=section.get('azure_cert_file_name', 'UtilitiesAssist.pem'),
                    osa_file_name=section.get('osa_file_name', None),
                    opensearch_endpoint=section.get('opensearch_endpoint', None),
                    index_name=section.get('index_name', None)
                )
            
            # OpenSearch section
            if config.has_section('opensearch'):
                section = config['opensearch']
                config_data['opensearch'] = OpenSearchConfig(
                    endpoint=section.get('endpoint', 'http://localhost:9200'),
                    index_name=section.get('index_name', 'confluence_current')
                )
            
            # File paths section
            if config.has_section('file_paths'):
                section = config['file_paths']
                config_data.update({
                    'synonyms_file_path': section.get('synonyms_file_path', 'data/synonyms.json'),
                    'api_file_path': section.get('api_file_path', 'data/api_description.json'),
                    'questions_intent_path': section.get('questions_intent', 'data/questions_intent.csv')
                })
            
            # Confluence section
            if config.has_section('confluence_info'):
                section = config['confluence_info']
                config_data['confluence_directory'] = section.get('directory', 'data')
            
            logger.debug(f"Loaded config.ini from: {file_path}")
            return config_data
            
        except Exception as e:
            logger.warning(f"Failed to load config.ini: {e}")
            return {}
    
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
    
    @property
    def chat(self):
        """Compatibility property for LangGraph nodes expecting settings.chat.*"""
        if not self.azure_openai:
            return None
        
        class ChatConfig:
            def __init__(self, azure_config):
                self.api_version = azure_config.api_version
                self.model = azure_config.deployment_name
                self.api_base = azure_config.azure_openai_endpoint
                self.api_key = azure_config.api_key
                self.temperature = azure_config.temperature
                self.max_tokens_2k = azure_config.max_tokens_2k
                self.max_tokens_500 = azure_config.max_tokens_500
        
        return ChatConfig(self.azure_openai)
    
    @property 
    def embed(self):
        """Compatibility property for LangGraph nodes expecting settings.embed.*"""
        if not self.azure_openai:
            return None
        
        class EmbedConfig:
            def __init__(self, azure_config):
                self.api_version = azure_config.api_version
                self.model = azure_config.azure_openai_embedding_model
                self.api_base = azure_config.azure_openai_endpoint
                self.api_key = azure_config.api_key
        
        return EmbedConfig(self.azure_openai)


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