# src/infra/azure_auth.py
"""Azure certificate-based authentication for JPMC environment.

Provides enterprise-grade token management with automatic refresh,
background threading, and S3 certificate retrieval.
"""

import logging
import time
import threading
from functools import lru_cache
from typing import Optional
from azure.identity import CertificateCredential
from azure.core.exceptions import ClientAuthenticationError
import boto3
import os

logger = logging.getLogger(__name__)


class AzureTokenManager:
    """Thread-safe Azure token manager with automatic refresh."""
    
    TOKEN_VALIDITY_PERIOD = 58 * 60  # 58 minutes
    REFRESH_INTERVAL = 50 * 60       # 50 minutes
    
    def __init__(self):
        self.access_token: Optional[str] = None
        self.expiry_time: Optional[float] = None
        self._lock = threading.Lock()
        
    def get_token(self) -> str:
        """Get current token, refreshing if necessary."""
        current_time = time.time()
        
        with self._lock:
            if self.access_token is None or current_time >= (self.expiry_time or 0):
                self._refresh_token()
            return self.access_token or ""
    
    def _refresh_token(self):
        """Internal token refresh method."""
        logger.info("Refreshing Azure token...")
        try:
            self.access_token = _get_azure_access_token()
            self.expiry_time = time.time() + self.TOKEN_VALIDITY_PERIOD
            logger.info(f"Token refreshed. Expires at: {time.ctime(self.expiry_time)}")
        except Exception as e:
            logger.error(f"Failed to refresh token: {e}")
            raise
    
    def start_background_refresh(self):
        """Start background token refresh thread."""
        def refresh_task():
            while True:
                try:
                    time.sleep(self.REFRESH_INTERVAL)
                    current_time = time.time()
                    if self.expiry_time and current_time >= self.expiry_time - 300:  # 5 min buffer
                        logger.info("Background refresh: refreshing token...")
                        with self._lock:
                            self._refresh_token()
                except Exception as e:
                    logger.error(f"Background refresh failed: {e}")
        
        thread = threading.Thread(target=refresh_task, daemon=True)
        thread.start()
        logger.info("Background token refresh started")


def _get_azure_access_token() -> str:
    """Obtain Azure access token using certificate credentials."""
    logger.info("Obtaining Azure access token via certificate")
    
    # Try environment variables first
    tenant_id = os.getenv("AZURE_TENANT_ID")
    client_id = os.getenv("AZURE_CLIENT_ID") 
    scope = os.getenv("AZURE_SCOPE", "https://cognitiveservices.azure.com/.default")
    bucket_name = os.getenv("S3_BUCKET_NAME")
    cert_file_name = os.getenv("AZURE_CERT_FILE_NAME", "UtilitiesAssist.pem")
    
    # Fallback to centralized settings if environment variables not available
    if not tenant_id or not client_id:
        try:
            from src.infra.settings import get_settings
            settings = get_settings()
            
            if settings.azure_openai:
                tenant_id = tenant_id or settings.azure_openai.azure_tenant_id
                client_id = client_id or settings.azure_openai.azure_client_id
                scope = scope or settings.azure_openai.scope
                
            if settings.aws_info:
                bucket_name = bucket_name or settings.aws_info.s3_bucket_name
                cert_file_name = cert_file_name or settings.aws_info.azure_cert_file_name
                
            logger.info("Using Azure configuration from centralized settings")
        except Exception as e:
            logger.warning(f"Failed to load centralized settings: {e}")
            
            # Final fallback to legacy config.ini
            try:
                from utils import load_config
                config = load_config()
                tenant_id = tenant_id or config.get('azure_openai', 'azure_tenant_id', fallback=None)
                client_id = client_id or config.get('azure_openai', 'azure_client_id', fallback=None)
                scope = scope or config.get('azure_openai', 'scope', fallback="https://cognitiveservices.azure.com/.default")
                bucket_name = bucket_name or config.get('aws_info', 's3_bucket_name', fallback=None)
                cert_file_name = cert_file_name or config.get('aws_info', 'azure_cert_file_name', fallback="UtilitiesAssist.pem")
                logger.info("Using Azure configuration from legacy config.ini")
            except Exception as e2:
                logger.warning(f"Failed to load legacy config.ini: {e2}")
    
    if not all([tenant_id, client_id, bucket_name]):
        raise ValueError("Missing required Azure configuration: AZURE_TENANT_ID, AZURE_CLIENT_ID, S3_BUCKET_NAME")
    
    try:
        # Load certificate from S3
        pem_content = _load_certificate_from_s3(bucket_name, cert_file_name)
        
        # Create certificate credential
        credential = CertificateCredential(
            tenant_id=tenant_id,
            client_id=client_id, 
            certificate_data=pem_content
        )
        
        # Get access token
        token_result = credential.get_token(scope)
        logger.info("Azure access token obtained successfully")
        return token_result.token
        
    except ClientAuthenticationError as e:
        logger.error(f"Azure authentication failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error getting Azure token: {e}")
        raise


def _load_certificate_from_s3(bucket_name: str, file_name: str) -> bytes:
    """Load certificate from S3 bucket."""
    try:
        s3_client = boto3.client('s3')
        response = s3_client.get_object(Bucket=bucket_name, Key=file_name)
        return response['Body'].read()
    except Exception as e:
        logger.error(f"Failed to load certificate from S3: {e}")
        raise


# Optional singleton token manager - can be disabled if tokens change frequently
@lru_cache(maxsize=1)  # Single manager instance only
def get_azure_token_manager() -> AzureTokenManager:
    """Get singleton Azure token manager."""
    manager = AzureTokenManager()
    # Pre-fetch token to reduce first-request latency
    try:
        manager.get_token()
        manager.start_background_refresh()
    except Exception as e:
        logger.warning(f"Initial token fetch failed: {e}")
    return manager


def azure_token_provider() -> str:
    """Token provider function for client factory."""
    return get_azure_token_manager().get_token()