# token_manager.py - Token management module for Azure authentication
import logging
from azure.identity import CertificateCredential
from azure.core.exceptions import ClientAuthenticationError
import time
import threading

from utils import load_config, load_certificate_from_s3


class TokenManager:
    _instance = None  # Class-level attribute to hold the singleton instance
    TOKEN_VALIDITY_PERIOD = 58 * 60
    REFRESH_INTERVAL = 50 * 60
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TokenManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.access_token = None
            cls._instance.expiry_time = None
        return cls._instance
    
    def get_token(self):
        current_time = time.time()
        logging.debug(f"Checking token validity. Current time: {current_time}, Expiry time: {self.expiry_time}")
        if self.access_token is None or current_time >= self.expiry_time:
            self.refresh_token()
        return self.access_token
    
    def refresh_token(self):
        logging.info("Refreshing token...")
        try:
            self.access_token = get_access_token()
            self.expiry_time = time.time() + self.TOKEN_VALIDITY_PERIOD
            
            logging.info(
                f"Token refreshed successfully. Expires at: {self.expiry_time} (Readable: {time.ctime(self.expiry_time)})")
            # Notify ClientSingleton to refresh clients--Ankita
            from client_manager import ClientSingleton
            if ClientSingleton._instance:
                ClientSingleton._instance.refresh_clients()
        
        except Exception as e:
            logging.error(f"Failed to refresh token: {e}")
    
    def pre_fetch_token(self):
        """
        Pre-fetch the token during application startup to reduce latency for the first request.
        """
        logging.info("Starting pre-fetching of token...")
        try:
            self.get_token()  # This will fetch and cache the token if not already available.
            logging.info("Token pre-fetching completed successfully.")
        except Exception as e:
            logging.error(f"Token pre-fetching failed: {e}")
    
    def start_background_refresh(self):
        """
        Start a background thread to refresh the token periodically.
        """
        
        def refresh_task():
            # Calculate the initial delay based on the token's expiration time
            initial_delay = max(0, (self.expiry_time - time.time() - self.REFRESH_INTERVAL) if self.expiry_time else 0)
            if initial_delay > 0:
                logging.info(f"Delaying background task start by {initial_delay:.2f} seconds.")
                time.sleep(initial_delay)
            
            while True:
                try:
                    # Check if the token is close to expiration
                    current_time = time.time()
                    if self.expiry_time and current_time < self.expiry_time - self.REFRESH_INTERVAL:
                        logging.info("Token is still valid. Skipping refresh.")
                    else:
                        logging.info("Background task: Refreshing token...")
                        self.refresh_token()
                except Exception as e:
                    logging.error(f"Background task failed to refresh token: {e}")
                finally:
                    # Wait before the next refresh
                    time.sleep(self.REFRESH_INTERVAL)
        
        # Start the background thread
        thread = threading.Thread(target=refresh_task, daemon=True)
        thread.start()
        logging.info("Background token refresh task started.")


def get_access_token():
    """Obtain an access token using Azure credentials."""
    logging.info("Obtaining access token.")
    config = load_config()
    bucket_name = config['aws_info']['s3_bucket_name']
    file_name = config['aws_info']['azure_cert_file_name']
    pem_content = load_certificate_from_s3(bucket_name, file_name)
    try:
        credential = CertificateCredential(
            tenant_id=config['azure_openai']['azure_tenant_id'],
            client_id=config['azure_openai']['azure_client_id'],
            certificate_data=pem_content
        )
        access_token = credential.get_token(config['azure_openai']['scope']).token
        logging.info("Access token obtained successfully.")
        return access_token
    
    except ClientAuthenticationError as e:
        logging.error("Authentication failed: %s", e)
    except Exception as e:
        logging.error("An unexpected error occurred: %s", e)
    return None