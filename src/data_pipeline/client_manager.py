# client_manager.py - Client manager module for Azure OpenAI integration
import logging
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from utils import load_config
import boto3
from requests_awsauth import AWS4Auth

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ClientSingleton:
    _instance = None
    
    @staticmethod
    def get_instance(token_manager):
        """Static access method."""
        if ClientSingleton._instance is None:
            ClientSingleton(token_manager)
        return ClientSingleton._instance
    
    def __init__(self, token_manager):
        """Virtually private constructor."""
        if ClientSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            self.token_manager = token_manager
            self.refresh_clients()  # Initialize clients with the latest token
            ClientSingleton._instance = self
            logging.info(
                "AzureChatOpenAI and AzureOpenAIEmbeddings clients initialized and stored in singleton instance.")
    
    def refresh_clients(self):
        config = load_config()
        """Refresh the chat and embeddings clients with the latest token."""
        access_token = self.token_manager.get_token()
        self.chat_client = AzureChatOpenAI(
            azure_endpoint=config['azure_openai']['azure_openai_endpoint'],
            openai_api_version=config['azure_openai']['api_version'],
            deployment_name=config['azure_openai']['deployment_name'],
            openai_api_key=config['azure_openai']['api_key'],
            openai_api_type=config['azure_openai']['openai_api_type'],
            max_tokens=config['azure_openai']['max_tokens_500'],
            temperature=config['azure_openai']['temperature'],
            streaming=True,
            default_headers={
                "Authorization": f"Bearer {access_token}",
                "user_sid": "REPLACE"
            }
        )
        self.embeddings_client = AzureOpenAIEmbeddings(
            azure_endpoint=config['azure_openai']['azure_openai_endpoint'],
            openai_api_version=config['azure_openai']['api_version'],
            openai_api_key=config['azure_openai']['api_key'],
            model=config['azure_openai']['azure_openai_embedding_model'],
            openai_api_type=config['azure_openai']['openai_api_type'],
            default_headers={
                "Authorization": f"Bearer {access_token}",
                "user_sid": "REPLACE"
            }
        )
    
    def get_chat_client(self):
        """Return the chat client with the latest token."""
        # Ensure the client uses the latest token
        self.chat_client.default_headers["Authorization"] = f"Bearer {self.token_manager.get_token()}"
        return self.chat_client
    
    def get_embeddings_client(self):
        """Return the embeddings client with the latest token."""
        # Ensure the client uses the latest token
        self.embeddings_client.default_headers["Authorization"] = f"Bearer {self.token_manager.get_token()}"
        return self.embeddings_client
    
    @staticmethod
    def get_awsauth():
        session = boto3.Session()
        logging.info(session)
        region = 'us-east-1'
        credentials = session.get_credentials()
        logging.info(credentials)
        return AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)