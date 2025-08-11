# embedding_creation.py - Module for creating embeddings and vector stores
import logging
import aiohttp
from client_manager import ClientSingleton  # Import from client_manager
from tenacity import retry, wait_exponential, stop_after_attempt

from utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@retry(wait=wait_exponential(multiplier=1, min=4, max=5), stop=stop_after_attempt(10))
async def create_embeddings_with_retry(client, input_data, model):
    """Create embeddings with retry logic asynchronously."""
    logging.info("Creating embeddings with retry logic.")
    logging.debug(f"Embedding endpoint: {client.embeddings.endpoint}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                client.embeddings.endpoint,
                json={'input': input_data, 'model': model}
            ) as response:
                
                # Ensure it's JSON and handle error response
                if response.status != 200:
                    text = await response.text()
                    raise RuntimeError(f"Non-200 response: {response.status}, body: {text}")
                
                json_data = await response.json()
                if "data" not in json_data:
                    raise KeyError(f"'data' key missing in embedding response: {json_data}")
                
                return json_data
    
    except Exception as e:
        logging.error(f"Failed to create embeddings: {e}")
        raise


async def create_vector_store(chunks_with_metadata, token_manager):
    """Create a vector store using the text chunks and their embeddings asynchronously."""
    logging.info("Creating vector store.")
    
    # Get the singleton client instance
    client_singleton = ClientSingleton.get_instance(token_manager)
    client = client_singleton.embeddings_client
    
    embeddings = []
    batch_size = 5
    for i in range(0, len(chunks_with_metadata), batch_size):
        batch_chunks = [chunk['content'] for chunk in chunks_with_metadata[i:i + batch_size]]
        logging.info("Processing batch %d to %d", i, i + batch_size)
        
        try:
            config = load_config()
            # Ensure the client is updated with the latest token
            client_singleton.refresh_clients_if_needed(token_manager)
            embedding_data_list = await create_embeddings_with_retry(client, batch_chunks, config['azure_openai']['azure_openai_embedding_model'])
            for embedding_data in embedding_data_list:
                embeddings.append(embedding_data['embedding'])
            logging.info("Batch processed successfully.")
        except Exception as e:
            logging.error("Failed to create embeddings after retries: %s", e)
            return None
    
    if len(chunks_with_metadata) != len(embeddings):
        raise ValueError("Metadata and embedding must have same length")
    
    logging.info("Vector store created successfully.")
    return embeddings  # Return only the embeddings