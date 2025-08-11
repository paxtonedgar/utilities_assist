# Scripts Directory

This directory contains scripts for local development and testing of the utilities assist application.

## OpenSearch Management

### `start_opensearch_local.sh`
Starts a local OpenSearch container for testing.
```bash
./scripts/start_opensearch_local.sh
# Or use: make start-opensearch
```

Features:
- Runs OpenSearch 2.11.0 in Docker
- Disables security for local development  
- Checks port availability
- Waits for cluster to be ready
- Provides helpful usage information

### `stop_opensearch_local.sh`
Stops (and optionally removes) the OpenSearch container.
```bash
./scripts/stop_opensearch_local.sh          # Stop container
./scripts/stop_opensearch_local.sh --remove # Stop and remove
# Or use: make stop-opensearch
```

## Document Indexing

### `index_mock_docs.py`
Indexes mock documents into local OpenSearch instance.
```bash
python scripts/index_mock_docs.py
# Or use: make index-local
```

Features:
- Reads `data/mock_docs/*.json` files
- Creates `mock_confluence` index with BM25 mapping
- Ignores 400 errors for existing index
- Logs each document ID as it indexes
- Verifies indexing with search test

## Embeddings & Vector Search  

### `embed_mock_docs.py`
Generates embeddings using Azure OpenAI and creates FAISS index.
```bash
UTILITIES_CONFIG=config.local.ini python scripts/embed_mock_docs.py
# Or use: make embed-local
```

Requirements:
- Configured Azure OpenAI in `config.local.ini`
- `faiss-cpu` package installed

Output files:
- `data/mock_faiss.index` - FAISS index file
- `data/mock_faiss_metadata.json` - Document metadata
- `data/mock_embeddings.npy` - Raw embeddings array
- `data/embedding_summary.json` - Generation summary

### `test_vector_search.py`
Tests FAISS vector search functionality.
```bash
python scripts/test_vector_search.py
# Or use: make test-vector
```

Tests:
- FAISS index loading
- Azure OpenAI embedding generation
- Vector similarity search
- Multiple test queries

## Usage Workflow

### Complete Setup
```bash
# 1. Install dependencies
make install

# 2. Start OpenSearch and index documents  
make setup-local

# 3. Generate embeddings (requires Azure OpenAI config)
make embed-local

# 4. Test vector search
make test-vector

# 5. Run application
make run-local
```

### Quick Testing (No Azure OpenAI needed)
```bash
make install
make run-mock  # Uses BM25 mock search only
```

### Development Cycle
```bash
# Start services
make start-opensearch

# Make changes to mock documents
python generate_mock_docs.py

# Re-index documents
make index-local

# Re-generate embeddings (if changed)  
make embed-local

# Test changes
make run-local
```

## Environment Variables

Scripts respect these environment variables:

- `UTILITIES_CONFIG` - Config file to use (default: config.local.ini)
- `USE_MOCK_SEARCH` - Use mock search instead of OpenSearch (true/false)
- `USE_LOCAL_AZURE` - Use local Azure config instead of enterprise (true/false)

## Error Handling

Scripts include comprehensive error handling:

- **Connection errors**: Clear messages about starting OpenSearch
- **Missing files**: Instructions on generating mock documents  
- **Configuration errors**: Guidance on Azure OpenAI setup
- **Dependency errors**: Install instructions for missing packages

## Docker Requirements

OpenSearch scripts require Docker to be installed and running:
- Docker Desktop (Mac/Windows)
- Docker Engine (Linux)
- Minimum 4GB RAM allocated to Docker