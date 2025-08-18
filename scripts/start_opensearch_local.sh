#!/bin/bash

# OpenSearch Local Development Setup Script
# This script starts OpenSearch in a Docker container for local development

set -e

echo "Starting OpenSearch local development environment..."

# Configuration
OPENSEARCH_VERSION="2.11.1"
CONTAINER_NAME="opensearch-local"
PORT="9200"
DASHBOARD_PORT="5601"
NETWORK_NAME="opensearch-net"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create Docker network if it doesn't exist
if ! docker network ls | grep -q "$NETWORK_NAME"; then
    echo "Creating Docker network: $NETWORK_NAME"
    docker network create "$NETWORK_NAME"
fi

# Stop and remove existing container if it exists
if docker ps -a --format 'table {{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "Stopping and removing existing OpenSearch container..."
    docker stop "$CONTAINER_NAME" || true
    docker rm "$CONTAINER_NAME" || true
fi

# Start OpenSearch container
echo "Starting OpenSearch container..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --network "$NETWORK_NAME" \
    -p "$PORT:9200" \
    -p "9600:9600" \
    -e "discovery.type=single-node" \
    -e "OPENSEARCH_INITIAL_ADMIN_PASSWORD=admin123!" \
    -e "DISABLE_SECURITY_PLUGIN=true" \
    -e "bootstrap.memory_lock=true" \
    -e "OPENSEARCH_JAVA_OPTS=-Xms1g -Xmx1g" \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=65536:65536 \
    opensearchproject/opensearch:"$OPENSEARCH_VERSION"

# Wait for OpenSearch to be ready
echo "Waiting for OpenSearch to start..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s "http://localhost:$PORT" > /dev/null 2>&1; then
        echo "OpenSearch is ready!"
        break
    fi
    
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts: Waiting for OpenSearch..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "Error: OpenSearch failed to start within expected time"
    echo "Check container logs with: docker logs $CONTAINER_NAME"
    exit 1
fi

# Create indices with proper mappings
echo "Setting up indices..."

# Create main index
echo "Creating main index (khub-opensearch-index)..."
curl -X PUT "http://localhost:$PORT/khub-opensearch-index" \
    -H 'Content-Type: application/json' \
    -d @src/search/mappings/confluence_v2.json || echo "Main index may already exist"

# Create swagger index with same mapping structure
echo "Creating swagger index (khub-opensearch-swagger-index)..."
curl -X PUT "http://localhost:$PORT/khub-opensearch-swagger-index" \
    -H 'Content-Type: application/json' \
    -d @src/search/mappings/confluence_v2.json || echo "Swagger index may already exist"

# Verify indices
echo "Verifying indices..."
curl -s "http://localhost:$PORT/_cat/indices?v"

echo ""
echo "✅ OpenSearch local development environment is ready!"
echo "📍 OpenSearch URL: http://localhost:$PORT"
echo "🔍 Health check: curl http://localhost:$PORT/_cluster/health"
echo "📋 Indices: curl http://localhost:$PORT/_cat/indices?v"
echo ""
echo "To stop OpenSearch: docker stop $CONTAINER_NAME"
echo "To view logs: docker logs $CONTAINER_NAME"
echo "To remove completely: docker stop $CONTAINER_NAME && docker rm $CONTAINER_NAME"