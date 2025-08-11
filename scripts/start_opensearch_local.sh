#!/bin/bash
# start_opensearch_local.sh - Start OpenSearch container for local development

set -e

CONTAINER_NAME="opensearch-local"
OPENSEARCH_PORT=9200
PERFORMANCE_PORT=9600
OPENSEARCH_VERSION="2.11.0"

echo "🚀 Starting OpenSearch container for local development..."

# Check if container already exists
if docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "📦 Container ${CONTAINER_NAME} already exists."
    
    # Check if it's running
    if docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo "✅ Container is already running."
        echo "🌐 OpenSearch available at: http://localhost:${OPENSEARCH_PORT}"
        exit 0
    else
        echo "🔄 Starting existing container..."
        docker start ${CONTAINER_NAME}
        echo "✅ Container started successfully."
        echo "🌐 OpenSearch available at: http://localhost:${OPENSEARCH_PORT}"
        exit 0
    fi
fi

# Check if ports are available
if lsof -Pi :${OPENSEARCH_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ Port ${OPENSEARCH_PORT} is already in use. Please free the port and try again."
    exit 1
fi

if lsof -Pi :${PERFORMANCE_PORT} -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "❌ Port ${PERFORMANCE_PORT} is already in use. Please free the port and try again."
    exit 1
fi

# Start OpenSearch container
echo "📦 Creating and starting OpenSearch container..."
docker run -d \
    --name ${CONTAINER_NAME} \
    -p ${OPENSEARCH_PORT}:9200 \
    -p ${PERFORMANCE_PORT}:9600 \
    -e "discovery.type=single-node" \
    -e "plugins.security.disabled=true" \
    -e "OPENSEARCH_JAVA_OPTS=-Xms512m -Xmx512m" \
    opensearchproject/opensearch:${OPENSEARCH_VERSION}

# Wait for OpenSearch to be ready
echo "⏳ Waiting for OpenSearch to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:${OPENSEARCH_PORT}/_cluster/health >/dev/null 2>&1; then
        echo "✅ OpenSearch is ready!"
        break
    fi
    echo "   Attempt ${i}/30: waiting for OpenSearch to start..."
    sleep 2
done

# Verify OpenSearch is responding
if curl -s http://localhost:${OPENSEARCH_PORT}/_cluster/health >/dev/null 2>&1; then
    echo "🎉 OpenSearch container started successfully!"
    echo ""
    echo "📊 Container Information:"
    echo "   Name: ${CONTAINER_NAME}"
    echo "   Image: opensearchproject/opensearch:${OPENSEARCH_VERSION}"
    echo "   OpenSearch API: http://localhost:${OPENSEARCH_PORT}"
    echo "   Performance API: http://localhost:${PERFORMANCE_PORT}"
    echo ""
    echo "🔧 Useful Commands:"
    echo "   Check cluster health: curl http://localhost:${OPENSEARCH_PORT}/_cluster/health"
    echo "   List indices: curl http://localhost:${OPENSEARCH_PORT}/_cat/indices?v"
    echo "   View container logs: docker logs ${CONTAINER_NAME}"
    echo "   Stop container: docker stop ${CONTAINER_NAME}"
    echo "   Remove container: docker rm ${CONTAINER_NAME}"
else
    echo "❌ OpenSearch failed to start properly. Check container logs:"
    echo "   docker logs ${CONTAINER_NAME}"
    exit 1
fi