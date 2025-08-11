#!/bin/bash
# stop_opensearch_local.sh - Stop and optionally remove OpenSearch container

set -e

CONTAINER_NAME="opensearch-local"
REMOVE_CONTAINER=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --remove|-r)
            REMOVE_CONTAINER=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--remove|-r] [--help|-h]"
            echo ""
            echo "Options:"
            echo "  --remove, -r    Remove the container after stopping"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🛑 Stopping OpenSearch container..."

# Check if container exists
if ! docker ps -a --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "❌ Container ${CONTAINER_NAME} does not exist."
    exit 1
fi

# Check if container is running
if docker ps --format 'table {{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🔄 Stopping container ${CONTAINER_NAME}..."
    docker stop ${CONTAINER_NAME}
    echo "✅ Container stopped successfully."
else
    echo "ℹ️  Container ${CONTAINER_NAME} is already stopped."
fi

# Remove container if requested
if [ "$REMOVE_CONTAINER" = true ]; then
    echo "🗑️  Removing container ${CONTAINER_NAME}..."
    docker rm ${CONTAINER_NAME}
    echo "✅ Container removed successfully."
    echo ""
    echo "🎉 OpenSearch container has been completely removed."
    echo "   Run ./scripts/start_opensearch_local.sh to create a fresh container."
else
    echo ""
    echo "✅ OpenSearch container has been stopped."
    echo "   Run ./scripts/start_opensearch_local.sh to restart it."
    echo "   Use --remove flag to completely remove the container."
fi