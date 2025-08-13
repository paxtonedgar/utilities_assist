.PHONY: install test run clean run-local test-local index-local embed-local start-opensearch stop-opensearch eval eval-no-embed test-eval clean-eval

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/

run:
	UTILITIES_CONFIG=src/config.ini CLOUD_PROFILE=jpmc_azure streamlit run streamlit_app.py

# Local development targets
run-local:
	UTILITIES_CONFIG=config.local.ini USE_MOCK_SEARCH=true USE_LOCAL_AZURE=true streamlit run streamlit_app.py

test-local:
	UTILITIES_CONFIG=config.local.ini USE_MOCK_SEARCH=true USE_LOCAL_AZURE=true python -m pytest -v --maxfail=1 --disable-warnings

# OpenSearch container management
start-opensearch:
	./scripts/start_opensearch_local.sh

stop-opensearch:
	./scripts/stop_opensearch_local.sh

# Index mock documents into local OpenSearch
index-local:
	python scripts/index_mock_docs.py

# Generate embeddings and create FAISS index
embed-local:
	UTILITIES_CONFIG=config.local.ini USE_LOCAL_AZURE=true python scripts/embed_mock_docs.py

# Test vector search
test-vector:
	python scripts/test_vector_search.py

# Run with only mock search (useful for testing without Azure OpenAI)
run-mock:
	UTILITIES_CONFIG=config.local.ini USE_MOCK_SEARCH=true streamlit run streamlit_app.py

# Run demo UI (standalone, no backend dependencies)
run-demo:
	streamlit run demo_ui.py

# Complete local setup workflow
setup-local: install start-opensearch
	@echo "⏳ Waiting for OpenSearch to be ready..."
	@sleep 5
	@$(MAKE) index-local
	@echo "✅ Local OpenSearch setup complete!"
	@echo "💡 Run 'make embed-local' to create FAISS index (requires Azure OpenAI config)"
	@echo "💡 Run 'make run-local' to start the application"

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Clean all local data (dangerous!)
clean-local:
	@echo "🚨 This will remove all local data and containers!"
	@read -p "Are you sure? [y/N] " -n 1 -r; echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker stop opensearch-local 2>/dev/null || true; \
		docker rm opensearch-local 2>/dev/null || true; \
		rm -f data/mock_faiss.index data/mock_faiss_metadata.json data/mock_embeddings.npy; \
		echo "✅ Local data cleaned"; \
	else \
		echo "❌ Cancelled"; \
	fi

# Evaluation targets
eval:
	@echo "🚀 Running full evaluation with embeddings..."
	python eval/run_eval.py --corpus eval/mock_corpus/utilities_docs.jsonl --golden-set eval/golden_set.yaml --output eval/latest.json

eval-no-embed:
	@echo "🚀 Running BM25-only evaluation..."
	python eval/run_eval.py --corpus eval/mock_corpus/utilities_docs.jsonl --golden-set eval/golden_set.yaml --output eval/latest_bm25.json --no-embeddings

test-eval:
	@echo "🧪 Running test evaluation..."
	python eval/run_eval.py --corpus eval/mock_corpus/utilities_docs.jsonl --golden-set eval/golden_set.yaml --output eval/test.json --verbose

clean-eval:
	@echo "🧹 Cleaning evaluation artifacts..."
	rm -f eval/latest.json eval/latest_bm25.json eval/test.json
	@echo "✅ Cleaned evaluation files"

check-opensearch:
	@echo "🔍 Checking OpenSearch connection..."
	@curl -s http://localhost:9200 > /dev/null && echo "✅ OpenSearch is running" || echo "❌ OpenSearch is not running - please start it first"

# Show help
help:
	@echo "🚀 Utilities Assist - Development Commands"
	@echo ""
	@echo "📦 Installation:"
	@echo "  make install          Install Python dependencies"
	@echo ""
	@echo "🏃 Running:"
	@echo "  make run              Run production version"
	@echo "  make run-local        Run with local mocks + Azure"
	@echo "  make run-mock         Run with mocks only (no Azure needed)"
	@echo "  make run-demo         Run demo UI (no backend dependencies)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test             Run production tests"
	@echo "  make test-local       Run tests in local development mode"
	@echo "  make test-vector      Test FAISS vector search"
	@echo ""
	@echo "🔍 OpenSearch:"
	@echo "  make start-opensearch Start OpenSearch container"
	@echo "  make stop-opensearch  Stop OpenSearch container"
	@echo "  make index-local      Index mock docs into OpenSearch"
	@echo ""
	@echo "🤖 AI/Embeddings:"
	@echo "  make embed-local      Generate embeddings + FAISS index"
	@echo ""
	@echo "⚡ Quick Setup:"
	@echo "  make setup-local      Complete OpenSearch setup"
	@echo ""
	@echo "📊 Evaluation:"
	@echo "  make eval             Run full evaluation with embeddings"
	@echo "  make eval-no-embed    Run BM25-only evaluation"
	@echo "  make test-eval        Quick evaluation test"
	@echo "  make clean-eval       Clean evaluation files"
	@echo "  make check-opensearch Check if OpenSearch is running"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean            Clean Python cache files"
	@echo "  make clean-local      Clean all local data (dangerous!)"