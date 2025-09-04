.PHONY: install test run clean run-local test-local index-local embed-local start-opensearch stop-opensearch eval eval-no-embed test-eval clean-eval

install:
	pip install -r requirements.txt

test:
	python -m pytest tests/

run:
	UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure streamlit run streamlit_app.py

# Local development targets
run-local:
	UTILITIES_CONFIG=config.ini streamlit run streamlit_app.py

test-local:
	UTILITIES_CONFIG=config.ini python -m pytest -v --maxfail=1 --disable-warnings

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
	UTILITIES_CONFIG=config.ini python scripts/embed_mock_docs.py

# Test vector search
test-vector:
	python scripts/test_vector_search.py

# Run with local config only
run-mock:
	UTILITIES_CONFIG=config.ini streamlit run streamlit_app.py

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
	@echo "  make run-local        Run with local config"
	@echo "  make run-mock         Run with local config (no Azure needed)"
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

# Ontology probe (OpenSearch -> steps + domain edges -> scoring + QC)
.PHONY: ontology-probe check-settings

ontology-probe:
	@echo "🚀 Running ontology probe against OpenSearch (config.ini)"
	UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.pipeline --query "$(or $(QUERY),install OR configure OR team OR division OR application OR diagram)" --max-docs $(or $(MAX),50) --csv-out $(or $(CSV),qc_edges.csv)

# Full corpus scan (document-by-document via PIT)
.PHONY: ontology-scan
ontology-scan:
	@echo "📚 Scanning corpus document-by-document (PIT)"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.doc_by_doc --index "$(or $(INDEX),)" --limit $(or $(LIMIT),1000) --batch $(or $(BATCH),200) --out-dir $(or $(OUT),outputs/ontology_scan)

check-settings:
	@echo "🔧 Printing resolved OpenSearch settings from config.ini"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -c "from src.infra.settings import get_settings; s=get_settings(); print('profile=', s.cloud_profile); print('host=', s.opensearch_host); print('index=', s.search_index_alias)"
