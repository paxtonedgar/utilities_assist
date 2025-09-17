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
	@echo "ℹ️ ontology-probe deprecated; use 'make ontology-run' or 'make ontology-queue' instead."

# Full corpus scan (document-by-document via PIT)
.PHONY: ontology-scan
ontology-scan:
	@echo "📚 Scanning corpus document-by-document (PIT)"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.doc_by_doc --index "$(or $(INDEX),)" --limit $(or $(LIMIT),1000) --batch $(or $(BATCH),200) --out-dir $(or $(OUT),outputs/ontology_scan) $(if $(RESUME),--resume,) $(if $(CKPT),--checkpoint-file $(CKPT),)

# ID queue workflow: build ids via match_all (non-PIT) and process by id
.PHONY: ontology-queue
ontology-queue:
	@echo "🧾 Building/processing ID queue"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.id_queue --index "$(or $(INDEX),)" --queue-file $(or $(QUEUE),outputs/ontology_queue/ids.ndjson) --batch $(or $(BATCH),500) --limit $(or $(LIMIT),0) --out-dir $(or $(OUT),outputs/ontology_queue) $(if $(RESUME),--resume,) $(if $(LEDGER),--ledger $(LEDGER),) --mode $(or $(MODE),both)

# Continuous scan across indices until completion (main + swagger by default)
.PHONY: ontology-run
ontology-run:
	@echo "🔁 Continuous scan across indices"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.continuous_queue --indices "$(or $(INDICES),)" --out-dir $(or $(OUT),outputs/continuous) --batch $(or $(BATCH),500)

# Push NDJSON outputs into Neo4j
.PHONY: push-neo4j
push-neo4j:
	@echo "📤 Pushing ontology outputs to Neo4j"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.neo4j_writer --inputs $(or $(INPUTS),outputs/ontology_queue outputs/ontology_scan outputs/continuous/khub-opensearch-index outputs/continuous/khub-opensearch-swagger-index) --database $(or $(DB),neo4j)

# Export NDJSON to CSV for LOAD CSV (no APOC)
.PHONY: ontology-export-csv
ontology-export-csv:
	@echo "📦 Converting NDJSON -> CSV for LOAD CSV"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.export_csv --input $(or $(INPUT),outputs/continuous/khub-opensearch-swagger-index) --out $(or $(OUT),outputs/continuous/khub-opensearch-swagger-index/csv)
	@echo ""
	@echo "➡️  Start an HTTP server in the CSV folder (from your VDI):"
	@echo "   cd $(or $(OUT),outputs/continuous/khub-opensearch-swagger-index/csv) && python -m http.server 9000"
	@echo ""
	@echo "🔗 Then, in Neo4j, run LOAD CSV commands (replace <host> with your VDI hostname):"
	@echo "   CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Doc) REQUIRE d.id IS UNIQUE;"
	@echo "   CREATE CONSTRAINT step_id IF NOT EXISTS FOR (s:Step) REQUIRE s.id IS UNIQUE;"
	@echo "   LOAD CSV WITH HEADERS FROM 'http://<host>:9000/docs.csv' AS row MERGE (d:Doc {id: row.id}) SET d.index_name=row.index_name, d.step_cnt=toInteger(row.step_cnt), d.edge_cnt=toInteger(row.edge_cnt);"
	@echo "   LOAD CSV WITH HEADERS FROM 'http://<host>:9000/steps.csv' AS row MERGE (s:Step {id: row.id}) SET s.label=row.label, s.verb=row.verb, s.obj=row.obj, s.doc_id=row.doc_id, s.section=row.section, s.order=toInteger(row.order), s.page_url=row.page_url WITH s,row MERGE (d:Doc {id: row.doc_composite_id}) MERGE (s)-[:OF_DOC]->(d);"
	@echo "   LOAD CSV WITH HEADERS FROM 'http://<host>:9000/edges.csv' AS row MATCH (a:Step {id: row.src}), (b:Step {id: row.dst}) MERGE (a)-[r:NEXT]->(b) SET r.confidence=toFloat(row.score), r.accepted=coalesce(r.accepted, row.accepted='true');"

# Build Document Semantic Map (Phase 1) from diagnostics
.PHONY: ontology-semantic-map
ontology-semantic-map:
	@echo "🗺  Building Document Semantic Map"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.semantic_map_runner --diag-dir $(if $(DIAG),$(DIAG),outputs/diagnostics/khub-test-md) --out $(if $(OUT),$(OUT),outputs/semantic_map/khub-test-md) --max-docs $(if $(MAX),$(MAX),500) $(if $(DEBUG),--debug-structure,)

# Phase 1 Quality report
.PHONY: ontology-quality-report
ontology-quality-report:
	@echo "📊 Generating Phase 1 quality report"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.semantic_quality --semantic-dir $(if $(SEM),$(SEM),outputs/semantic_map/khub-test-md) --diagnostics-dir $(if $(DIAG),$(DIAG),outputs/diagnostics/khub-test-md) $(if $(TERMS),--terms $(TERMS),)

# Taxonomy term suggestions via contrastive embeddings
.PHONY: ontology-taxonomy-terms
ontology-taxonomy-terms:
	@echo "🧭 Generating taxonomy term suggestions"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.taxonomy_terms.cli pipeline \
		--semantic-dir $(if $(SEM),$(SEM),$(if $(DIAG),$(DIAG),outputs/semantic_map/khub-test-md)) \
		--out $(if $(OUT),$(OUT),outputs/taxonomy_terms/khub-test-md) \
		$(if $(CONFIG),--config-path $(CONFIG),)

# Investigate table formats in an index
.PHONY: ontology-table-detective
ontology-table-detective:
	@echo "🕵️  Hunting for tables in index"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.table_detective --index $(or $(INDEX),khub-test-md) --limit $(or $(LIMIT),10) $(if $(DIAG),--diag-dir $(DIAG),) $(if $(SEM),--semantic-dir $(SEM),)

# Phase 2: Extract entities from semantic map segments
.PHONY: ontology-extract-entities
ontology-extract-entities:
	@echo "🔎 Extracting entities from segments"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.phase2_extractor --semantic-dir $(or $(SEM),outputs/semantic_map/khub-test-md) --out $(or $(OUT),outputs/semantic_map/khub-test-md/entities.jsonl)

# Diagnose index content structure and HTML/text fields
.PHONY: ontology-diagnose
ontology-diagnose:
	@echo "🔎 Diagnosing index content fields (sampling)"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -m src.ontology.diagnose_index --indices $(or $(INDICES),khub) --limit $(or $(LIMIT),25) --out $(or $(OUT),outputs/diagnostics) --redact $(or $(REDACT),keywords) --max-candidates $(or $(CANDS),4) --min-len $(or $(MINLEN),120) $(if $(GZIP),--gzip,) $(if $(SPLIT),--split $(SPLIT),) $(if $(SUMMARY_ONLY),--summary-only,)

# List all indices in OpenSearch (uses config.ini + jpmc_azure auth)
.PHONY: list-indices
list-indices:
	@echo "📚 Listing OpenSearch indices from config.ini host"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -c "from src.infra.settings import get_settings; from src.infra.clients import _get_aws_auth, _setup_jpmc_proxy; import requests; s=get_settings(); host=s.opensearch_host.rstrip('/'); _setup_jpmc_proxy(); auth=_get_aws_auth(); url=f'{host}/_cat/indices?format=json'; r=requests.get(url, auth=auth, timeout=30); r.raise_for_status(); inds=r.json(); [print(f'{it.get('index',''):<40} status={it.get('status','')} docs={it.get('docs.count','')} pri={it.get('pri','')} rep={it.get('rep','')}') for it in sorted(inds, key=lambda x: x.get('index',''))]"

check-settings:
	@echo "🔧 Printing resolved OpenSearch settings from config.ini"
	@UTILITIES_CONFIG=config.ini CLOUD_PROFILE=jpmc_azure \
	python -c "from src.infra.settings import get_settings; s=get_settings(); print('profile=', s.cloud_profile); print('host=', s.opensearch_host); print('index=', s.search_index_alias)"
