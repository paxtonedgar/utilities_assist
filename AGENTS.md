# Repository Guidelines

## Project Structure & Module Organization
Core logic sits in `src/`. Use `src/agent` for LangGraph nodes and regex intent slotting, `src/retrieval` for search, reranking, and view builders, and `src/compose` plus `src/services` for response assembly and shared business logic. Infrastructure and telemetry helpers live in `src/infra` and `src/telemetry`. Operational helpers live in `scripts/`, reference material in `docs/`, cached models inside `models/`, and config templates beside `config_*.ini`. Repository tests live in `tests/`, with smoke scripts like `test_context_flow.py` at the root for quick checks.

## Build, Test, and Development Commands
Install dependencies with `make install`. Start the Streamlit UI using `make run-local` (expects `config.ini`) or `make run-mock` when Azure access is unavailable. `make setup-local` provisions OpenSearch and loads sample docs; follow with `make embed-local` if you need FAISS embeddings. Run the primary suite through `make test` for CI parity or `make test-local` for verbose local runs. OpenSearch helpers include `make start-opensearch`, `make stop-opensearch`, and `make index-local`.

## Coding Style & Naming Conventions
Target Python 3.11. Format and lint before committing: `ruff format src` and `ruff check src --fix` (88 char line length, modern Python rules). Add or update type hints and validate with `mypy src --strict`. Modules and functions should use snake_case, classes use PascalCase, and CLI entry points (`python -m src.ontology...`) should mirror existing verb-noun patterns.

## Testing Guidelines
Pytest discovers files under `tests/` named `test_*.py` and classes `Test*`, enforced by `pytest.ini`. Mark long-running or external-service checks with `@pytest.mark.slow` and higher-level workflows with `@pytest.mark.integration`. Always run `python -m pytest tests/ -v --strict-markers` before opening a PR and document any new markers in `pytest.ini`.

## Commit & Pull Request Guidelines
Commit messages follow the Conventional Commit style in history, e.g., `feat(ontology): improve table parsers`. Keep scopes aligned with top-level packages (`agent`, `retrieval`, `ontology`, etc.) and write imperative summaries. Every PR should include a succinct description, links to related tickets, explicit callouts for configuration changes, and proof of testing (pytest output, evaluation JSON, or Streamlit screenshots). Confirm Jenkins and Spinnaker pipelines are green before requesting merge.

## Configuration & Security Tips
Copy `config_neo4j_example.ini` or other templates instead of editing secrets in-place, and never commit live credentials. Use environment variables such as `UTILITIES_CONFIG=config.ini` and `CLOUD_PROFILE=jpmc_azure` to switch contexts, and validate resolved settings with `make check-settings`. When touching auth or ingestion modules, run security checks (`bandit`, `semgrep`) and rotate any leaked keys immediately.
