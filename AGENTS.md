# Repository Guidelines

## Project Structure & Module Organization
- `src/agent`: LangGraph orchestration, nodes, tools, prompts.
- `src/retrieval`: Search, ranking, views, and suite manifests.
- `src/infra`: Config, clients (OpenAI/Azure/AWS), OpenSearch client, telemetry.
- `src/services`: Passage extraction, schema learner, models.
- `src/compose`, `src/controllers`, `src/util`, `src/quality`, `src/telemetry`, `src/app`.
- `tests/`: Pytest test suite; use `test_*.py` naming.
- `scripts/`: Local OpenSearch, indexing, embedding helpers.

## Build, Test, and Development Commands
- Install: `make install`
- Run (local): `make run-local` (uses `UTILITIES_CONFIG=config.local.ini`)
- Start OpenSearch (local): `make start-opensearch`; index mock docs: `make index-local`
- Embeddings + FAISS (local): `make embed-local`
- Tests: `make test` or `make test-local`
- Lint/format: `ruff check src --fix`; `ruff format src`
- Type-check: `mypy src --strict`

## Coding Style & Naming Conventions
- Python 3.11, 4-space indentation, type hints required for new/changed code.
- Naming: `snake_case` for functions/modules, `PascalCase` for classes, `CONSTANT_CASE` for constants.
- Keep functions small and pure; prefer dependency injection.
- Logging via `src/telemetry/logger` (no bare prints). Avoid hardcoded endpoints; use `src/infra/settings.ApplicationSettings`.

## Testing Guidelines
- Framework: Pytest. Locations: `tests/` with `test_*.py`, classes `Test*`, functions `test_*`.
- Markers: `@pytest.mark.slow`, `@pytest.mark.integration` (see `pytest.ini`).
- Run: `pytest -v` or `make test-local`. Add unit tests for new modules and edge cases.

## Commit & Pull Request Guidelines
- Commits: Imperative mood, short subject (≤50 chars), explanatory body when needed. Scope examples: `infra:`, `retrieval:`, `agent:`.
- PRs: Clear description, linked issues, before/after notes, test evidence (`pytest -v` output). Small, focused diffs. Update docs when behavior/config changes.

## Security & Configuration Tips
- Profiles: `CLOUD_PROFILE` (`local`, `jpmc_azure`, `tests`). Config priority: env vars > `.env` > `config.ini`.
- Do not commit secrets. For Azure OpenAI/AWS, load via config/env; JPMC proxy set automatically in `src/infra/clients._setup_jpmc_proxy`.
- Verify OpenSearch locally: `make check-opensearch`. Use aliases via `ApplicationSettings.search_index_alias`.
