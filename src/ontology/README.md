# Ontology and Graph Construction

Purpose: research and prototype structural discovery over OpenSearch documents to extract steps, normalize entities, and build a directed process graph.

- Scope: step extraction (regex, spaCy, LLM), entity normalization/linking, sequence/precedence inference, graph schema and storage (Neo4j/OpenSearch), evaluation.
- Inputs: OpenSearch indices (`khub-opensearch-index`, `khub-opensearch-swagger-index`) via `src.infra.opensearch_client` and `src.services.passage_extractor`.
- Outputs: Step and Entity nodes with `NEXT`/`REQUIRES`/`MENTIONS`/`SAME_AS` relations; optional OpenSearch indices for steps/edges.

See `research_plan.md` for the detailed plan and milestones.
