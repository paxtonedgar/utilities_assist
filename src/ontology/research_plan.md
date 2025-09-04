# Research Plan: Structural Discovery and Graph Construction

## Objectives
- Extract procedural steps from documents (imperatives, bullets, numbered lists).
- Normalize and link entities across docs (e.g., "CI pipeline" ≈ "build pipeline").
- Infer directed relations (NEXT, REQUIRES) to construct multi-doc chains.
- Persist to Neo4j and/or OpenSearch indices for exploration and analytics.

## Key Questions
- Step extraction: How reliable are regex + dependency parsing vs. LLMs on our corpus?
- Entity vocab: What canonical forms and synonym mappings are required for utilities domain?
- Linking: Which cues best infer precedence across docs (temporal markers, section ordering, cross-refs)?
- Confidence: How do we quantify edge confidence and support provenance for QA?

## Data Sources
- OpenSearch indices: `khub-opensearch-index`, `khub-opensearch-swagger-index`.
- Access via `src.infra.opensearch_client` and `src.services.passage_extractor`.

## Methods
- Retrieval
  - Seed queries biasing step-like content (configure|deploy|verify|submit|install|provision|enable|request|register|create|update).
  - Use nested section search (`sections.content`) and hybrid retrieval.

- Step Extraction
  - Regex: bullets/numbering; imperative phrase patterns.
  - spaCy: dependency parsing for imperative detection (`sent.root.pos_ == VERB`, no explicit subject) and (verb, object, modifiers) tuples.
  - LLM: JSON extraction for tough formats; constrain schema and validate with heuristics.
  - Provenance: retain `doc_id`, `index`, `section_title`, offsets, evidence text.

- Entity Linking
  - Dictionary + synonyms (`data/synonyms.json`) to normalize common variants.
  - Embedding similarity (sentence-transformers or existing embeddings) for clustering and canonicalization.
  - String similarity (Levenshtein) as fallback.

- Relation Inference
  - Intra-doc: list ordering → `NEXT` edges; preface cues ("before", "after", "prerequisite") → `REQUIRES`.
  - Inter-doc: link steps sharing canonical entities + temporal cues, API/task co-mentions, URL anchors.
  - Optional LLM pairwise entailment checks to validate edges above a threshold.

- Graph Schema
  - Nodes: `Step(id, label, verb, object, entities[], doc_id, section, order, evidence)`
  - Nodes: `Entity(id, name, type, synonyms[])`
  - Edges: `NEXT`, `REQUIRES`, `MENTIONS`, `SAME_AS` with `confidence`, `evidence_refs[]`.
  - Neo4j primary; optional OpenSearch indices `utility-steps-graph`, `utility-steps-edges`.

## Tooling
- spaCy (dependency parsing), sentence-transformers (similarity), FuzzyWuzzy/Levenshtein.
- Neo4j Python Driver for persistence; OpenSearch Dashboards for validating indexed steps.

## Evaluation
- Precision/recall on a labeled subset; edge coherence (acyclic chains within flows unless explicitly cyclic).
- Human-in-the-loop audits using provenance; inter-annotator agreement for edge types.

## Milestones
1) Corpus probe + heuristics baseline
2) spaCy extractor + provenance, initial `NEXT` edges
3) Entity normalization + cross-doc linking
4) Neo4j graph and OpenSearch steps index
5) LLM-assisted extraction for edge cases
6) Metrics and dashboards

## Risks & Mitigations
- Noisy lists/tables → stricter section filtering + heuristics.
- Ambiguous entities → conservative thresholds + manual seed dictionary growth.
- Over-linking across docs → edge confidence with multi-signal requirement.

## Next Actions
- Implement scaffolds in `src/ontology`: extractors, entity_linking, graph_schema, graph_builder.
- Create minimal notebooks/scripts to iterate on extraction heuristics.
