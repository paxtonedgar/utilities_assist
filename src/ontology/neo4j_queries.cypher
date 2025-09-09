-- Utilities Assist Ontology - Neo4j Analysis Queries
-- Copy/paste these into Neo4j Browser for analysis

-- ============================================================================
-- BASIC EXPLORATION QUERIES
-- ============================================================================

-- Overview: Count of each node type
MATCH (n)
RETURN labels(n) AS node_type, count(n) AS count
ORDER BY count DESC;

-- Overview: Count of each relationship type  
MATCH ()-[r]->()
RETURN type(r) AS relationship_type, count(r) AS count
ORDER BY count DESC;

-- Sample data: Show a few steps with their documents
MATCH (s:Step)-[:OF_DOC]->(d:Doc)
RETURN s.label, s.section, s.order, d.id, d.index_name
LIMIT 10;

-- ============================================================================
-- PROCESS CHAIN ANALYSIS
-- ============================================================================

-- Find longest process chains across all documents
MATCH p=(start:Step)-[:NEXT*1..20]->(end:Step)
WHERE NOT ()-[:NEXT]->(start) -- Chain starts here
RETURN 
  start.doc_id AS document,
  start.section AS section,
  length(p) AS chain_length,
  [n IN nodes(p) | n.label] AS process_steps
ORDER BY chain_length DESC
LIMIT 10;

-- Show complete process for a specific document
MATCH (d:Doc {id: 'YOUR_DOC_ID_HERE'})<-[:OF_DOC]-(s:Step)
OPTIONAL MATCH path=(s)-[:NEXT*0..50]->(:Step)-[:OF_DOC]->(d)
RETURN path, s.order
ORDER BY s.order;

-- Find steps that start new processes (no incoming NEXT)
MATCH (s:Step)-[:OF_DOC]->(d:Doc)
WHERE NOT ()-[:NEXT]->(s)
RETURN s.doc_id, s.section, s.label, s.order, d.index_name
ORDER BY s.doc_id, s.order;

-- Find steps that end processes (no outgoing NEXT)  
MATCH (s:Step)-[:OF_DOC]->(d:Doc)
WHERE NOT (s)-[:NEXT]->()
RETURN s.doc_id, s.section, s.label, s.order, d.index_name
ORDER BY s.doc_id, s.order;

-- ============================================================================
-- QUALITY ANALYSIS
-- ============================================================================

-- Show unaccepted NEXT relationships (need manual curation)
MATCH (a:Step)-[r:NEXT]->(b:Step)
WHERE r.accepted = false
RETURN 
  a.doc_id AS document,
  a.label + ' → ' + b.label AS transition,
  a.section AS section,
  r.confidence AS confidence
ORDER BY a.doc_id, a.order
LIMIT 50;

-- Show accepted NEXT relationships (already curated)
MATCH (a:Step)-[r:NEXT]->(b:Step)  
WHERE r.accepted = true
RETURN 
  a.doc_id AS document,
  a.label + ' → ' + b.label AS transition,
  r.confidence AS confidence
ORDER BY r.confidence DESC
LIMIT 20;

-- Find potential quality issues: very short or very long labels
MATCH (s:Step)
WHERE length(s.label) < 5 OR length(s.label) > 200
RETURN s.doc_id, s.label, length(s.label) AS label_length, s.section
ORDER BY label_length;

-- ============================================================================
-- DOCUMENT ANALYSIS  
-- ============================================================================

-- Documents by step count (most complex processes)
MATCH (d:Doc)
OPTIONAL MATCH (d)<-[:OF_DOC]-(s:Step)
RETURN 
  d.id AS document,
  d.index_name AS source_index,
  count(s) AS step_count
ORDER BY step_count DESC;

-- Sections with most steps (most detailed areas)
MATCH (s:Step)
RETURN 
  s.section AS section_name,
  count(s) AS step_count,
  count(DISTINCT s.doc_id) AS document_count
ORDER BY step_count DESC
LIMIT 20;

-- Find documents with missing process chains (no NEXT relationships)
MATCH (d:Doc)<-[:OF_DOC]-(s:Step)
WHERE NOT (s)-[:NEXT]-() AND NOT ()-[:NEXT]->(s)
RETURN 
  d.id AS document,
  count(s) AS isolated_steps
ORDER BY isolated_steps DESC;

-- ============================================================================
-- SEARCH & FILTERING
-- ============================================================================

-- Find steps containing specific keywords
MATCH (s:Step)
WHERE s.label CONTAINS 'deploy' OR s.label CONTAINS 'config'
RETURN s.doc_id, s.section, s.label, s.evidence
ORDER BY s.doc_id, s.order;

-- Find steps by verb (action type)
MATCH (s:Step) 
WHERE s.verb IS NOT NULL
RETURN s.verb, count(s) AS frequency
ORDER BY frequency DESC;

-- Search within specific document sections
MATCH (s:Step)
WHERE s.section CONTAINS 'Installation'
RETURN s.doc_id, s.label, s.order, s.evidence
ORDER BY s.doc_id, s.order;

-- ============================================================================
-- DATA QUALITY CHECKS
-- ============================================================================

-- Check for circular NEXT relationships (should be empty)
MATCH (s:Step)-[:NEXT*2..10]->(s)
RETURN s.doc_id, s.label, s.section;

-- Find orphaned steps (no document relationship)
MATCH (s:Step)
WHERE NOT (s)-[:OF_DOC]->()
RETURN s.label, s.doc_id
LIMIT 20;

-- Find steps with missing required fields
MATCH (s:Step)
WHERE s.label IS NULL OR s.doc_id IS NULL OR s.order IS NULL
RETURN s, labels(s), properties(s)
LIMIT 10;

-- Check relationship consistency
MATCH (a:Step)-[r:NEXT]->(b:Step)
WHERE a.doc_id <> b.doc_id
RETURN 
  'Cross-document NEXT relationship' AS issue,
  a.doc_id AS source_doc, 
  b.doc_id AS target_doc
LIMIT 10;

-- ============================================================================
-- CURATION HELPERS
-- ============================================================================

-- Mark high-confidence relationships as accepted
MATCH (a:Step)-[r:NEXT]->(b:Step)
WHERE r.confidence >= 0.8 AND r.accepted = false
SET r.accepted = true
RETURN count(r) AS relationships_accepted;

-- Find relationships that need manual review
MATCH (a:Step)-[r:NEXT]->(b:Step)
WHERE r.accepted = false AND r.confidence < 0.5
RETURN 
  a.doc_id AS document,
  a.label + ' → ' + b.label AS transition,
  r.confidence AS confidence,
  a.evidence AS context
ORDER BY r.confidence ASC
LIMIT 20;

-- Promote curated relationships (example for future use)
MATCH (a:Step)-[r:NEXT]->(b:Step)
WHERE r.accepted = true
MERGE (a)-[cr:CURATED_NEXT]->(b)
SET cr = properties(r)
RETURN count(cr) AS curated_relationships;

-- ============================================================================
-- ANALYTICS & INSIGHTS
-- ============================================================================

-- Most common step verbs
MATCH (s:Step)
WHERE s.verb IS NOT NULL
RETURN s.verb, count(*) AS frequency
ORDER BY frequency DESC
LIMIT 20;

-- Most common step patterns (first 3 words)
MATCH (s:Step)
RETURN 
  substring(s.label, 0, 30) AS pattern,
  count(*) AS frequency
ORDER BY frequency DESC
LIMIT 20;

-- Document complexity scoring
MATCH (d:Doc)<-[:OF_DOC]-(s:Step)
OPTIONAL MATCH (s)-[r:NEXT]->()
RETURN 
  d.id AS document,
  count(s) AS total_steps,
  count(r) AS total_transitions,
  round(100.0 * count(r) / count(s)) AS connectivity_percent
ORDER BY total_steps DESC;

-- Section complexity analysis
MATCH (s:Step)
OPTIONAL MATCH (s)-[r:NEXT]->()
RETURN 
  s.section AS section,
  count(s) AS steps,
  count(r) AS transitions,
  count(DISTINCT s.doc_id) AS documents
ORDER BY steps DESC
LIMIT 15;