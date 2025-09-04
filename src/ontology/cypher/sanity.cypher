// Indexes and constraints
CREATE CONSTRAINT step_id IF NOT EXISTS ON (s:Step) ASSERT s.id IS UNIQUE;
CREATE CONSTRAINT canon_id IF NOT EXISTS ON (c:CanonicalEntity) ASSERT c.id IS UNIQUE;
CREATE INDEX step_doc_section IF NOT EXISTS FOR (s:Step) ON (s.doc_id, s.section);
CREATE INDEX canon_name IF NOT EXISTS FOR (c:CanonicalEntity) ON (c.name);

// NEXT cycles (should be none)
MATCH (s:Step {doc_id:$doc}) CALL apoc.path.expandConfig(s,{relationshipFilter:'NEXT>', minLevel:2, maxLevel:10, terminateNodes:[s]}) YIELD path
RETURN path LIMIT 3;

// Conflicting orders
MATCH (a:Step)-[:NEXT]->(b:Step), (b)-[:NEXT]->(a:Step) RETURN a,b LIMIT 10;

// SAME_AS hubs
MATCH (c:CanonicalEntity)<-[:ALIAS_OF]-(:SurfaceEntity)
WITH c, count(*) AS deg WHERE deg > 25
RETURN c.name AS canonical, deg ORDER BY deg DESC LIMIT 25;

// Low-confidence cross-doc REQUIRES
MATCH (a:Step)-[r:REQUIRES]->(b:Step)
WHERE a.doc_id <> b.doc_id AND r.confidence < 0.55
RETURN a.label AS from, b.label AS to, r.confidence AS conf, r.evidence_refs[0] AS ev LIMIT 50;

