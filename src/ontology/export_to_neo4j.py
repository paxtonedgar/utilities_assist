#!/usr/bin/env python3
"""Export ontology pipeline NDJSON files to Neo4j graph database."""

import sys
import json
import configparser
from pathlib import Path
from typing import Dict, Any
from neo4j import GraphDatabase

def load_config() -> configparser.ConfigParser:
    """Load configuration from config.ini."""
    config = configparser.ConfigParser()
    config_paths = ["config.ini", "src/config.ini"]
    
    for path in config_paths:
        if Path(path).exists():
            config.read(path)
            print(f"✅ Loaded config from: {path}")
            return config
    
    raise FileNotFoundError("config.ini not found in current directory or src/")

def get_neo4j_config(config: configparser.ConfigParser) -> Dict[str, str]:
    """Extract Neo4j configuration."""
    if "neo4j" not in config:
        print("❌ No [neo4j] section in config.ini")
        print("Please add:\n[neo4j]\nuri=bolt://localhost:7687\nuser=neo4j\npassword=your_password\nimport_dir=/var/lib/neo4j/import")
        sys.exit(1)
    
    neo4j_config = dict(config["neo4j"])
    required = ["uri", "user", "password"]
    
    for key in required:
        if key not in neo4j_config:
            print(f"❌ Missing required Neo4j config: {key}")
            sys.exit(1)
    
    return neo4j_config

def create_constraints_and_indexes(tx):
    """Create constraints and indexes for optimal performance."""
    queries = [
        # Unique constraints
        "CREATE CONSTRAINT step_id IF NOT EXISTS FOR (s:Step) REQUIRE s.id IS UNIQUE",
        "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Doc) REQUIRE d.id IS UNIQUE",
        
        # Performance indexes
        "CREATE INDEX step_doc_section IF NOT EXISTS FOR (s:Step) ON (s.doc_id, s.section)",
        "CREATE INDEX doc_index_name IF NOT EXISTS FOR (d:Doc) ON (d.index_name)",
        "CREATE INDEX step_order IF NOT EXISTS FOR (s:Step) ON (s.order)",
        "CREATE INDEX step_label IF NOT EXISTS FOR (s:Step) ON (s.label)",
    ]
    
    for query in queries:
        try:
            tx.run(query)
            print(f"✅ {query.split()[1]} constraint/index")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"⚠️  {query.split()[1]} already exists")
            else:
                print(f"❌ Error creating constraint/index: {e}")

def load_docs_ndjson(tx):
    """Load docs.ndjson using APOC."""
    query = """
    CALL apoc.load.json('file:///docs.ndjson') YIELD value AS v
    MERGE (d:Doc {id: v.doc_id})
      SET d.index_name = v.index,
          d.step_cnt = v.steps,
          d.edge_cnt = v.edges,
          d.updated_at = datetime();
    """
    
    result = tx.run(query)
    summary = result.consume()
    print(f"✅ Loaded {summary.counters.nodes_created} documents")

def load_steps_ndjson(tx):
    """Load steps.ndjson using APOC."""
    query = """
    CALL apoc.load.json('file:///steps.ndjson') YIELD value AS v
    MERGE (s:Step {id: v.doc_id + '_' + toString(v.order)})
      SET s.label = v.canonical_label,
          s.verb = v.verb,
          s.object = v.object,
          s.doc_id = v.doc_id,
          s.section = v.section_title,
          s.order = v.order,
          s.evidence = v.snippet,
          s.evidence_hash = v.content_hash,
          s.page_url = v.page_url,
          s.offsets = v.offsets;
    
    WITH s, v
    MATCH (d:Doc {id: v.doc_id})
    MERGE (s)-[:OF_DOC]->(d);
    """
    
    result = tx.run(query)
    summary = result.consume()
    print(f"✅ Loaded {summary.counters.nodes_created} steps, {summary.counters.relationships_created} OF_DOC relationships")

def load_edges_ndjson(tx):
    """Load edges.ndjson using APOC."""
    query = """
    CALL apoc.load.json('file:///edges.ndjson') YIELD value AS v
    WITH v WHERE coalesce(v.type, 'NEXT') = 'NEXT'
    WITH v.doc_id + '_' + toString(v.src_order) AS src_id,
         v.doc_id + '_' + toString(v.dst_order) AS dst_id,
         v
    MATCH (a:Step {id: src_id})
    MATCH (b:Step {id: dst_id})
    MERGE (a)-[r:NEXT]->(b)
      SET r.accepted = coalesce(v.accepted, false),
          r.confidence = coalesce(v.score, 1.0),
          r.signals = coalesce(v.signals, []),
          r.evidence = coalesce(v.evidence_refs, []);
    """
    
    result = tx.run(query)
    summary = result.consume()
    print(f"✅ Loaded {summary.counters.relationships_created} NEXT relationships")

def run_sanity_checks(tx):
    """Run sanity checks on imported data."""
    print("\n🔍 Running sanity checks...")
    
    # Check for cycles
    cycle_query = "MATCH (s:Step)-[:NEXT*2..10]->(s) RETURN count(s) AS cycles"
    cycles = tx.run(cycle_query).single()["cycles"]
    if cycles > 0:
        print(f"⚠️  Found {cycles} circular NEXT relationships")
    else:
        print("✅ No circular NEXT relationships")
    
    # Check for orphaned steps
    orphan_query = "MATCH (s:Step) WHERE NOT (s)-[:OF_DOC]->() RETURN count(s) AS orphans"
    orphans = tx.run(orphan_query).single()["orphans"]
    if orphans > 0:
        print(f"⚠️  Found {orphans} steps without document links")
    else:
        print("✅ All steps linked to documents")
    
    # Show stats
    doc_count = tx.run("MATCH (d:Doc) RETURN count(d) AS cnt").single()["cnt"]
    step_count = tx.run("MATCH (s:Step) RETURN count(s) AS cnt").single()["cnt"]
    edge_count = tx.run("MATCH ()-[r:NEXT]->() RETURN count(r) AS cnt").single()["cnt"]
    
    print(f"\n📊 Import Summary:")
    print(f"   Documents: {doc_count}")
    print(f"   Steps: {step_count}")
    print(f"   NEXT edges: {edge_count}")
    
    # Show longest chains
    chain_query = """
    MATCH p=(s:Step)-[:NEXT*1..20]->(t:Step)
    RETURN length(p) AS chain_length, s.doc_id AS doc_id
    ORDER BY chain_length DESC LIMIT 3
    """
    
    chains = list(tx.run(chain_query))
    if chains:
        print(f"\n🔗 Longest process chains:")
        for record in chains:
            print(f"   {record['doc_id']}: {record['chain_length']} steps")

def export_to_neo4j(ndjson_dir: str):
    """Main export function."""
    print("🚀 Starting Neo4j export...")
    
    # Load configuration
    config = load_config()
    neo4j_config = get_neo4j_config(config)
    
    # Check NDJSON files exist
    ndjson_path = Path(ndjson_dir)
    required_files = ["docs.ndjson", "steps.ndjson", "edges.ndjson"]
    
    for filename in required_files:
        file_path = ndjson_path / filename
        if not file_path.exists():
            print(f"❌ Missing required file: {file_path}")
            sys.exit(1)
        print(f"✅ Found: {file_path}")
    
    # Check Neo4j import directory
    import_dir = neo4j_config.get("import_dir", "/var/lib/neo4j/import")
    import_path = Path(import_dir)
    
    print(f"\n📁 Neo4j import directory: {import_path}")
    print(f"   Please ensure NDJSON files are copied to this directory")
    print(f"   Commands:")
    for filename in required_files:
        src = ndjson_path / filename
        dst = import_path / filename
        print(f"   cp {src} {dst}")
    
    # Connect to Neo4j
    print(f"\n🔌 Connecting to Neo4j: {neo4j_config['uri']}")
    
    try:
        with GraphDatabase.driver(
            neo4j_config["uri"],
            auth=(neo4j_config["user"], neo4j_config["password"])
        ) as driver:
            # Test connection
            driver.verify_connectivity()
            print("✅ Connected to Neo4j")
            
            with driver.session() as session:
                print("\n1️⃣  Creating constraints and indexes...")
                session.execute_write(create_constraints_and_indexes)
                
                print("\n2️⃣  Loading documents...")
                session.execute_write(load_docs_ndjson)
                
                print("\n3️⃣  Loading steps...")
                session.execute_write(load_steps_ndjson)
                
                print("\n4️⃣  Loading NEXT relationships...")
                session.execute_write(load_edges_ndjson)
                
                print("\n5️⃣  Running sanity checks...")
                session.execute_read(run_sanity_checks)
                
                print("\n🎉 Export completed successfully!")
                print("\n📊 Ready for Bloom visualization!")
                
    except Exception as e:
        print(f"❌ Neo4j connection error: {e}")
        print(f"   Check your Neo4j configuration in config.ini")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python export_to_neo4j.py <ndjson_directory>")
        print("Example: python export_to_neo4j.py outputs/continuous/khub-opensearch-swagger-index")
        sys.exit(1)
    
    export_to_neo4j(sys.argv[1])