from src.ontology.domain_extractors import extract_domain_entities_and_relations


def test_domain_extractors_team_division_app_platform_diagram():
    passage = {
        "doc_id": "D1",
        "text": (
            "Team: Payments Core\n"
            "Division: Consumer Technology\n"
            "Application: LedgerService\n"
            "Architecture diagram: https://example.com/arch.png\n"
            "Deploy to EKS using kubectl apply -f ...\n"
        ),
    }
    ents, edges = extract_domain_entities_and_relations(passage)
    # Entities
    names = {(e["type"], e["name"]) for e in ents}
    assert ("Team", "Payments Core") in names
    assert ("Division", "Consumer Technology") in names
    assert ("Application", "LedgerService") in names
    assert any(t for t in ents if t.get("type") == "Platform" and t.get("name") == "EKS")
    assert any(t for t in ents if t.get("type") == "Diagram")
    # Relations
    rel_types = {e["type"] for e in edges}
    assert "IN_DIVISION" in rel_types
    assert "OWNS" in rel_types
    assert "RUNS_ON" in rel_types
    assert "DOCUMENTED_BY" in rel_types

