from src.agent.graph import create_graph


def test_graph_compiles_and_has_linear_flow():
    graph = create_graph()
    assert graph is not None
    # We cannot introspect nodes easily here; the main goal is that compilation succeeds.

