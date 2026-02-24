import torch
from src.graph.career_graph import build_career_graph, get_neighbors


def test_build_graph_basic():
    transitions = [
        {"from_job": "Software Engineer", "to_job": "Senior Engineer", "years": 2.5},
        {"from_job": "Software Engineer", "to_job": "Tech Lead", "years": 4.0},
    ]

    graph, mapping = build_career_graph(transitions)

    assert graph.edge_index.shape == (2, 2)
    assert graph.edge_attr.shape == (2, 1)
    assert "Software Engineer" in mapping


def test_empty_transitions():
    graph, mapping = build_career_graph([])
    assert graph.edge_index is None or graph.edge_index.numel() == 0
    assert mapping == {}


def test_get_neighbors():
    transitions = [
        {"from_job": "A", "to_job": "B", "years": 1.0},
        {"from_job": "A", "to_job": "C", "years": 2.0},
    ]

    graph, mapping = build_career_graph(transitions)

    a_idx = mapping["A"]
    neighbors = get_neighbors(graph, a_idx)

    assert len(neighbors) == 2