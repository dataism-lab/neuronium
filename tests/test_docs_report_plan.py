from __future__ import annotations

from neuronium_agent.planning.runbooks import plan_docs_report_v1


def test_docs_report_plan_is_deterministic_over_doc_path_order() -> None:
    graph_a = plan_docs_report_v1(
        objective="O",
        constraints=[],
        doc_paths=["b.md", "a.md"],
        plan_id="p",
    )
    graph_b = plan_docs_report_v1(
        objective="O",
        constraints=[],
        doc_paths=["a.md", "b.md"],
        plan_id="p",
    )

    # Node IDs must match in the same order (paths are sorted internally).
    assert [n.node_id for n in graph_a.nodes] == [n.node_id for n in graph_b.nodes]

    # First read node should target a.md after sorting.
    read0 = [n for n in graph_a.nodes if n.node_id == "read_000"][0]
    assert read0.parameters["tool_args"]["path"] == "a.md"
    assert read0.parameters["tool_args"]["out_key"] == "doc_000"

