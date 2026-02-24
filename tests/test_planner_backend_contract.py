from __future__ import annotations

import json

from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphMetadata, GraphNode
from neuronium_agent.planning.planner_backend import get_planner_backend
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec, PlannerRequest


def test_legacy_planner_backend_returns_planner_result_contract() -> None:
    backend = get_planner_backend("legacy_dynamic_v1")
    spec = DynamicPlannerSpec(allowed_tool_names=["fs.read_text"])
    request = PlannerRequest(
        objective="Read one doc and draft report",
        constraints=[],
        metadata={"doc_paths": ["/tmp/a.md"]},
        runbook_id="dynamic_planner_demo_v1",
        stage_id="dynamic_planner_demo_v1:stage1",
        execution_id="abc123def456",
        spec=spec,
        operator_catalog_hash="hash-001",
    )

    planned = ActionGraph(
        metadata=GraphMetadata(plan_id="plan-test-1", description="test plan"),
        nodes=[
            GraphNode(
                node_id="read_000",
                node_type="mcp",
                parameters={"tool_name": "fs.read_text", "tool_args": {"path": "/tmp/a.md"}},
            ),
        ],
        edges=[],
    ).model_dump(mode="json")

    captured: dict[str, object] = {}

    def fake_execute(graph: ActionGraph, initial_inputs: dict[str, object], suppress: bool) -> dict[str, NodeOutput]:
        captured["planner_graph"] = graph
        captured["initial_inputs"] = initial_inputs
        captured["suppress"] = suppress
        return {
            spec.planner_node_id: NodeOutput(
                outputs={"parsed": planned, "content": json.dumps(planned)},
                status="COMPLETED",
            )
        }

    result = backend.plan(request=request, execute_graph=fake_execute)
    assert result.backend_name == "legacy_dynamic_v1"
    assert result.backend_version == "1"
    assert result.operator_catalog_hash == "hash-001"
    assert result.action_graph.metadata.plan_id == "plan-test-1"
    assert isinstance(captured["initial_inputs"], dict)
    assert "prompt" in captured["initial_inputs"]
    assert captured["suppress"] is True


def test_htn_recursive_backend_returns_planner_result_contract() -> None:
    backend = get_planner_backend("htn_recursive_v0")
    spec = DynamicPlannerSpec(
        backend_name="htn_recursive_v0",
        backend_version="0",
    )
    request = PlannerRequest(
        objective="Read one doc and draft report",
        constraints=[],
        metadata={"doc_paths": ["/tmp/a.md"]},
        runbook_id="htn_recursive_demo_v0",
        stage_id="htn_recursive_demo_v0:stage1",
        execution_id="abc123def456",
        spec=spec,
        operator_catalog_hash="hash-htn-001",
    )

    def fake_execute(
        graph: ActionGraph,
        initial_inputs: dict[str, object],
        suppress: bool,
    ) -> dict[str, NodeOutput]:
        assert suppress is True
        node_ids = {n.node_id for n in graph.nodes}
        if node_ids == {"persist_user_request", "extract_entities"}:
            return {
                "persist_user_request": NodeOutput(
                    outputs={"artifact_id": "sha256:req-001"},
                    status="COMPLETED",
                ),
                "extract_entities": NodeOutput(
                    outputs={"urls": [], "file_paths": [], "basenames": []},
                    status="COMPLETED",
                ),
            }
        extract_nodes = [nid for nid in node_ids if nid.endswith("_extract_envelope")]
        if len(extract_nodes) == 1:
            return {
                extract_nodes[0]: NodeOutput(
                    outputs={
                        "parsed": {
                            "intent": {"task_type": "docs_summary", "confidence": 0.9},
                            "inputs": {"doc_paths": ["/tmp/a.md"]},
                            "missing_fields": [],
                            "extras": {},
                        }
                    },
                    status="COMPLETED",
                )
            }
        raise AssertionError(f"Unexpected planner graph nodes: {sorted(node_ids)}")

    result = backend.plan(request=request, execute_graph=fake_execute)

    assert result.backend_name == "htn_recursive_v0"
    assert result.backend_version == "0"
    assert result.operator_catalog_hash == "hash-htn-001"
    assert result.decision_trace is not None
    assert result.decision_trace.notes.get("context_kind") == "docs"
    assert result.decision_trace.decomposition_steps
    assert result.decision_trace.leaf_operators
    assert {n.node_id for n in result.action_graph.nodes} >= {
        "read_000",
        "merge_docs",
        "draft_report",
        "critic_report",
    }


def test_htn_recursive_backend_can_use_model_assisted_method_selection() -> None:
    backend = get_planner_backend("htn_recursive_v0")
    spec = DynamicPlannerSpec(
        backend_name="htn_recursive_v0",
        backend_version="0",
        backend_options={
            "model_assisted_method_selection": True,
            "model_assisted_max_calls": 2,
            "planner_node_prefix": "htn_select",
        },
    )
    request = PlannerRequest(
        objective="Read one doc and draft report",
        constraints=[],
        metadata={"doc_paths": ["/tmp/a.md"]},
        runbook_id="htn_recursive_demo_v0",
        stage_id="htn_recursive_demo_v0:stage1",
        execution_id="assisted001",
        spec=spec,
        operator_catalog_hash="hash-htn-002",
    )
    calls: list[str] = []

    def fake_execute(
        graph: ActionGraph,
        initial_inputs: dict[str, object],
        suppress: bool,
    ) -> dict[str, NodeOutput]:
        assert suppress is True
        node_ids = {n.node_id for n in graph.nodes}
        if node_ids == {"persist_user_request", "extract_entities"}:
            return {
                "persist_user_request": NodeOutput(
                    outputs={"artifact_id": "sha256:req-002"},
                    status="COMPLETED",
                ),
                "extract_entities": NodeOutput(
                    outputs={"urls": [], "file_paths": [], "basenames": []},
                    status="COMPLETED",
                ),
            }
        planner_node_id = graph.nodes[0].node_id
        if planner_node_id.endswith("_extract_envelope"):
            return {
                planner_node_id: NodeOutput(
                    outputs={
                        "parsed": {
                            "intent": {"task_type": "docs_summary", "confidence": 0.9},
                            "inputs": {"doc_paths": ["/tmp/a.md"]},
                            "missing_fields": [],
                            "extras": {},
                        }
                    },
                    status="COMPLETED",
                )
            }
        assert "prompt" in initial_inputs
        calls.append(planner_node_id)
        method_id = "root_docs_pipeline" if len(calls) == 1 else "draft_direct"
        return {
            planner_node_id: NodeOutput(
                outputs={"parsed": {"method_id": method_id}},
                status="COMPLETED",
            )
        }

    result = backend.plan(request=request, execute_graph=fake_execute)
    assert result.decision_trace is not None
    assert result.decision_trace.notes.get("planner_calls") == 2
    assert calls == ["htn_select_00_root", "htn_select_01_synthesize"]


def test_htn_recursive_backend_routes_https_url_to_web_branch() -> None:
    backend = get_planner_backend("htn_recursive_v0")
    request = PlannerRequest(
        objective="сделай сводку https://arxiv.org/abs/1234.5678",
        constraints=[],
        metadata={},
        runbook_id="super_agent_v0",
        stage_id="super_agent_v0:stage1",
        execution_id="webcase001",
        spec=DynamicPlannerSpec(backend_name="htn_recursive_v0", backend_version="0"),
        operator_catalog_hash="hash-web-001",
    )

    def fake_execute(graph: ActionGraph, initial_inputs: dict[str, object], suppress: bool) -> dict[str, NodeOutput]:
        _ = initial_inputs, suppress
        node_ids = {n.node_id for n in graph.nodes}
        if node_ids == {"persist_user_request", "extract_entities"}:
            return {
                "persist_user_request": NodeOutput(outputs={"artifact_id": "sha256:req-web-1"}, status="COMPLETED"),
                "extract_entities": NodeOutput(
                    outputs={"urls": ["https://arxiv.org/abs/1234.5678"], "file_paths": [], "basenames": []},
                    status="COMPLETED",
                ),
            }
        extract_nodes = [nid for nid in node_ids if nid.endswith("_extract_envelope")]
        if len(extract_nodes) == 1:
            return {
                extract_nodes[0]: NodeOutput(
                    outputs={
                        "parsed": {
                            "intent": {"task_type": "news_summary", "confidence": 0.9},
                            "inputs": {"urls": ["https://arxiv.org/abs/1234.5678"]},
                            "missing_fields": [],
                            "extras": {},
                        }
                    },
                    status="COMPLETED",
                )
            }
        raise AssertionError(f"unexpected graph {sorted(node_ids)}")

    result = backend.plan(request=request, execute_graph=fake_execute)
    assert result.action_graph is not None
    node_ids = {n.node_id for n in result.action_graph.nodes}
    assert {"fetch_html", "extract_article", "draft_report", "critic_report"} <= node_ids
    assert "read_000" not in node_ids


def test_htn_recursive_backend_prioritizes_urls_over_doc_paths() -> None:
    backend = get_planner_backend("htn_recursive_v0")
    request = PlannerRequest(
        objective="сделай сводку //arxiv.org/html/2511.12869v2",
        constraints=[],
        metadata={},
        runbook_id="super_agent_v0",
        stage_id="super_agent_v0:stage1",
        execution_id="webcase002",
        spec=DynamicPlannerSpec(backend_name="htn_recursive_v0", backend_version="0"),
        operator_catalog_hash="hash-web-002",
    )

    def fake_execute(graph: ActionGraph, initial_inputs: dict[str, object], suppress: bool) -> dict[str, NodeOutput]:
        _ = initial_inputs, suppress
        node_ids = {n.node_id for n in graph.nodes}
        if node_ids == {"persist_user_request", "extract_entities"}:
            return {
                "persist_user_request": NodeOutput(outputs={"artifact_id": "sha256:req-web-2"}, status="COMPLETED"),
                "extract_entities": NodeOutput(
                    outputs={
                        "urls": ["https://arxiv.org/html/2511.12869v2"],
                        "file_paths": [],
                        "basenames": [],
                    },
                    status="COMPLETED",
                ),
            }
        extract_nodes = [nid for nid in node_ids if nid.endswith("_extract_envelope")]
        if len(extract_nodes) == 1:
            return {
                extract_nodes[0]: NodeOutput(
                    outputs={
                        "parsed": {
                            "intent": {"task_type": "news_summary", "confidence": 0.9},
                            # Intentionally includes doc_paths to verify URL priority.
                            "inputs": {
                                "urls": ["https://arxiv.org/html/2511.12869v2"],
                                "doc_paths": ["//arxiv.org/html/2511.12869v2"],
                            },
                            "missing_fields": [],
                            "extras": {},
                        }
                    },
                    status="COMPLETED",
                )
            }
        raise AssertionError(f"unexpected graph {sorted(node_ids)}")

    result = backend.plan(request=request, execute_graph=fake_execute)
    node_ids = {n.node_id for n in result.action_graph.nodes}
    assert "read_000" not in node_ids
    assert {"fetch_html", "extract_article"} <= node_ids


def test_htn_recursive_backend_routes_explicit_relative_path_to_docs_branch() -> None:
    backend = get_planner_backend("htn_recursive_v0")
    request = PlannerRequest(
        objective="сделай сводку ./report.pdf",
        constraints=[],
        metadata={},
        runbook_id="super_agent_v0",
        stage_id="super_agent_v0:stage1",
        execution_id="doccase001",
        spec=DynamicPlannerSpec(backend_name="htn_recursive_v0", backend_version="0"),
        operator_catalog_hash="hash-doc-001",
    )

    def fake_execute(graph: ActionGraph, initial_inputs: dict[str, object], suppress: bool) -> dict[str, NodeOutput]:
        _ = initial_inputs, suppress
        node_ids = {n.node_id for n in graph.nodes}
        if node_ids == {"persist_user_request", "extract_entities"}:
            return {
                "persist_user_request": NodeOutput(outputs={"artifact_id": "sha256:req-doc-1"}, status="COMPLETED"),
                "extract_entities": NodeOutput(
                    outputs={"urls": [], "file_paths": ["./report.pdf"], "basenames": []},
                    status="COMPLETED",
                ),
            }
        extract_nodes = [nid for nid in node_ids if nid.endswith("_extract_envelope")]
        if len(extract_nodes) == 1:
            return {
                extract_nodes[0]: NodeOutput(
                    outputs={
                        "parsed": {
                            "intent": {"task_type": "docs_summary", "confidence": 0.9},
                            "inputs": {},
                            "missing_fields": [],
                            "extras": {},
                        }
                    },
                    status="COMPLETED",
                )
            }
        raise AssertionError(f"unexpected graph {sorted(node_ids)}")

    result = backend.plan(request=request, execute_graph=fake_execute)
    node_ids = {n.node_id for n in result.action_graph.nodes}
    assert "read_000" in node_ids
    assert "fetch_html" not in node_ids


def test_htn_recursive_backend_resolves_llm_basename_via_glob_when_unique() -> None:
    backend = get_planner_backend("htn_recursive_v0")
    request = PlannerRequest(
        objective="сделай сводку data.csv",
        constraints=[],
        metadata={},
        runbook_id="super_agent_v0",
        stage_id="super_agent_v0:stage1",
        execution_id="doccase002",
        spec=DynamicPlannerSpec(backend_name="htn_recursive_v0", backend_version="0"),
        operator_catalog_hash="hash-doc-002",
    )
    glob_calls = 0

    def fake_execute(graph: ActionGraph, initial_inputs: dict[str, object], suppress: bool) -> dict[str, NodeOutput]:
        nonlocal glob_calls
        _ = initial_inputs, suppress
        node_ids = {n.node_id for n in graph.nodes}
        if node_ids == {"persist_user_request", "extract_entities"}:
            return {
                "persist_user_request": NodeOutput(outputs={"artifact_id": "sha256:req-doc-2"}, status="COMPLETED"),
                "extract_entities": NodeOutput(outputs={"urls": [], "file_paths": [], "basenames": []}, status="COMPLETED"),
            }
        if node_ids == {"resolve_path_glob"}:
            glob_calls += 1
            return {
                "resolve_path_glob": NodeOutput(
                    outputs={"paths": ["/tmp/data.csv"]},
                    status="COMPLETED",
                )
            }
        extract_nodes = [nid for nid in node_ids if nid.endswith("_extract_envelope")]
        if len(extract_nodes) == 1:
            return {
                extract_nodes[0]: NodeOutput(
                    outputs={
                        "parsed": {
                            "intent": {"task_type": "docs_summary", "confidence": 0.8},
                            "inputs": {"doc_paths": ["data.csv"]},
                            "missing_fields": [],
                            "extras": {},
                        }
                    },
                    status="COMPLETED",
                )
            }
        raise AssertionError(f"unexpected graph {sorted(node_ids)}")

    result = backend.plan(request=request, execute_graph=fake_execute)
    assert glob_calls == 1
    read_nodes = [n for n in result.action_graph.nodes if n.node_id == "read_000"]
    assert read_nodes
    assert read_nodes[0].parameters.get("tool_args", {}).get("path") == "/tmp/data.csv"


def test_htn_recursive_backend_uses_web_specific_critic_prompt() -> None:
    backend = get_planner_backend("htn_recursive_v0")
    request = PlannerRequest(
        objective="summarize https://arxiv.org/abs/1234.5678",
        constraints=[],
        metadata={},
        runbook_id="super_agent_v0",
        stage_id="super_agent_v0:stage1",
        execution_id="webcritic001",
        spec=DynamicPlannerSpec(backend_name="htn_recursive_v0", backend_version="0"),
        operator_catalog_hash="hash-web-critic",
    )

    def fake_execute(graph: ActionGraph, initial_inputs: dict[str, object], suppress: bool) -> dict[str, NodeOutput]:
        _ = initial_inputs, suppress
        node_ids = {n.node_id for n in graph.nodes}
        if node_ids == {"persist_user_request", "extract_entities"}:
            return {
                "persist_user_request": NodeOutput(outputs={"artifact_id": "sha256:req-web-critic"}, status="COMPLETED"),
                "extract_entities": NodeOutput(
                    outputs={"urls": ["https://arxiv.org/abs/1234.5678"], "file_paths": [], "basenames": []},
                    status="COMPLETED",
                ),
            }
        extract_nodes = [nid for nid in node_ids if nid.endswith("_extract_envelope")]
        if len(extract_nodes) == 1:
            return {
                extract_nodes[0]: NodeOutput(
                    outputs={
                        "parsed": {
                            "intent": {"task_type": "news_summary", "confidence": 0.9},
                            "inputs": {"urls": ["https://arxiv.org/abs/1234.5678"]},
                            "missing_fields": [],
                            "extras": {},
                        }
                    },
                    status="COMPLETED",
                )
            }
        raise AssertionError(f"unexpected graph {sorted(node_ids)}")

    result = backend.plan(request=request, execute_graph=fake_execute)
    critic = next(n for n in result.action_graph.nodes if n.node_id == "critic_report")
    prompt = str(critic.parameters.get("system_prompt", ""))
    assert "web-summary critic" in prompt
    assert "Do NOT require document keys like doc_000 for web tasks." in prompt


def test_htn_recursive_backend_dynamic_schema_validation_escalates_with_tool_required_fields() -> None:
    backend = get_planner_backend("htn_recursive_v0")
    request = PlannerRequest(
        objective="нужен запуск инструмента без параметров",
        constraints=[],
        metadata={
            "dynamic_extraction_schema": True,
            "dynamic_extraction_schema_runbooks": ["super_agent_v0"],
            "dynamic_extraction_schema_stages": ["super_agent_v0:stage1"],
        },
        runbook_id="super_agent_v0",
        stage_id="super_agent_v0:stage1",
        execution_id="phase2val001",
        spec=DynamicPlannerSpec(
            backend_name="htn_recursive_v0",
            backend_version="0",
            allowed_tool_names=["web.fetch_html", "fs.glob"],
        ),
        operator_catalog_hash="hash-phase2-validation",
    )

    def fake_execute(graph: ActionGraph, initial_inputs: dict[str, object], suppress: bool) -> dict[str, NodeOutput]:
        _ = initial_inputs, suppress
        node_ids = {n.node_id for n in graph.nodes}
        if node_ids == {"persist_user_request", "extract_entities"}:
            return {
                "persist_user_request": NodeOutput(
                    outputs={"artifact_id": "sha256:req-phase2-validation"},
                    status="COMPLETED",
                ),
                "extract_entities": NodeOutput(
                    outputs={"urls": [], "file_paths": [], "basenames": []},
                    status="COMPLETED",
                ),
            }
        extract_nodes = [nid for nid in node_ids if nid.endswith("_extract_envelope")]
        if len(extract_nodes) == 1:
            return {
                extract_nodes[0]: NodeOutput(
                    outputs={
                        "parsed": {
                            "intent": {"task_type": "generic_task", "confidence": 0.7},
                            "inputs": {},
                            "missing_fields": [],
                            "extras": {},
                        }
                    },
                    status="COMPLETED",
                )
            }
        clarification_nodes = [nid for nid in node_ids if nid.endswith("_clarification_questions")]
        if len(clarification_nodes) == 1:
            return {
                clarification_nodes[0]: NodeOutput(
                    outputs={"parsed": {"questions": []}},
                    status="COMPLETED",
                )
            }
        if node_ids == {"persist_clarification_request"}:
            return {
                "persist_clarification_request": NodeOutput(
                    outputs={"artifact_id": "sha256:clarify-phase2-validation"},
                    status="COMPLETED",
                )
            }
        raise AssertionError(f"Unexpected planner graph nodes: {sorted(node_ids)}")

    result = backend.plan(request=request, execute_graph=fake_execute)
    assert result.reason == "missing_critical_parameters"
    fields = {
        str(item.get("field", ""))
        for item in result.missing_fields
        if isinstance(item, dict)
    }
    assert {"url", "root", "pattern"} <= fields
