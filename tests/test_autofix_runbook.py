"""Tests for AutofixDemoRunbook and graph_builder contract."""

from __future__ import annotations

from neuronium_agent.planning.autofix_demo_runbook import AutofixDemoRunbook
from neuronium_agent.planning.dag import ActionGraph, GraphMetadata, GraphNode
from neuronium_agent.planning.runbook_contract import (
    ActionGraphStage,
    Runbook,
    StageSuccessGate,
)
from neuronium_agent.planning.runbook_registry import get_runbook


def test_autofix_demo_is_registered() -> None:
    """autofix_demo runbook is registered and returns AutofixDemoRunbook."""
    runbook = get_runbook("autofix_demo")
    assert runbook is not None
    assert runbook.runbook_id == "autofix_demo"
    assert isinstance(runbook, AutofixDemoRunbook)


def test_autofix_demo_build_stages_returns_two_stages() -> None:
    """AutofixDemoRunbook.build_stages returns two stages: iter1 with graph, iter2 with graph_builder."""
    runbook = AutofixDemoRunbook()
    stages = runbook.build_stages(
        objective="Print hello",
        constraints=[],
        metadata={},
        execution_id="test-exec-1",
    )
    assert len(stages) == 2

    # Stage 1: explicit graph, gate with execute + critic; exit on success, proceed on fail
    s1 = stages[0]
    assert s1.stage_id == "autofix_demo:iter1"
    assert s1.graph is not None
    assert s1.graph_builder is None
    assert "execute" in s1.success_gate.required_completed_nodes
    assert s1.success_gate.critic_node_id == "critic"
    assert s1.exit_run_on_success is True
    assert s1.proceed_to_next_stage_on_fail is True

    # Stage 2: graph_builder, no graph; gate with execute_fix + critic_fix
    s2 = stages[1]
    assert s2.stage_id == "autofix_demo:iter2"
    assert s2.graph is None
    assert s2.graph_builder is not None
    assert "execute_fix" in s2.success_gate.required_completed_nodes
    assert s2.success_gate.critic_node_id == "critic_fix"
    assert s2.exit_run_on_success is False
    assert s2.proceed_to_next_stage_on_fail is False


def test_autofix_demo_iter2_graph_builder_returns_graph_and_override() -> None:
    """Stage 2 graph_builder returns (ActionGraph, fix_context dict)."""
    runbook = AutofixDemoRunbook()
    stages = runbook.build_stages(
        objective="Print hello",
        constraints=[],
        metadata={},
        execution_id="test-exec-2",
    )
    s2 = stages[1]
    assert s2.graph_builder is not None

    # Minimal prev_stage_results and verdict to build iter2
    from neuronium_agent.nodes.base import NodeOutput
    from neuronium_agent.verification.demo_critic import DemoCriticVerdict

    prev_results = {
        "generate": NodeOutput(
            outputs={"content": "print('hi')"},
            status="COMPLETED",
        ),
        "execute": NodeOutput(
            outputs={"exit_code": 1, "stdout": "", "stderr": "NameError: name 'x' is not defined"},
            status="FAILED",
        ),
    }
    prev_verdict = DemoCriticVerdict(
        verdict="FAIL",
        confidence=0.9,
        evidence=[],
        gaps=["Code failed at runtime"],
    )
    context = {
        "objective": "Print hello",
        "constraints": [],
        "prev_stage_results": prev_results,
        "prev_stage_verdict": prev_verdict,
        "execution_id": "test-exec-2",
        "metadata": {},
    }
    result = s2.graph_builder(context)
    assert isinstance(result, tuple)
    graph, initial_inputs = result[0], result[1]
    assert graph is not None
    assert graph.metadata.plan_id
    assert [n.node_id for n in graph.nodes] == ["fix", "execute_fix", "critic_fix"]
    assert initial_inputs is not None
    assert "previous_code" in initial_inputs
    assert "previous_verdict" in initial_inputs
    assert initial_inputs["previous_gaps"] == ["Code failed at runtime"]


class TestStageDefaultModelId:
    """B13 Part 2: ActionGraphStage.default_model_id contract."""

    def test_action_graph_stage_accepts_default_model_id(self) -> None:
        """ActionGraphStage can be constructed with default_model_id."""
        stage = ActionGraphStage(
            stage_id="test:stage1",
            graph=ActionGraph(
                metadata=GraphMetadata(plan_id="p1", description=""),
                nodes=[GraphNode(node_id="m1", node_type="model", label="M")],
                edges=[],
            ),
            success_gate=StageSuccessGate(required_completed_nodes=["m1"]),
            default_model_id="default",
        )
        assert stage.default_model_id == "default"

    def test_runbook_stage_with_default_model_id_builds_stage(self) -> None:
        """Runbook can return a stage with default_model_id set."""
        class RunbookWithStageDefaultModel(Runbook):
            @property
            def runbook_id(self) -> str:
                return "stage_default_model_test"

            def build_stages(
                self,
                *,
                objective: str,
                constraints: list[str],
                metadata: dict,
                execution_id: str,
            ) -> list[ActionGraphStage]:
                graph = ActionGraph(
                    metadata=GraphMetadata(plan_id="p1", description=""),
                    nodes=[
                        GraphNode(
                            node_id="m1",
                            node_type="model",
                            label="Task",
                            parameters={},
                        ),
                    ],
                    edges=[],
                )
                return [
                    ActionGraphStage(
                        stage_id="stage_default_model_test:stage1",
                        graph=graph,
                        success_gate=StageSuccessGate(required_completed_nodes=["m1"]),
                        default_model_id="default",
                    ),
                ]

        runbook = RunbookWithStageDefaultModel()
        stages = runbook.build_stages(
            objective="Test",
            constraints=[],
            metadata={},
            execution_id="exec-1",
        )
        assert len(stages) == 1
        assert stages[0].default_model_id == "default"
