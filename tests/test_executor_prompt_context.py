"""Tests for default prompt construction with upstream context."""

from __future__ import annotations

from neuronium_agent.execution.executor import DAGExecutor
from neuronium_agent.nodes.base import BaseNode, NodeInput, NodeOutput
from neuronium_agent.planning.dag import ActionGraph, GraphEdge, GraphMetadata, GraphNode


class StaticOutputsNode(BaseNode):
    def __init__(self, node_id: str, outputs: dict) -> None:
        super().__init__(node_id)
        self._outputs = dict(outputs)

    def execute(self, node_input: NodeInput) -> NodeOutput:
        return NodeOutput(outputs=dict(self._outputs), status="COMPLETED")


class EchoPromptNode(BaseNode):
    def execute(self, node_input: NodeInput) -> NodeOutput:
        return NodeOutput(outputs={"prompt": node_input.inputs.get("prompt", "")}, status="COMPLETED")


def test_model_prompt_includes_objective_and_upstream_context() -> None:
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p1", description="prompt context"),
        nodes=[
            GraphNode(node_id="src", node_type="aggregate", label="src", priority=0),
            GraphNode(node_id="draft", node_type="model", label="draft", priority=1),
        ],
        edges=[GraphEdge(source="src", target="draft", edge_type="data")],
    )
    registry = {
        "src": StaticOutputsNode("src", outputs={"doc_000": "HELLO"}),
        "draft": EchoPromptNode("draft"),
    }
    ex = DAGExecutor(registry, execution_id="e1", trace_id="t1", random_seed=0)

    results = ex.execute(graph, initial_inputs={"objective": "OBJ"})
    prompt = results["draft"].outputs["prompt"]
    assert "OBJ" in prompt
    assert "doc_000" in prompt
    assert "HELLO" in prompt


def test_critic_prompt_includes_source_context_for_web_like_inputs() -> None:
    graph = ActionGraph(
        metadata=GraphMetadata(plan_id="p2", description="critic context"),
        nodes=[
            GraphNode(node_id="src", node_type="aggregate", label="src", priority=0),
            GraphNode(
                node_id="critic",
                node_type="model",
                label="critic",
                priority=1,
                parameters={"json_schema": {"type": "object"}, "context_kind": "web"},
            ),
        ],
        edges=[GraphEdge(source="src", target="critic", edge_type="data")],
    )
    registry = {
        "src": StaticOutputsNode(
            "src",
            outputs={"title_guess": "Arxiv title", "text": "Article text", "final_url": "https://arxiv.org/x"},
        ),
        "critic": EchoPromptNode("critic"),
    }
    ex = DAGExecutor(registry, execution_id="e2", trace_id="t2", random_seed=0)

    results = ex.execute(graph, initial_inputs={"objective": "Summarize this article"})
    prompt = results["critic"].outputs["prompt"]
    assert "CONTEXT_KIND: web" in prompt
    assert "title_guess" in prompt
    assert "Arxiv title" in prompt
    assert "final_url" in prompt

