from __future__ import annotations

import json

from neuronium_agent.cli import main as cli_main
from neuronium_agent.nodes.base import NodeOutput
from neuronium_agent.planning.extraction_contract import MissingField
from neuronium_agent.planning.htn_recursive_backend import (
    HtnRecursivePlannerBackend,
    _BackendOptions,
)
from neuronium_agent.planning.planner_contract import DynamicPlannerSpec, PlannerRequest


def test_fallback_questions_are_human_friendly_with_examples() -> None:
    questions = HtnRecursivePlannerBackend._build_clarification_questions_fallback([
        MissingField(field="url", reason="URL is required", critical=True),
        MissingField(field="doc_paths", reason="Need source files", critical=True),
    ])
    by_key = {q.key: q for q in questions}

    assert "url" in by_key
    assert "ссыл" in by_key["url"].prompt.lower()
    assert by_key["url"].examples == ["https://example.com/article"]

    assert "doc_paths" in by_key
    assert by_key["doc_paths"].expected_type == "string_list"
    assert by_key["doc_paths"].examples
    assert "," in by_key["doc_paths"].examples[0]


def test_model_question_normalization_adds_prompt_examples_and_group_sorting() -> None:
    backend = HtnRecursivePlannerBackend()
    request = PlannerRequest(
        objective="Сделай отчёт",
        constraints=[],
        metadata={},
        runbook_id="rb",
        stage_id="stage",
        execution_id="exec-123",
        spec=DynamicPlannerSpec(),
    )

    def _execute_graph(graph, _inputs, _deterministic):
        node_id = graph.nodes[0].node_id
        payload = {
            "questions": [
                {
                    "key": "path",
                    "prompt": "",
                    "path": "/tool_args/path",
                    "expected_schema": {"type": "string"},
                    "expected_type": "string",
                    "required": True,
                    "examples": [],
                },
                {
                    "key": "recipient_name",
                    "prompt": "",
                    "path": "/inputs/recipient_name",
                    "expected_schema": {"type": "string"},
                    "expected_type": "string",
                    "required": True,
                    "examples": [],
                },
            ]
        }
        return {node_id: NodeOutput(outputs={"content": json.dumps(payload)})}

    questions = backend._build_clarification_questions_with_model(
        request=request,
        missing_fields=[
            MissingField(field="recipient_name", reason="Need recipient", critical=True),
            MissingField(field="path", reason="Need output path", critical=True),
        ],
        execute_graph=_execute_graph,
        options=_BackendOptions(),
    )

    assert [q.key for q in questions] == ["recipient_name", "path"]
    assert all(q.prompt.strip() for q in questions)
    assert all(q.examples for q in questions)


def test_backend_question_grouping_uses_path_prefix() -> None:
    assert HtnRecursivePlannerBackend._question_group_from_path("/inputs/url") == "/inputs"
    assert HtnRecursivePlannerBackend._question_group_from_path("/tool_args/path") == "/tool_args"
    assert HtnRecursivePlannerBackend._question_group_from_path("/url") == "/"


def test_cli_prompt_helpers_include_examples_and_groups() -> None:
    assert cli_main._question_group_from_path("/inputs/url") == "inputs"
    assert cli_main._question_group_from_path("/url") == "root"

    prompt = cli_main._question_prompt_with_examples({
        "key": "url",
        "prompt": "Укажи URL",
        "examples": ["https://example.com/news"],
    })
    assert "Пример:" in prompt
    assert "https://example.com/news" in prompt
