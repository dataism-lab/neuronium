"""Minimal LLM-critic contract for the autofix demo loop.

Provides ``DemoCriticVerdict`` — a tightly-scoped Pydantic model used as
the structured-output schema for the LLM critic node.

Hard rule: verdict ``PASS`` is allowed **only** when ``evidence`` is
non-empty.  This is enforced both by the system prompt and at runtime
via :func:`parse_critic_verdict`.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class DemoCriticVerdict(BaseModel):
    """Minimal critic verdict contract (demo only).

    Fields are intentionally limited. Optional ``suggestions`` (B2 §7.2.2)
    allows critics to propose fix hints for verdict-driven local fix.
    """

    # Required for OpenAI `response_format: json_schema` strict mode:
    # top-level objects must explicitly forbid additional properties.
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["PASS", "FAIL", "UNCERTAIN"]
    confidence: float = 1.0
    evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    # Optional fix hints: list of dicts with e.g. action, expected_improvement, effort_estimate
    suggestions: list[dict[str, Any]] = Field(default_factory=list)


def critic_json_schema() -> dict[str, Any]:
    """Return an OpenAI-strict-compatible JSON schema for the critic output.

    Some OpenAI-compatible endpoints require:
    - `additionalProperties: false` for objects
    - `required` to include every key in `properties` except optional ones (e.g. suggestions)
    """
    schema: dict[str, Any] = DemoCriticVerdict.model_json_schema()
    props = schema.get("properties")
    if isinstance(props, dict):
        # Keep suggestions optional so old responses without it still parse
        required = [k for k in props if k != "suggestions"]
        schema["required"] = required
        schema["additionalProperties"] = False
    return schema


# -- Prompts -----------------------------------------------------------------

CRITIC_SYSTEM_PROMPT = (
    "You are a strict code-execution critic.  You receive:\n"
    "- The user's OBJECTIVE\n"
    "- The generated Python CODE\n"
    "- The execution result (exit_code, stdout, stderr)\n\n"
    "Evaluate whether the code correctly accomplishes the objective.\n\n"
    "Reply with a JSON object matching this EXACT schema:\n"
    '  {"verdict": "PASS"|"FAIL"|"UNCERTAIN",\n'
    '   "confidence": <float 0..1>,\n'
    '   "evidence": [<string>, ...],\n'
    '   "gaps": [<string>, ...]}\n\n'
    "Rules:\n"
    '- verdict "PASS" is allowed ONLY when evidence is NON-EMPTY.\n'
    '- If the code crashed or produced wrong output, verdict MUST be "FAIL".\n'
    "- List concrete evidence (e.g. successful stdout, correct values).\n"
    "- List gaps (missing checks, edge cases, etc.).\n"
    "- Do NOT add any text outside the JSON object."
)


# -- Parsing helper ----------------------------------------------------------

def parse_critic_verdict(raw_content: str) -> DemoCriticVerdict:
    """Parse LLM output into a ``DemoCriticVerdict``.

    Enforces the hard rule: PASS requires non-empty evidence.
    If parsing fails, returns UNCERTAIN with an explanatory gap.
    """
    try:
        data = json.loads(raw_content)
        verdict = DemoCriticVerdict(**data)
    except Exception:
        return DemoCriticVerdict(
            verdict="UNCERTAIN",
            confidence=0.0,
            evidence=[],
            gaps=[f"Failed to parse critic output: {raw_content[:200]}"],
        )

    # Hard rule: PASS requires evidence
    if verdict.verdict == "PASS" and not verdict.evidence:
        return DemoCriticVerdict(
            verdict="UNCERTAIN",
            confidence=verdict.confidence,
            evidence=[],
            gaps=["Critic returned PASS without evidence — downgraded to UNCERTAIN"],
        )

    return verdict
