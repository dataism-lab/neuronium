"""ModelNode — LLM inference via OpenAI (v1 default provider).

Records every LLM response for replay (IBS §3.3).
Supports structured output via ``response_format`` (CONFIG_SPEC §2.6).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from neuronium_agent.nodes.base import (
    BaseNode,
    NodeInput,
    NodeOutput,
    QualitySignals,
)

logger = logging.getLogger(__name__)


class ModelNode(BaseNode):
    """LLM call node — default provider: OpenAI."""

    node_type: str = "model"

    def __init__(
        self,
        node_id: str,
        parameters: dict[str, Any] | None = None,
        *,
        model: str = "gpt-4.1-mini",
        provider: str = "openai",
        api_key_env: str = "NEURONIUM_OPENAI_API_KEY",
        base_url: str | None = None,
        structured_output: bool = True,
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 2,
    ) -> None:
        super().__init__(node_id, parameters)
        self.model = model
        self.provider = provider
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.structured_output = structured_output
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    # -- Replay support ------------------------------------------------------

    #: If set, responses will be read from this list instead of calling LLM.
    _replay_responses: list[dict[str, Any]] | None = None
    _replay_index: int = 0

    #: Mutable recording list — trace recorder attaches it before execution.
    _recorded_responses: list[dict[str, Any]] | None = None

    def set_replay_responses(self, responses: list[dict[str, Any]]) -> None:
        """Inject pre-recorded LLM responses for deterministic replay."""
        self._replay_responses = list(responses)
        self._replay_index = 0

    def enable_recording(self) -> list[dict[str, Any]]:
        """Enable recording and return the (mutable) recording list."""
        self._recorded_responses = []
        return self._recorded_responses

    # -- Execute -------------------------------------------------------------

    def execute(self, node_input: NodeInput) -> NodeOutput:
        prompt = node_input.inputs.get("prompt", "")
        system_prompt = node_input.parameters.get(
            "system_prompt",
            self.parameters.get("system_prompt", "You are a helpful assistant."),
        )
        json_schema = node_input.parameters.get(
            "json_schema",
            self.parameters.get("json_schema"),
        )

        # Replay path ---------------------------------------------------------
        if self._replay_responses is not None:
            if self._replay_index >= len(self._replay_responses):
                return NodeOutput(
                    status="FAILED",
                    error="Replay exhausted: no more recorded responses",
                )
            resp = self._replay_responses[self._replay_index]
            self._replay_index += 1
            return NodeOutput(
                outputs=resp.get("outputs", {}),
                quality_signals=QualitySignals(
                    **resp.get("quality_signals", {})
                ),
                status="COMPLETED",
            )

        # Live LLM call -------------------------------------------------------
        import openai

        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return NodeOutput(
                status="FAILED",
                error=f"Missing env var {self.api_key_env}",
            )

        client = openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "seed": node_input.context.random_seed,
        }

        if self.structured_output and json_schema:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as exc:
            return NodeOutput(status="FAILED", error=str(exc))

        choice = completion.choices[0]
        content = choice.message.content or ""
        usage = completion.usage

        quality = QualitySignals(
            tokens_used=usage.total_tokens if usage else None,
        )

        outputs: dict[str, Any] = {"content": content}

        # Try to parse as JSON for structured output
        if self.structured_output and json_schema:
            try:
                outputs["parsed"] = json.loads(content)
            except json.JSONDecodeError:
                outputs["parsed"] = None

        # Record for replay
        record = {
            "outputs": outputs,
            "quality_signals": quality.model_dump(mode="json"),
        }
        if self._recorded_responses is not None:
            self._recorded_responses.append(record)

        return NodeOutput(
            outputs=outputs,
            quality_signals=quality,
            status="COMPLETED",
        )
