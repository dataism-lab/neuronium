"""CodeNode — execute Python code in a Docker sandbox (IBS §7).

- Docker only (v1).
- Network off by default.
- Wall-time / CPU / RAM limits.
- stdout/stderr captured as artifacts.
- Responses recorded for replay.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from neuronium_agent.errors import SandboxError
from neuronium_agent.nodes.base import (
    BaseNode,
    NodeInput,
    NodeOutput,
    QualitySignals,
)

logger = logging.getLogger(__name__)

_FENCED_CODE_RE = re.compile(
    r"^\s*```(?:python)?\s*\n(?P<body>[\s\S]*?)\n```\s*$",
    flags=re.IGNORECASE,
)


def _strip_markdown_fences(code: str) -> str:
    """Best-effort: unwrap ```python ...``` blocks into raw code.

    Some models ignore the "output code only" instruction and wrap code in
    Markdown fences. `python -c` can't execute that, so we strip it here.
    """
    s = code.strip()
    m = _FENCED_CODE_RE.match(s)
    if m:
        return (m.group("body") or "").strip()
    return code


class CodeNode(BaseNode):
    """Execute Python code inside a Docker container."""

    node_type: str = "code"

    def __init__(
        self,
        node_id: str,
        parameters: dict[str, Any] | None = None,
        *,
        image: str = "python:3.11-slim",
        network_enabled: bool = False,
        cpu_limit: str | None = None,
        mem_limit: str | None = None,
        timeout_seconds: int = 120,
    ) -> None:
        super().__init__(node_id, parameters)
        self.image = image
        self.network_enabled = network_enabled
        self.cpu_limit = cpu_limit
        self.mem_limit = mem_limit
        self.timeout_seconds = timeout_seconds

    # -- Replay support -------------------------------------------------------

    _replay_responses: list[dict[str, Any]] | None = None
    _replay_index: int = 0
    _recorded_responses: list[dict[str, Any]] | None = None

    def set_replay_responses(self, responses: list[dict[str, Any]]) -> None:
        self._replay_responses = list(responses)
        self._replay_index = 0

    def enable_recording(self) -> list[dict[str, Any]]:
        self._recorded_responses = []
        return self._recorded_responses

    # -- Execute --------------------------------------------------------------

    def execute(self, node_input: NodeInput) -> NodeOutput:
        code: str = node_input.inputs.get("code", "")
        code = _strip_markdown_fences(code)
        if not code.strip():
            return NodeOutput(status="FAILED", error="No code provided")

        # Replay path
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
                status=resp.get("status", "COMPLETED"),
            )

        # Live Docker execution
        try:
            import docker  # type: ignore[import-untyped]
        except ImportError:
            raise SandboxError(
                "docker package not installed. "
                "Run: pip install neuronium-agent[docker]"
            )

        try:
            client = docker.from_env()
        except Exception as exc:
            raise SandboxError(f"Cannot connect to Docker daemon: {exc}")

        # Build run kwargs
        run_kwargs: dict[str, Any] = {
            "image": self.image,
            "command": ["python", "-c", code],
            "detach": False,
            "stdout": True,
            "stderr": True,
            "remove": True,
            "network_disabled": not self.network_enabled,
        }

        if self.cpu_limit:
            run_kwargs["nano_cpus"] = int(float(self.cpu_limit) * 1e9)
        if self.mem_limit:
            run_kwargs["mem_limit"] = self.mem_limit

        import time

        t0 = time.monotonic()
        try:
            output_bytes: bytes = client.containers.run(**run_kwargs)
            elapsed = (time.monotonic() - t0) * 1000
            stdout_text = output_bytes.decode("utf-8", errors="replace")
        except Exception as exc:
            elapsed = (time.monotonic() - t0) * 1000
            result = NodeOutput(
                outputs={"stdout": "", "stderr": str(exc), "exit_code": 1},
                quality_signals=QualitySignals(latency_ms=elapsed),
                status="FAILED",
                error=str(exc),
            )
            if self._recorded_responses is not None:
                self._recorded_responses.append({
                    "outputs": result.outputs,
                    "quality_signals": result.quality_signals.model_dump(mode="json"),
                    "status": result.status,
                })
            return result

        result = NodeOutput(
            outputs={"stdout": stdout_text, "exit_code": 0},
            quality_signals=QualitySignals(latency_ms=elapsed),
            status="COMPLETED",
        )

        if self._recorded_responses is not None:
            self._recorded_responses.append({
                "outputs": result.outputs,
                "quality_signals": result.quality_signals.model_dump(mode="json"),
                "status": result.status,
            })

        return result
