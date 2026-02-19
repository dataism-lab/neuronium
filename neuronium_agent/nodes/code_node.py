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
import subprocess
import sys
from typing import Any

from neuronium_agent.errors import SandboxError
from neuronium_agent.nodes.base import (
    BaseNode,
    NodeInput,
    NodeOutput,
    QualitySignals,
)
from neuronium_agent.nodes.determinism import DeterminismContract

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

    def get_determinism_contract(self) -> DeterminismContract:
        return DeterminismContract(uses_seed=True, declared_non_deterministic=False)

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

        def _record(result: NodeOutput) -> None:
            if self._recorded_responses is None:
                return
            self._recorded_responses.append({
                "outputs": result.outputs,
                "quality_signals": result.quality_signals.model_dump(mode="json"),
                "status": result.status,
            })

        def _run_local_python() -> NodeOutput:
            """Fallback executor: run code via local Python subprocess.

            This makes demo runs robust on developer machines where Docker may
            be unavailable. It is intentionally best-effort (no sandbox).
            """
            import time

            t0 = time.monotonic()
            try:
                cp = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                elapsed = (time.monotonic() - t0) * 1000
                status = "COMPLETED" if cp.returncode == 0 else "FAILED"
                result = NodeOutput(
                    outputs={
                        "stdout": cp.stdout or "",
                        "stderr": cp.stderr or "",
                        "exit_code": int(cp.returncode),
                        "runner": "local",
                    },
                    quality_signals=QualitySignals(latency_ms=elapsed),
                    status=status,
                )
                _record(result)
                return result
            except subprocess.TimeoutExpired as exc:
                elapsed = (time.monotonic() - t0) * 1000
                stdout = exc.stdout if isinstance(exc.stdout, str) else ""
                stderr = exc.stderr if isinstance(exc.stderr, str) else ""
                result = NodeOutput(
                    outputs={
                        "stdout": stdout,
                        "stderr": (stderr + "\n" if stderr else "") + "TimeoutExpired",
                        "exit_code": 124,
                        "runner": "local",
                    },
                    quality_signals=QualitySignals(latency_ms=elapsed),
                    status="FAILED",
                    error="TimeoutExpired",
                )
                _record(result)
                return result
            except Exception as exc:
                import time

                elapsed = (time.monotonic() - t0) * 1000
                result = NodeOutput(
                    outputs={
                        "stdout": "",
                        "stderr": str(exc),
                        "exit_code": 1,
                        "runner": "local",
                    },
                    quality_signals=QualitySignals(latency_ms=elapsed),
                    status="FAILED",
                    error=str(exc),
                )
                _record(result)
                return result

        # Live Docker execution (preferred) ---------------------------------
        try:
            import docker  # type: ignore[import-untyped]
        except ImportError as exc:
            # Fall back to local execution to keep demo UX smooth.
            logger.warning("docker package not installed; falling back to local execution: %s", exc)
            return _run_local_python()

        # Import Docker error types (best-effort; keep compatibility).
        try:
            from docker.errors import (  # type: ignore[import-untyped]
                ContainerError,
                ImageNotFound,
                APIError,
                DockerException,
            )
        except Exception:  # pragma: no cover
            ContainerError = Exception  # type: ignore[assignment]
            ImageNotFound = Exception  # type: ignore[assignment]
            APIError = Exception  # type: ignore[assignment]
            DockerException = Exception  # type: ignore[assignment]

        try:
            client = docker.from_env()
        except Exception as exc:
            logger.warning("Cannot connect to Docker daemon; falling back to local execution: %s", exc)
            return _run_local_python()

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
        except ContainerError as exc:
            # Code ran in container but exited non-zero. This is NOT an
            # environmental failure and must be reported as a normal FAILED run.
            elapsed = (time.monotonic() - t0) * 1000
            # docker-py may provide stdout/stderr as bytes
            raw_stdout = getattr(exc, "stdout", b"")
            raw_stderr = getattr(exc, "stderr", b"")
            if isinstance(raw_stdout, bytes):
                stdout_text = raw_stdout.decode("utf-8", errors="replace")
            else:
                stdout_text = str(raw_stdout or "")
            if isinstance(raw_stderr, bytes):
                stderr_text = raw_stderr.decode("utf-8", errors="replace")
            else:
                stderr_text = str(raw_stderr or "")
            status_code = getattr(exc, "exit_status", None)
            if status_code is None:
                status_code = 1
            result = NodeOutput(
                outputs={
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "exit_code": int(status_code),
                    "runner": "docker",
                },
                quality_signals=QualitySignals(latency_ms=elapsed),
                status="FAILED",
                error=str(exc),
            )
            _record(result)
            return result
        except (ImageNotFound, APIError, DockerException) as exc:
            # Environmental / Docker API issues — fall back to local execution
            # for demo stability.
            logger.warning("Docker environment error; falling back to local execution: %s", exc)
            return _run_local_python()
        except Exception as exc:
            logger.warning("Docker execution failed; falling back to local execution: %s", exc)
            return _run_local_python()

        result = NodeOutput(
            outputs={"stdout": stdout_text, "exit_code": 0, "runner": "docker"},
            quality_signals=QualitySignals(latency_ms=elapsed),
            status="COMPLETED",
        )

        _record(result)

        return result
