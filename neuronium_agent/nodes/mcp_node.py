"""McpToolNode — MCP server tool invocation (IBS §Stage 3).

v1: stub implementation with capability discovery contract
and policy gates.  Actual MCP protocol calls are future work.

v0.2: local transport (in-process) via ``neuronium_agent.tools.local_tools``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from neuronium_agent.errors import McpError
from neuronium_agent.nodes.base import (
    BaseNode,
    NodeInput,
    NodeOutput,
    QualitySignals,
)
from neuronium_agent.tools.local_tools import (
    ToolCall,
    ToolExecutionError,
    ToolPolicyError,
    invoke_local_tool,
)

logger = logging.getLogger(__name__)


class McpToolNode(BaseNode):
    """Invoke a tool on an MCP server.

    v1 records requests/responses for replay and enforces policy gates.
    """

    node_type: str = "mcp"

    def __init__(
        self,
        node_id: str,
        parameters: dict[str, Any] | None = None,
        *,
        server_name: str = "",
        server_url: str = "",
        timeout_seconds: int = 60,
        policy: dict[str, Any] | None = None,
        tool_runtime: Any | None = None,
    ) -> None:
        super().__init__(node_id, parameters)
        self.server_name = server_name
        self.server_url = server_url
        self.timeout_seconds = timeout_seconds
        self.policy = policy or {}
        self.tool_runtime = tool_runtime

    # -- Replay ---------------------------------------------------------------
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
        # Prefer graph parameters (static tool calls), then dynamic inputs.
        tool_name = (
            node_input.parameters.get("tool_name")
            or node_input.inputs.get("tool_name", "")
        )
        static_tool_args = node_input.parameters.get("tool_args")
        if not isinstance(static_tool_args, dict):
            static_tool_args = {}
        dynamic_tool_args = node_input.inputs.get("tool_args", {})
        if not isinstance(dynamic_tool_args, dict):
            dynamic_tool_args = {}

        # Data-flow by default: predecessor outputs become candidate tool args.
        # Explicit tool_args keep priority over auto-wired inputs.
        auto_tool_args = {
            k: v
            for k, v in node_input.inputs.items()
            if k not in {"tool_name", "tool_args"}
        }
        tool_args: dict[str, Any] = {}
        tool_args.update(auto_tool_args)
        tool_args.update(dynamic_tool_args)
        tool_args.update(static_tool_args)
        logger.info(
            "tool_call_start node_id=%s tool=%s input_keys=%s",
            self.node_id,
            str(tool_name),
            sorted(tool_args.keys()),
        )
        started = time.perf_counter()

        # Replay path
        if self._replay_responses is not None:
            if self._replay_index >= len(self._replay_responses):
                return NodeOutput(
                    status="FAILED",
                    error="Replay exhausted",
                )
            resp = self._replay_responses[self._replay_index]
            self._replay_index += 1
            result = NodeOutput(
                outputs=resp.get("outputs", {}),
                quality_signals=QualitySignals(
                    **resp.get("quality_signals", {})
                ),
                status=resp.get("status", "COMPLETED"),
            )
            if self._recorded_responses is not None:
                self._recorded_responses.append({
                    "outputs": result.outputs,
                    "quality_signals": result.quality_signals.model_dump(mode="json"),
                    "status": result.status,
                })
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "tool_call_end node_id=%s tool=%s replay=true status=%s elapsed_ms=%d",
                self.node_id,
                str(tool_name),
                result.status,
                elapsed_ms,
            )
            return result

        # Policy gate check (v0.2: local allowlist; approvals not interactive yet)
        require_approval = self.policy.get("require_approval_for", [])
        if require_approval:
            logger.warning(
                "McpToolNode policy gate: tool=%s requires approval for %s "
                "(auto-approved in v1 stub)",
                tool_name,
                require_approval,
            )

        try:
            outputs = invoke_local_tool(
                ToolCall(tool_name=str(tool_name), tool_args=dict(tool_args)),
                policy=self.policy,
                runtime=self.tool_runtime,
            )
            result = NodeOutput(outputs=outputs, status="COMPLETED")
        except ToolPolicyError as exc:
            result = NodeOutput(status="FAILED", error=f"Policy denied: {exc}")
        except ToolExecutionError as exc:
            result = NodeOutput(status="FAILED", error=str(exc))
        except Exception as exc:
            # Keep a stable error surface.
            result = NodeOutput(status="FAILED", error=f"Tool failure: {exc}")

        if self._recorded_responses is not None:
            self._recorded_responses.append({
                "outputs": result.outputs,
                "quality_signals": result.quality_signals.model_dump(mode="json"),
                "status": result.status,
            })

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        extra: list[str] = []
        if isinstance(result.outputs.get("path"), str) and result.outputs.get("path", "").strip():
            extra.append(f"path={result.outputs['path']}")
        if result.outputs.get("bytes_written") is not None:
            extra.append(f"bytes={result.outputs.get('bytes_written')}")
        if result.outputs.get("status_code") is not None:
            extra.append(f"http={result.outputs.get('status_code')}")
        if isinstance(result.error, str) and result.error.strip():
            extra.append(f"error={result.error}")
        logger.info(
            "tool_call_end node_id=%s tool=%s status=%s elapsed_ms=%d %s",
            self.node_id,
            str(tool_name),
            result.status,
            elapsed_ms,
            " ".join(extra).strip(),
        )

        return result
