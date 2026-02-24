from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from neuronium_agent.cli import main as cli_main
from neuronium_agent.types import ControlCommand, RunHandle, RunRequest, RunStatus


def _make_handle() -> RunHandle:
    return RunHandle(
        trace_id="trace-bug5",
        execution_id="exec-bug5",
        created_at=datetime.now(timezone.utc),
    )


class _SupervisedLoopRunnerStub:
    def __init__(
        self,
        *,
        pause_context: dict[str, Any] | None,
        initial_status: str = "PAUSED",
    ) -> None:
        self._status = RunStatus(state=initial_status)
        self._pause_context = pause_context
        self.commands: list[str] = []

    def get_status(self, _handle: RunHandle) -> RunStatus:
        return self._status

    def get_latest_pause_context(self, _trace_id: str) -> dict[str, Any] | None:
        return self._pause_context

    def control(self, _handle: RunHandle, command: ControlCommand) -> RunStatus:
        self.commands.append(command.type)
        if command.type == "continue":
            self._status = RunStatus(state="RUNNING")
        elif command.type == "revise":
            self._status = RunStatus(state="PAUSED")
        return self._status

    def read_artifact_json(self, _artifact_id: str) -> dict[str, Any]:
        return {
            "questions": [
                {
                    "key": "url",
                    "prompt": "URL",
                }
            ]
        }

    def resume_run(self, _trace_id: str) -> RunHandle:
        self._status = RunStatus(state="COMPLETED")
        return _make_handle()


def test_supervised_loop_skips_blind_continue_for_non_clarification_pause() -> None:
    runner = _SupervisedLoopRunnerStub(pause_context={"escalation_reason": "critic_fail"})
    handle = _make_handle()

    _out_handle, out_status = cli_main._interactive_supervised_loop(runner, handle)

    assert out_status.state == "PAUSED"
    assert runner.commands == []


def test_supervised_loop_sends_continue_only_for_clarification_pause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _SupervisedLoopRunnerStub(
        pause_context={"clarification_request_artifact_id": "aid-clarify-1"}
    )
    handle = _make_handle()
    monkeypatch.setattr(cli_main.click, "prompt", lambda *args, **kwargs: "https://example.com")

    _out_handle, out_status = cli_main._interactive_supervised_loop(runner, handle)

    assert out_status.state == "COMPLETED"
    assert runner.commands == ["continue", "revise"]


class _InteractiveLoopRunnerStub:
    def __init__(self) -> None:
        self.handle = _make_handle()
        self._status_calls = 0

    def start(self, _request: RunRequest, on_handle_ready=None) -> RunHandle:
        if on_handle_ready is not None:
            on_handle_ready(self.handle)
        return self.handle

    def resume_run(self, _trace_id: str) -> RunHandle:
        return self.handle

    def get_status(self, _handle: RunHandle) -> RunStatus:
        self._status_calls += 1
        if self._status_calls == 1:
            return RunStatus(state="PAUSED")
        return RunStatus(state="RUNNING")

    def get_latest_pause_context(self, _trace_id: str) -> dict[str, Any] | None:
        return {"clarification_request_artifact_id": "aid-clarify-2"}

    def control(self, _handle: RunHandle, _command: ControlCommand) -> RunStatus:
        return RunStatus(state="RUNNING")


def test_interactive_loop_refreshes_status_after_supervised_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _InteractiveLoopRunnerStub()

    def stale_supervised_loop(_runner: Any, handle: RunHandle) -> tuple[RunHandle, RunStatus]:
        return handle, RunStatus(state="PAUSED")

    monkeypatch.setattr(cli_main, "_interactive_supervised_loop", stale_supervised_loop)

    _handle, status = cli_main._interactive_run_loop(
        runner,
        request=RunRequest(objective="test"),
    )

    assert status.state == "RUNNING"
