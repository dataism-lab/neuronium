"""CLI commands for ``neuronium-agent`` (PUBLIC_API_SPEC §6).

Commands:
  run      — Start an agent run
  status   — Check run status
  control  — Send control command
  replay   — Replay from trace (experimental)
  schema   — Export JSON Schemas (Stage 1 deliverable)
  worker   — Start Redis+RQ worker
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, TYPE_CHECKING, Callable

import click

from neuronium_agent.config import load_config
from neuronium_agent.api import create_runner
from neuronium_agent.types import ControlCommand, RunHandle, RunRequest, RunStatus

if TYPE_CHECKING:
    from neuronium_agent.api import AgentRunner


def _setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    fmt = (
        '{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
        if json_logs
        else "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)
    # Avoid noisy per-request transport logs; we log node-level lifecycle ourselves.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def _interactive_supervised_loop(
    runner: AgentRunner,
    handle: RunHandle,
) -> tuple[RunHandle, RunStatus]:
    """Handle PAUSED clarification flow in supervised mode."""
    status = runner.get_status(handle)
    while status.state == "PAUSED":
        # Ensure DB state is RUNNING before resume_run (resume requires it).
        runner.control(handle, ControlCommand(type="continue", payload={}))  # type: ignore[arg-type]
        pause_context = runner.get_latest_pause_context(handle.trace_id)
        if not pause_context:
            break
        request_artifact_id = str(
            pause_context.get("clarification_request_artifact_id", "")
        ).strip()
        if not request_artifact_id:
            break

        clarification = runner.read_artifact_json(request_artifact_id)
        questions = clarification.get("questions", [])
        if not isinstance(questions, list):
            questions = []

        click.echo("Run paused: требуется уточнение входных параметров.")
        # Optional bulk JSON shortcut (fast paste), but default UX is per-question.
        parsed: dict[str, object] = {}
        if len(questions) >= 2:
            click.echo("Можно вставить JSON-объект с ответами (Enter чтобы отвечать по одному).")
            click.echo('Пример: {"url":"https://...","doc_paths":["a.md","b.md"]}')
            raw = click.prompt("answers_json (optional)", default="", show_default=False).strip()
            if raw:
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        parsed = obj
                except Exception:
                    parsed = {}

        answers: dict[str, object] = {}
        # 1) Apply any parsed bulk answers (only for known keys)
        if parsed:
            for q in questions:
                if not isinstance(q, dict):
                    continue
                key = str(q.get("key", "")).strip()
                if key and key in parsed:
                    answers[key] = parsed[key]

        # 2) Ask remaining questions one-by-one
        for q in questions:
            if not isinstance(q, dict):
                continue
            key = str(q.get("key", "")).strip()
            if not key or key in answers:
                continue
            prompt = str(q.get("prompt", key)).strip() or key
            answer = click.prompt(prompt, default="", show_default=False).strip()
            if key in {"doc_paths", "paths"}:
                answers[key] = [p.strip() for p in answer.split(",") if p.strip()]
            elif key == "urls":
                answers[key] = [p.strip() for p in answer.split(",") if p.strip()]
            else:
                answers[key] = answer

        payload = {
            "clarification_request_artifact_id": request_artifact_id,
            "answers": answers,
        }
        runner.control(handle, ControlCommand(type="revise", payload=payload))  # type: ignore[arg-type]
        handle = runner.resume_run(handle.trace_id)
        status = runner.get_status(handle)

    return handle, status


def _print_pause_help(runner: AgentRunner, trace_id: str) -> None:
    """Best-effort print clarification questions when a run is PAUSED."""
    pause_context = runner.get_latest_pause_context(trace_id)
    if not pause_context:
        return
    request_artifact_id = str(
        pause_context.get("clarification_request_artifact_id", "")
    ).strip()
    if not request_artifact_id:
        return
    try:
        clarification = runner.read_artifact_json(request_artifact_id)
    except Exception:
        return
    questions = clarification.get("questions", [])
    if not isinstance(questions, list) or not questions:
        return
    click.echo("")
    click.echo("PAUSED: требуется уточнение параметров.")
    click.echo("Вопросы:")
    for q in questions:
        if not isinstance(q, dict):
            continue
        key = str(q.get("key", "")).strip()
        prompt = str(q.get("prompt", "")).strip()
        if key and prompt:
            click.echo(f"- {key}: {prompt}")
    click.echo("")
    click.echo("Чтобы ответить интерактивно, запусти:")
    click.echo(f"  neuronium-agent run --mode supervised --trace-id {trace_id}")


def _extract_latest_plan_payload(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("kind") != "decision":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        description = str(payload.get("description", ""))
        if description.startswith("Plan created"):
            return payload
    return None


def _extract_latest_planner_request_payload(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("kind") != "decision":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if payload.get("description") == "Planner request envelope":
            return payload
    return None


def _extract_latest_verdict_payload(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("kind") != "critic_verdict":
            continue
        payload = event.get("payload", {})
        if isinstance(payload, dict):
            return payload
    return None


def _extract_best_effort_summary(events: list[dict[str, Any]]) -> str | None:
    preferred_nodes = (
        "summarize",
        "summary",
        "draft_report",
        "write_report",
        "finalize",
    )
    for event in reversed(events):
        if event.get("kind") != "node_end":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if str(payload.get("status", "")) != "COMPLETED":
            continue
        node_id = str(payload.get("node_id", ""))
        outputs = payload.get("outputs_summary", {})
        if not isinstance(outputs, dict):
            continue
        for key in ("summary", "content", "report", "final_summary"):
            value = outputs.get(key)
            if isinstance(value, str) and value.strip() and (
                node_id in preferred_nodes or key in {"summary", "final_summary"}
            ):
                return value.strip()
    return None


def _extract_user_output_payload(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("kind") != "decision":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if payload.get("description") == "User output extracted":
            return payload
    return None


def _extract_latest_exported_file(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        if event.get("kind") != "node_end":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if str(payload.get("node_id", "")) != "export_user_output":
            continue
        if str(payload.get("status", "")) != "COMPLETED":
            continue
        outputs = payload.get("outputs_summary", {})
        if not isinstance(outputs, dict):
            continue
        value = outputs.get("path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _phase_from_boundary(boundary: str) -> str | None:
    value = str(boundary).strip().lower()
    if "after_commit" in value:
        return "commit"
    if "after_execute" in value:
        return "execute"
    if "after_control" in value:
        return "control"
    if "after_adapt" in value:
        return "adapt"
    if value == "paused":
        return "paused"
    if value == "final":
        return "final"
    return None


def _extract_phase_timeline(events: list[dict[str, Any]]) -> list[str]:
    timeline: list[str] = []
    for event in events:
        if event.get("kind") != "checkpoint":
            continue
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        resume_ctx = payload.get("resume_context", {})
        if not isinstance(resume_ctx, dict):
            continue
        phase = _phase_from_boundary(str(resume_ctx.get("phase_boundary", "")))
        if phase is None:
            continue
        if not timeline or timeline[-1] != phase:
            timeline.append(phase)
    return timeline


def _preview_text(value: Any, *, max_len: int = 50) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _preview_url(value: Any, *, max_len: int = 120) -> str:
    raw = str(value).strip()
    if not raw:
        return ""
    if len(raw) <= max_len:
        return raw
    return raw[:max_len] + "..."


def _preview_io(value: Any, *, max_len: int = 160) -> str:
    """Preview stdout/stderr safely for demo logs (single-line, truncated)."""
    if value is None:
        return ""
    text = str(value)
    # Keep it single-line for console, preserve intent.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    text = text.strip()
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _extract_best_effort_critic_summary(outputs: dict[str, Any]) -> str | None:
    """Extract a compact verdict summary from critic-like model outputs."""
    parsed = outputs.get("parsed")
    if isinstance(parsed, dict) and isinstance(parsed.get("verdict"), str):
        verdict = str(parsed.get("verdict", "UNCERTAIN")).strip() or "UNCERTAIN"
        conf = parsed.get("confidence", None)
        evidence = parsed.get("evidence", [])
        ev_n = len(evidence) if isinstance(evidence, list) else 0
        conf_sfx = f" conf={conf}" if isinstance(conf, (int, float)) else ""
        return f"{verdict}{conf_sfx} evidence={ev_n}"
    return None


def _human_step_title(*, node_id: str, node_type: str, tool_name: str | None) -> str:
    nid = node_id.strip().lower()
    tname = (tool_name or "").strip().lower()

    # Prefer semantic node_id (planner controls naming).
    if nid in {"fetch_html", "web_fetch_html"}:
        return "Fetch web page"
    if nid in {"extract_article", "web_extract_article"}:
        return "Extract article"
    if nid in {"draft_report", "write_report", "summarize", "summary"}:
        return "Generate summary"
    if nid in {"critic_report", "critic", "quality_gate"}:
        return "Critic evaluation"
    if nid in {"export_user_output", "export", "write_output"}:
        return "Write output file"

    # Fallback to tool_name hints.
    if tname.endswith("web.fetch_html"):
        return "Fetch web page"
    if tname.endswith("web.extract_article"):
        return "Extract article"
    if tname.endswith("export.write_text"):
        return "Write output file"

    if node_type == "mcp":
        return f"Run tool {tool_name or node_id}"
    if node_type == "model":
        return f"LLM call {node_id}"
    if node_type == "code":
        return f"Run code {node_id}"
    return f"Run {node_type} {node_id}"


def _extract_demo_timeline_steps(
    events: list[dict[str, Any]],
    *,
    limit: int = 20,
) -> list[str]:
    """Build a concise, human-readable execution timeline from trace events."""
    node_types: dict[str, str] = {}
    node_parameters: dict[str, dict[str, Any]] = {}

    steps: list[str] = []
    for event in events:
        kind = event.get("kind")
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue

        if kind == "node_start":
            node_id = str(payload.get("node_id", "")).strip()
            if node_id:
                node_types[node_id] = str(payload.get("node_type", "")).strip()
                params = payload.get("parameters", {})
                if isinstance(params, dict):
                    node_parameters[node_id] = params
            continue

        if kind != "node_end":
            continue

        node_id = str(payload.get("node_id", "")).strip()
        if not node_id:
            continue
        status = str(payload.get("status", "")).strip() or "UNKNOWN"
        node_type = node_types.get(node_id, str(payload.get("node_type", "")).strip())
        params = node_parameters.get(node_id, {})
        if not isinstance(params, dict):
            params = {}

        inputs = payload.get("inputs", {})
        if not isinstance(inputs, dict):
            inputs = {}
        outputs = payload.get("outputs", {})
        if not isinstance(outputs, dict):
            outputs = {}

        tool_name = None
        tool_args: dict[str, Any] = {}
        if node_type == "mcp":
            tool_name = (
                str(params.get("tool_name", "")).strip()
                or str(inputs.get("tool_name", "")).strip()
                or None
            )
            tool_args_raw = params.get("tool_args", {})
            if isinstance(tool_args_raw, dict):
                tool_args = dict(tool_args_raw)

        title = _human_step_title(node_id=node_id, node_type=node_type, tool_name=tool_name)

        elapsed_ms = payload.get("elapsed_ms")
        elapsed_part = f" ({elapsed_ms}ms)" if isinstance(elapsed_ms, int) else ""

        # Build small, demo-safe details
        detail_parts: list[str] = []
        if node_type == "mcp":
            url = (
                tool_args.get("url")
                or inputs.get("url")
                or outputs.get("final_url")
                or outputs.get("url")
            )
            if url:
                detail_parts.append(_preview_url(url))
            if isinstance(outputs.get("status_code"), (int, str)):
                detail_parts.append(f"http={outputs.get('status_code')}")
            if isinstance(outputs.get("bytes_written"), (int, str)):
                detail_parts.append(f"bytes={outputs.get('bytes_written')}")
            if isinstance(outputs.get("path"), str) and outputs.get("path", "").strip():
                detail_parts.append(f'path="{outputs["path"]}"')
        elif node_type == "model":
            qs = payload.get("quality_signals", {})
            if isinstance(qs, dict) and qs.get("tokens_used") is not None:
                detail_parts.append(f"tokens={qs.get('tokens_used')}")
            content = outputs.get("content")
            if isinstance(content, str) and content:
                detail_parts.append(f"chars={len(content)}")
            critic_summary = _extract_best_effort_critic_summary(outputs)
            if critic_summary:
                detail_parts.append(f"critic={critic_summary}")
        elif node_type == "code":
            runner = outputs.get("runner")
            if isinstance(runner, str) and runner.strip():
                detail_parts.append(f"runner={runner.strip()}")
            exit_code = outputs.get("exit_code")
            if isinstance(exit_code, (int, str)):
                detail_parts.append(f"exit={exit_code}")
            stdout = outputs.get("stdout")
            if isinstance(stdout, str) and stdout.strip():
                detail_parts.append(f'stdout="{_preview_io(stdout, max_len=90)}"')
            stderr = outputs.get("stderr")
            if isinstance(stderr, str) and stderr.strip():
                detail_parts.append(f'stderr="{_preview_io(stderr, max_len=90)}"')

        details = (" — " + " ".join(detail_parts)) if detail_parts else ""
        steps.append(f"EXECUTE: {title}{elapsed_part} [{status}]{details}")

    if len(steps) > limit:
        return steps[-limit:]
    return steps


def _build_live_demo_listener(*, verbose: bool = False) -> Callable[[dict[str, Any]], None]:
    """Return a trace event listener that prints demo timeline lines live."""
    node_types: dict[str, str] = {}
    node_parameters: dict[str, dict[str, Any]] = {}

    def on_event(ev: dict[str, Any]) -> None:
        kind = ev.get("kind")
        payload = ev.get("payload", {})
        if not isinstance(payload, dict):
            return

        if kind == "critic_verdict":
            verdict = str(payload.get("verdict", "UNCERTAIN")).strip() or "UNCERTAIN"
            conf = payload.get("confidence", None)
            evidence = payload.get("evidence", [])
            ev_n = len(evidence) if isinstance(evidence, list) else 0
            conf_sfx = f"{conf:.2f}" if isinstance(conf, (int, float)) else "n/a"
            click.echo(f"CONTROL: critic_verdict={verdict} conf={conf_sfx} evidence={ev_n}")
            if verbose and isinstance(evidence, list) and evidence:
                preview = "; ".join(_preview_text(x, max_len=80) for x in evidence[:2])
                click.echo(f"CONTROL: evidence: {preview}")
            return

        if kind == "replan":
            reason = str(payload.get("reason", "")).strip() or "n/a"
            it_from = payload.get("iteration_from", "?")
            it_to = payload.get("iteration_to", "?")
            added = payload.get("added_constraints", [])
            n_added = len(added) if isinstance(added, list) else 0
            click.echo(f"ADAPT: replan iter{it_from}->iter{it_to} reason={reason} added_constraints={n_added}")
            if verbose and isinstance(added, list) and added:
                preview = "; ".join(_preview_text(x, max_len=90) for x in added[:2])
                click.echo(f"ADAPT: constraints: {preview}")
            return

        if kind == "checkpoint":
            resume = payload.get("resume_context", {})
            if not isinstance(resume, dict):
                return
            phase = _phase_from_boundary(str(resume.get("phase_boundary", "")))
            if phase:
                click.echo(f"PHASE: {phase}")
            return

        if kind == "decision":
            desc = str(payload.get("description", "")).strip()
            if desc == "Planner request envelope":
                backend = str(payload.get("planner_backend", "")).strip()
                ver = str(payload.get("planner_backend_version", "")).strip()
                cat = str(payload.get("operator_catalog_hash", "")).strip()
                backend_sfx = f"{backend}/{ver}" if backend and ver else (backend or "n/a")
                cat_sfx = (cat[:12] + "...") if len(cat) > 12 else (cat or "n/a")
                click.echo(f"COMMIT: planner_request backend={backend_sfx} operator_catalog_hash={cat_sfx}")
            if desc.startswith("Plan created"):
                nodes = payload.get("nodes", [])
                edges = payload.get("edges", [])
                n = len(nodes) if isinstance(nodes, list) else 0
                e = len(edges) if isinstance(edges, list) else 0
                click.echo(f"PLAN: runtime DAG built (nodes={n}, edges={e})")
            if desc == "Local rendered artifact saved":
                artifact_path = str(payload.get("artifact_path", "")).strip()
                if artifact_path:
                    click.echo(f'OUTPUT: rendered_html="{artifact_path}"')
            return

        if kind == "node_start":
            node_id = str(payload.get("node_id", "")).strip()
            if node_id:
                node_types[node_id] = str(payload.get("node_type", "")).strip()
                params = payload.get("parameters", {})
                if isinstance(params, dict):
                    node_parameters[node_id] = params
            return

        if kind != "node_end":
            return

        node_id = str(payload.get("node_id", "")).strip()
        if not node_id:
            return
        status = str(payload.get("status", "")).strip() or "UNKNOWN"
        node_type = node_types.get(node_id, "unknown")
        params = node_parameters.get(node_id, {})
        if not isinstance(params, dict):
            params = {}
        outputs = payload.get("outputs", {})
        if not isinstance(outputs, dict):
            outputs = {}

        tool_name = None
        tool_args: dict[str, Any] = {}
        if node_type == "mcp":
            tool_name = str(params.get("tool_name", "")).strip() or None
            raw = params.get("tool_args", {})
            if isinstance(raw, dict):
                tool_args = dict(raw)

        title = _human_step_title(node_id=node_id, node_type=node_type, tool_name=tool_name)
        elapsed_ms = payload.get("elapsed_ms")
        elapsed_part = f" ({elapsed_ms}ms)" if isinstance(elapsed_ms, int) else ""

        detail_parts: list[str] = []
        if node_type == "mcp":
            url = tool_args.get("url") or outputs.get("final_url") or outputs.get("url")
            if url:
                detail_parts.append(_preview_url(url))
            if isinstance(outputs.get("status_code"), (int, str)):
                detail_parts.append(f"http={outputs.get('status_code')}")
            if isinstance(outputs.get("bytes_written"), (int, str)):
                detail_parts.append(f"bytes={outputs.get('bytes_written')}")
            if isinstance(outputs.get("path"), str) and outputs.get("path", "").strip():
                detail_parts.append(f'path="{outputs["path"]}"')
        elif node_type == "model":
            qs = payload.get("quality_signals", {})
            if isinstance(qs, dict) and qs.get("tokens_used") is not None:
                detail_parts.append(f"tokens={qs.get('tokens_used')}")
            content = outputs.get("content")
            if isinstance(content, str) and content:
                detail_parts.append(f"chars={len(content)}")
                if verbose and node_id in {"generate", "fix"}:
                    detail_parts.append(f'code="{_preview_io(content, max_len=90)}"')
            critic_summary = _extract_best_effort_critic_summary(outputs)
            if critic_summary:
                detail_parts.append(f"critic={critic_summary}")
        elif node_type == "code":
            runner = outputs.get("runner")
            if isinstance(runner, str) and runner.strip():
                detail_parts.append(f"runner={runner.strip()}")
            exit_code = outputs.get("exit_code")
            if isinstance(exit_code, (int, str)):
                detail_parts.append(f"exit={exit_code}")
            stdout = outputs.get("stdout")
            if isinstance(stdout, str) and stdout.strip():
                detail_parts.append(f'stdout="{_preview_io(stdout, max_len=90)}"')
            stderr = outputs.get("stderr")
            if isinstance(stderr, str) and stderr.strip():
                detail_parts.append(f'stderr="{_preview_io(stderr, max_len=90)}"')
            if verbose and isinstance(payload.get("error"), str) and payload.get("error", "").strip():
                detail_parts.append(f'error="{_preview_text(payload.get("error"), max_len=90)}"')

        details = (" — " + " ".join(detail_parts)) if detail_parts else ""
        click.echo(f"EXECUTE: {title}{elapsed_part} [{status}]{details}")

    return on_event


def _extract_tool_logs(events: list[dict[str, Any]], *, limit: int = 12) -> list[str]:
    node_types: dict[str, str] = {}
    logs: list[str] = []
    for event in events:
        kind = event.get("kind")
        payload = event.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if kind == "node_start":
            node_id = str(payload.get("node_id", "")).strip()
            node_type = str(payload.get("node_type", "")).strip()
            if node_id and node_type:
                node_types[node_id] = node_type
            continue
        if kind != "node_end":
            continue
        node_id = str(payload.get("node_id", "")).strip()
        status = str(payload.get("status", "")).strip() or "UNKNOWN"
        node_type = node_types.get(node_id, "")
        if node_type != "mcp":
            continue
        outputs = payload.get("outputs_summary", {})
        if not isinstance(outputs, dict):
            outputs = {}
        error = payload.get("error")

        parts: list[str] = [f"{node_id}: {status}"]
        if isinstance(outputs.get("path"), str) and outputs.get("path", "").strip():
            parts.append(f"path={outputs['path']}")
        if outputs.get("bytes_written") is not None:
            parts.append(f"bytes={outputs.get('bytes_written')}")
        if outputs.get("status_code") is not None:
            parts.append(f"http={outputs.get('status_code')}")
        if isinstance(outputs.get("final_url"), str) and outputs.get("final_url", "").strip():
            parts.append(f"url={outputs['final_url']}")
        if isinstance(outputs.get("warnings"), str) and outputs.get("warnings", "").strip():
            parts.append(f"warnings={_preview_text(outputs['warnings'])}")
        if isinstance(outputs.get("text"), str) and outputs.get("text", "").strip():
            parts.append(f"text[:50]={_preview_text(outputs['text'])}")
        if isinstance(error, str) and error.strip():
            parts.append(f"error={_preview_text(error)}")
        logs.append(" | ".join(parts))

    if len(logs) > limit:
        return logs[-limit:]
    return logs


def _print_summary_for_demo(*, summary_text: str | None, exported_file: str | None) -> None:
    if not summary_text:
        click.echo("final_summary: n/a")
        return
    if exported_file:
        click.echo("final_summary: omitted (written to exported_file)")
        return

    raw = summary_text.strip()
    lower = raw.lower()
    # Show in CLI only concise plain text (not file-like payloads).
    looks_like_structured = (
        lower.startswith("```")
        or "<html" in lower
        or "<!doctype html" in lower
        or raw.startswith("{")
        or raw.startswith("[")
    )
    too_long_for_cli = len(raw) > 500 or raw.count("\n") > 8
    if looks_like_structured or too_long_for_cli:
        click.echo("final_summary: omitted (non-plain or large payload)")
        return

    click.echo(f"final_summary: {raw}")


def _print_demo_report(runner: AgentRunner, trace_id: str) -> None:
    events = runner.get_trace_events(trace_id)
    planner_req = _extract_latest_planner_request_payload(events)
    plan = _extract_latest_plan_payload(events)
    verdict = _extract_latest_verdict_payload(events)
    user_out = _extract_user_output_payload(events) or {}
    summary_text = (
        str(user_out.get("summary")).strip()
        if isinstance(user_out.get("summary"), str) and user_out.get("summary").strip()
        else _extract_best_effort_summary(events)
    )
    title_text = (
        str(user_out.get("title")).strip()
        if isinstance(user_out.get("title"), str) and user_out.get("title").strip()
        else None
    )
    source_url = (
        str(user_out.get("source_url")).strip()
        if isinstance(user_out.get("source_url"), str) and user_out.get("source_url").strip()
        else None
    )
    exported_file = _extract_latest_exported_file(events)
    rendered_path = runner.get_latest_rendered_artifact_path(trace_id)
    phases = _extract_phase_timeline(events)
    steps = _extract_demo_timeline_steps(events)

    click.echo("")
    click.echo("=== demo timeline ===")

    if planner_req:
        backend = str(planner_req.get("planner_backend", "")).strip()
        ver = str(planner_req.get("planner_backend_version", "")).strip()
        cat = str(planner_req.get("operator_catalog_hash", "")).strip()
        backend_sfx = f"{backend}/{ver}" if backend and ver else (backend or "n/a")
        cat_sfx = (cat[:12] + "...") if len(cat) > 12 else (cat or "n/a")
        click.echo(f"COMMIT: planner_request backend={backend_sfx} operator_catalog_hash={cat_sfx}")

    if plan:
        nodes = plan.get("nodes", [])
        edges = plan.get("edges", [])
        planner_backend = str(plan.get("planner_backend", "")).strip()
        planner_version = str(plan.get("planner_backend_version", "")).strip()
        click.echo(
            "PLAN: runtime DAG built "
            + f"(nodes={len(nodes) if isinstance(nodes, list) else 0}, "
            + f"edges={len(edges) if isinstance(edges, list) else 0})"
        )
        if isinstance(nodes, list) and nodes:
            click.echo("PLAN: nodes: " + ", ".join(str(n) for n in nodes))
        if isinstance(edges, list) and edges:
            # Print compact edge preview for demo (avoid walls of text)
            edge_preview = ", ".join(f"{a}->{b}" for a, b in edges[:8])
            if len(edges) > 8:
                edge_preview += ", ..."
            click.echo("PLAN: edges: " + edge_preview)
        if planner_backend:
            suffix = f"/{planner_version}" if planner_version else ""
            click.echo(f"COMMIT: planner_backend={planner_backend}{suffix}")
    else:
        click.echo("PLAN: n/a")

    if title_text:
        click.echo(f'TITLE: "{title_text}"')
    if source_url:
        click.echo(f"SOURCE: {_preview_url(source_url)}")

    if verdict:
        click.echo(
            "CONTROL: critic_verdict="
            + f"{verdict.get('verdict', 'UNCERTAIN')} "
            + f"(confidence={verdict.get('confidence', 0.0)})"
        )
    else:
        click.echo("CONTROL: critic_verdict=n/a")

    if exported_file:
        click.echo(f'OUTPUT: exported_file="{exported_file}"')
    else:
        click.echo('OUTPUT: exported_file="n/a"')

    if rendered_path:
        click.echo(f'OUTPUT: rendered_html="{rendered_path}"')
    else:
        click.echo('OUTPUT: rendered_html="n/a"')

    if phases:
        click.echo("PHASES: " + " -> ".join(phases))
    else:
        click.echo("PHASES: n/a")

    if steps:
        click.echo("STEPS:")
        for line in steps:
            click.echo("  - " + line)
    else:
        click.echo("STEPS: n/a")

    # Final summary: only if concise plain text (avoid dumping HTML/JSON to console)
    _print_summary_for_demo(summary_text=summary_text, exported_file=exported_file)


@click.group()
@click.version_option(package_name="neuronium-agent")
def cli() -> None:
    """NEURONIUM Agent — commitment-aware AI Super Agent CLI."""


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--objective", "-o", default=None, help="Agent objective (required for new runs)")
@click.option(
    "--trace-id",
    "resume_trace_id",
    default=None,
    help="Trace ID of a paused run to resume",
)
@click.option(
    "--runbook",
    "runbook_id",
    default="super_agent_v0",
    show_default=True,
    help="Runbook ID (deterministic plan family) for new runs",
)
@click.option("--config", "config_path", default=None, help="Path to neuronium.toml")
@click.option(
    "--mode",
    type=click.Choice(["batch", "supervised"]),
    default=None,
    help="Execution mode",
)
@click.option("--trace-export", "trace_export", default=None, help="Export trace to path")
@click.option(
    "--summary",
    "demo_report",
    is_flag=True,
    default=False,
    help="Print execution summary (plan, verdict, artifacts) after run completes",
)
@click.option(
    "--raw-logs",
    "raw_logs",
    is_flag=True,
    default=False,
    help="Show raw logs instead of human-readable timeline",
)
@click.option(
    "--verbose", "-v",
    "demo_verbose",
    is_flag=True,
    default=False,
    help="Increase output detail level (stdout/stderr previews, critic evidence)",
)
@click.option(
    "--autofix-inject-bug/--no-autofix-inject-bug",
    "autofix_inject_bug",
    default=False,
    show_default=True,
    help="For autofix_demo: inject a deliberate runtime bug in iter1 to reliably trigger iter2 fix (demo only)",
)
@click.option(
    "--auto-clarify/--no-auto-clarify",
    "auto_clarify",
    default=True,
    show_default=True,
    help="If the run pauses for clarification, ask questions immediately in the same process (interactive terminals only)",
)
def run(
    objective: str | None,
    resume_trace_id: str | None,
    runbook_id: str,
    config_path: str | None,
    mode: str | None,
    trace_export: str | None,
    demo_report: bool,
    raw_logs: bool,
    demo_verbose: bool,
    autofix_inject_bug: bool,
    auto_clarify: bool,
) -> None:
    """Start a new agent run or resume an existing one.

    To start a new run:  neuronium-agent run -o "objective"
    To resume a run:     neuronium-agent run --trace-id <id>
    """
    cli_overrides: dict = {}
    if mode:
        cli_overrides["runtime"] = {"mode": mode}

    config = load_config(config_path=config_path, cli_overrides=cli_overrides)
    demo_live = not raw_logs
    if demo_live:
        _setup_logging("WARNING", json_logs=False)
    else:
        _setup_logging(config.logging.level, config.logging.json_logs)

    listener = _build_live_demo_listener(verbose=demo_verbose) if demo_live else None
    runner = create_runner(config, trace_event_listener=listener)

    if resume_trace_id:
        # Resume path:
        # - If PAUSED and in supervised mode, run interactive loop which will
        #   send revise+continue and then resume.
        # - If RUNNING, resume immediately.
        # - Otherwise, print status and helpful next-step hints.
        click.echo(f"Resuming run: trace_id={resume_trace_id}")
        from datetime import datetime, timezone

        handle = RunHandle(
            trace_id=resume_trace_id,
            execution_id="",
            created_at=datetime.now(timezone.utc),
        )
    elif objective:
        # New run path
        constraints: list[str] = []
        if autofix_inject_bug and runbook_id == "autofix_demo":
            constraints.append("__NEURONIUM_INTERNAL_DEMO_INJECT_BUG__")
        request = RunRequest(  # type: ignore[arg-type]
            objective=objective,
            constraints=constraints,
            mode=mode,
            metadata={"runbook_id": runbook_id},
        )
        click.echo(f"Starting run: {objective}")
        handle = runner.start(request)
    else:
        click.echo("Error: --objective or --trace-id is required.", err=True)
        sys.exit(1)

    status = runner.get_status(handle)

    # For resume runs, if state is RUNNING and we are not in supervised
    # clarification flow, resume immediately.
    if resume_trace_id and status.state == "RUNNING" and config.runtime.mode != "supervised":
        try:
            handle = runner.resume_run(handle.trace_id)
            status = runner.get_status(handle)
        except Exception as exc:
            click.echo(f"Resume failed: {exc}", err=True)
            sys.exit(1)

    if config.runtime.mode == "supervised":
        # Supervised mode handles PAUSED clarification and then resumes.
        # If the run isn't paused, it will just return current status.
        handle, status = _interactive_supervised_loop(runner, handle)
    elif auto_clarify and status.state == "PAUSED" and sys.stdin.isatty() and sys.stdout.isatty():
        # Auto-clarify even in batch mode to avoid requiring a second CLI command.
        handle, status = _interactive_supervised_loop(runner, handle)

    click.echo(f"trace_id: {handle.trace_id}")
    click.echo(f"state:    {status.state}")
    if status.message:
        click.echo(f"message:  {status.message}")
    if status.state == "PAUSED" and config.runtime.mode != "supervised":
        _print_pause_help(runner, handle.trace_id)

    if trace_export:
        fmt = "jsonl"
        if trace_export.endswith(".json"):
            fmt = "json"
        elif trace_export.endswith(".zip"):
            fmt = "zip"
        runner.export_trace(handle, fmt, trace_export)  # type: ignore[arg-type]
        click.echo(f"Trace exported to: {trace_export}")

    if demo_report:
        _print_demo_report(runner, handle.trace_id)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--trace-id", required=True, help="Trace ID")
@click.option("--config", "config_path", default=None, help="Path to neuronium.toml")
def status(trace_id: str, config_path: str | None) -> None:
    """Check run status."""
    config = load_config(config_path=config_path)
    runner = create_runner(config)

    from neuronium_agent.types import RunHandle
    from datetime import datetime, timezone

    handle = RunHandle(
        trace_id=trace_id,
        execution_id="",
        created_at=datetime.now(timezone.utc),
    )
    st = runner.get_status(handle)
    click.echo(json.dumps(st.model_dump(mode="json"), indent=2, default=str))


# ---------------------------------------------------------------------------
# control
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--trace-id", required=True, help="Trace ID")
@click.option(
    "--command",
    "cmd",
    required=True,
    type=click.Choice(["continue", "pause", "revise", "replan", "stop", "escalate"]),
    help="Control command",
)
@click.option("--payload", default="{}", help="JSON payload")
@click.option("--config", "config_path", default=None, help="Path to neuronium.toml")
def control(
    trace_id: str,
    cmd: str,
    payload: str,
    config_path: str | None,
) -> None:
    """Send a control command to a running agent."""
    config = load_config(config_path=config_path)
    runner = create_runner(config)

    from neuronium_agent.types import RunHandle
    from datetime import datetime, timezone

    handle = RunHandle(
        trace_id=trace_id,
        execution_id="",
        created_at=datetime.now(timezone.utc),
    )
    command = ControlCommand(type=cmd, payload=json.loads(payload))  # type: ignore[arg-type]
    st = runner.control(handle, command)
    click.echo(json.dumps(st.model_dump(mode="json"), indent=2, default=str))


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--trace-id", required=True, help="Trace ID to replay")
@click.option("--config", "config_path", default=None, help="Path to neuronium.toml")
def replay(trace_id: str, config_path: str | None) -> None:
    """Replay a run from recorded trace (experimental)."""
    config = load_config(config_path=config_path)
    runner = create_runner(config)

    try:
        handle = runner.replay(trace_id)
        click.echo(f"Replay started: trace_id={handle.trace_id}")
    except NotImplementedError as exc:
        click.echo(f"[experimental] {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

@cli.group()
def schema() -> None:
    """JSON Schema operations (Stage 1 deliverable)."""


@schema.command("export")
@click.option(
    "--out",
    "out_dir",
    required=True,
    type=click.Path(),
    help="Output directory for *.schema.json files",
)
def schema_export(out_dir: str) -> None:
    """Export canonical JSON Schemas for all registered Pydantic models."""
    from pathlib import Path

    from neuronium_agent.schemas.export import export_json_schemas

    target = Path(out_dir)
    written = export_json_schemas(target)
    for p in written:
        click.echo(p)
    click.echo(f"Exported {len(written)} schemas to {target}")


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--config", "config_path", default=None, help="Path to neuronium.toml")
def worker(config_path: str | None) -> None:
    """Start Redis+RQ worker (requires extras: neuronium-agent[redis])."""
    config = load_config(config_path=config_path)
    _setup_logging(config.logging.level, config.logging.json_logs)

    if not config.queue.enabled:
        click.echo("Queue is not enabled. Set [queue] enabled=true in config.", err=True)
        sys.exit(1)

    from neuronium_agent.queue.rq_runner import worker_main

    click.echo(f"Starting worker (queue={config.queue.queue_name}) ...")
    worker_main(config)
