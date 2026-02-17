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

import click

from neuronium_agent.config import load_config
from neuronium_agent.api import create_runner
from neuronium_agent.types import ControlCommand, RunRequest


def _setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    fmt = (
        '{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
        if json_logs
        else "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format=fmt)


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
    default="autofix_demo",
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
def run(
    objective: str | None,
    resume_trace_id: str | None,
    runbook_id: str,
    config_path: str | None,
    mode: str | None,
    trace_export: str | None,
) -> None:
    """Start a new agent run or resume an existing one.

    To start a new run:  neuronium-agent run -o "objective"
    To resume a run:     neuronium-agent run --trace-id <id>
    """
    cli_overrides: dict = {}
    if mode:
        cli_overrides["runtime"] = {"mode": mode}

    config = load_config(config_path=config_path, cli_overrides=cli_overrides)
    _setup_logging(config.logging.level, config.logging.json_logs)

    runner = create_runner(config)

    if resume_trace_id:
        # Resume path
        click.echo(f"Resuming run: trace_id={resume_trace_id}")
        try:
            handle = runner.resume_run(resume_trace_id)
        except Exception as exc:
            click.echo(f"Resume failed: {exc}", err=True)
            sys.exit(1)
    elif objective:
        # New run path
        request = RunRequest(  # type: ignore[arg-type]
            objective=objective,
            mode=mode,
            metadata={"runbook_id": runbook_id},
        )
        click.echo(f"Starting run: {objective}")
        handle = runner.start(request)
    else:
        click.echo("Error: --objective or --trace-id is required.", err=True)
        sys.exit(1)

    status = runner.get_status(handle)

    click.echo(f"trace_id: {handle.trace_id}")
    click.echo(f"state:    {status.state}")
    if status.message:
        click.echo(f"message:  {status.message}")

    if trace_export:
        fmt = "jsonl"
        if trace_export.endswith(".json"):
            fmt = "json"
        elif trace_export.endswith(".zip"):
            fmt = "zip"
        runner.export_trace(handle, fmt, trace_export)  # type: ignore[arg-type]
        click.echo(f"Trace exported to: {trace_export}")


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
