"""Redis + RQ runner (IBS §12, PUBLIC_API_SPEC §4.2).

Optional — requires ``neuronium-agent[redis]`` extra.
"""

from __future__ import annotations

import logging
from typing import Any

from neuronium_agent.config import AppConfig
from neuronium_agent.errors import ConfigError
from neuronium_agent.types import RunHandle, RunRequest

logger = logging.getLogger(__name__)


def _get_redis_connection(config: AppConfig) -> Any:
    """Create a Redis connection from config."""
    try:
        import redis as redis_lib
    except ImportError:
        raise ConfigError(
            "redis not installed. Run: pip install neuronium-agent[redis]"
        )
    url = config.queue.redis_url
    if not url:
        raise ConfigError("queue.redis_url is required when queue is enabled")
    return redis_lib.Redis.from_url(url)


def _get_queue(config: AppConfig) -> Any:
    """Return an RQ Queue instance."""
    try:
        from rq import Queue
    except ImportError:
        raise ConfigError(
            "rq not installed. Run: pip install neuronium-agent[redis]"
        )
    conn = _get_redis_connection(config)
    return Queue(
        config.queue.queue_name,
        connection=conn,
        default_timeout=config.queue.job_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue_run(request: RunRequest, config: AppConfig) -> RunHandle:
    """Enqueue a run request into the RQ queue (PUBLIC_API_SPEC §4.2)."""
    q = _get_queue(config)
    job = q.enqueue(
        _execute_run_job,
        request.model_dump(mode="json"),
        result_ttl=config.queue.result_ttl_seconds,
    )
    from datetime import datetime, timezone

    return RunHandle(
        trace_id=job.id,
        execution_id=job.id,
        created_at=datetime.now(timezone.utc),
    )


def worker_main(config: AppConfig) -> None:
    """Start an RQ worker process (CLI ``neuronium-agent worker``)."""
    try:
        from rq import Worker
    except ImportError:
        raise ConfigError(
            "rq not installed. Run: pip install neuronium-agent[redis]"
        )
    conn = _get_redis_connection(config)
    queues = [config.queue.queue_name]
    w = Worker(queues, connection=conn)
    logger.info("Starting RQ worker on queues: %s", queues)
    w.work()


# ---------------------------------------------------------------------------
# Job function (executed by worker)
# ---------------------------------------------------------------------------

def _execute_run_job(request_data: dict) -> dict:
    """RQ job: deserialise RunRequest, run agent, return status dict."""
    from neuronium_agent.api import create_runner
    from neuronium_agent.config import load_config

    config = load_config()
    runner = create_runner(config)
    request = RunRequest(**request_data)
    handle = runner.start(request)
    status = runner.get_status(handle)
    return {
        "trace_id": handle.trace_id,
        "state": status.state,
        "message": status.message,
    }
