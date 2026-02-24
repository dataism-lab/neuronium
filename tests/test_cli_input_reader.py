"""Tests for non-blocking stdin reader and _input_reader_loop (BUG-1 fix).

Validates that the background input-reader thread exits promptly when
``stop_event`` is set, eliminating the stdin race condition with
``click.prompt()`` in the clarification flow.
"""

from __future__ import annotations

import queue
import threading
import time

from neuronium_agent.cli.main import _input_reader_loop, _read_stdin_line_nonblocking


def test_read_stdin_line_nonblocking_returns_none_immediately_on_preset_stop() -> None:
    """When stop_event is already set, function must return None without delay."""
    stop = threading.Event()
    stop.set()

    t0 = time.monotonic()
    result = _read_stdin_line_nonblocking(stop, poll_interval=0.15)
    elapsed = time.monotonic() - t0

    assert result is None
    assert elapsed < 0.5, f"Should return instantly, took {elapsed:.2f}s"


def test_read_stdin_line_nonblocking_returns_none_when_stop_set_after_delay() -> None:
    """Function must return None within ~poll_interval after stop_event is set."""
    stop = threading.Event()

    result_holder: list[str | None] = []

    def reader() -> None:
        result_holder.append(_read_stdin_line_nonblocking(stop, poll_interval=0.05))

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    time.sleep(0.15)
    stop.set()
    t.join(timeout=2.0)

    assert not t.is_alive(), "Reader thread should have exited"
    assert result_holder == [None]


def test_input_reader_loop_exits_on_stop_event() -> None:
    """_input_reader_loop must exit promptly when stop_event fires."""
    cmd_queue: queue.Queue[str | None] = queue.Queue()
    stop = threading.Event()

    t = threading.Thread(target=_input_reader_loop, args=(cmd_queue, stop), daemon=True)
    t.start()

    time.sleep(0.3)
    stop.set()
    t.join(timeout=2.0)

    assert not t.is_alive(), (
        "_input_reader_loop should exit within ~poll_interval after stop_event"
    )


def test_input_reader_loop_does_not_enqueue_after_stop() -> None:
    """No commands should appear in the queue after stop_event is set."""
    cmd_queue: queue.Queue[str | None] = queue.Queue()
    stop = threading.Event()
    stop.set()

    t = threading.Thread(target=_input_reader_loop, args=(cmd_queue, stop), daemon=True)
    t.start()
    t.join(timeout=2.0)

    assert cmd_queue.empty(), "No commands should be enqueued when stop is pre-set"
