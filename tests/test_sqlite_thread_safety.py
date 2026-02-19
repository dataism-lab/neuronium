from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from neuronium_agent.storage.sqlite_store import SqliteIndexStore
from neuronium_agent.trace.recorder import TraceRecorder


def test_sqlite_trace_recorder_thread_safe(tmp_path) -> None:
    store = SqliteIndexStore(tmp_path / "index.sqlite3", auto_migrate=True)
    rec = TraceRecorder("trace-threadsafe", store)

    def worker(i: int) -> None:
        rec.record("test", {"i": i})

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, range(50)))

    events = list(store.get_trace_events("trace-threadsafe"))
    assert len(events) == 50

