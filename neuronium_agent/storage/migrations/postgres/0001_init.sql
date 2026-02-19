-- Migration 0001: initial schema for Postgres (STORAGE_SCHEMA_SPEC §3)

CREATE TABLE IF NOT EXISTS runs (
    trace_id             TEXT PRIMARY KEY,
    execution_id         TEXT NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL,
    state                TEXT NOT NULL,
    objective            TEXT NOT NULL,
    config_snapshot_json JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id          TEXT PRIMARY KEY,
    artifact_type        TEXT NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL,
    produced_by_node_ref TEXT NOT NULL,
    inputs_json          JSONB NOT NULL,
    quality_signals_json JSONB NOT NULL,
    blob_key             TEXT NOT NULL,
    media_type           TEXT NOT NULL,
    size_bytes           BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_created_at ON artifacts(created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_type ON artifacts(artifact_type);

CREATE TABLE IF NOT EXISTS lineage_edges (
    parent_artifact_id TEXT NOT NULL,
    child_artifact_id  TEXT NOT NULL,
    kind               TEXT NOT NULL,
    created_at         TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (parent_artifact_id, child_artifact_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage_edges(child_artifact_id);

CREATE TABLE IF NOT EXISTS trace_events (
    event_id       BIGSERIAL PRIMARY KEY,
    trace_id       TEXT NOT NULL,
    ts             TIMESTAMPTZ NOT NULL,
    span_id        TEXT,
    parent_span_id TEXT,
    kind           TEXT NOT NULL,
    payload_json   JSONB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trace_events_trace_ts ON trace_events(trace_id, ts);
CREATE INDEX IF NOT EXISTS idx_trace_events_kind ON trace_events(kind);

CREATE TABLE IF NOT EXISTS node_executions (
    node_execution_id TEXT PRIMARY KEY,
    trace_id          TEXT NOT NULL,
    node_ref          TEXT NOT NULL,
    attempt           INTEGER NOT NULL,
    status            TEXT NOT NULL,
    started_at        TIMESTAMPTZ,
    ended_at          TIMESTAMPTZ,
    inputs_json       JSONB NOT NULL,
    outputs_json      JSONB,
    error_json        JSONB
);
CREATE INDEX IF NOT EXISTS idx_node_exec_trace ON node_executions(trace_id);
CREATE INDEX IF NOT EXISTS idx_node_exec_node_ref ON node_executions(node_ref);

CREATE TABLE IF NOT EXISTS critic_evaluations (
    critic_evaluation_id TEXT PRIMARY KEY,
    trace_id             TEXT NOT NULL,
    ts                   TIMESTAMPTZ NOT NULL,
    input_json           JSONB NOT NULL,
    verdict_json         JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_chunks (
    chunk_id            TEXT PRIMARY KEY,
    source_artifact_id  TEXT NOT NULL,
    text                TEXT NOT NULL,
    metadata_json       JSONB NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_chunks_source ON memory_chunks(source_artifact_id);

CREATE TABLE IF NOT EXISTS memory_embeddings (
    chunk_id        TEXT PRIMARY KEY,
    embedding       JSONB NOT NULL,
    vector_dim      INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL
);
