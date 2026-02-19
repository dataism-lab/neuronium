# STORAGE SCHEMA & MIGRATIONS SPEC — NEURONIUM (v1)

Версия: v0.1  
Дата: 2026-02-10  
Статус: Binding для v1

Цель: зафиксировать **layout blob store**, **схему SQLite/PG**, и **модель миграций**, чтобы реализация была предсказуемой и совместимой с IBS (immutability, lineage, replay).

---

## 1. Blob store (content-addressed) — FS CAS (binding)

### 1.1 Корень
`fs_cas_root` из `CONFIG_SPEC.md`.

### 1.2 Layout (binding)
Артефакты сохраняются по Artifact ID. Для равномерного распределения используем шардирование по префиксу:

- `/<root>/sha256/<p1><p2>/<p3><p4>/<artifact_id>.blob`
- `/<root>/sha256/<p1><p2>/<p3><p4>/<artifact_id>.meta.json`

Где `p1..p4` — первые 4 символа hex/base58-представления (реализация должна быть стабильной).

### 1.3 Meta файл
`*.meta.json` хранит:
- `artifact_id`
- `media_type` (например `application/json`)
- `size_bytes`
- `created_at`
- `content_hash` (дублируется для проверки целостности)

Binding: blob store **не поддерживает update/delete** (только create/read; delete — только через retention policy в будущем).

---

## 2. Index store — SQLite (default OSS) (binding)

### 2.1 Общие правила
- SQLite хранит **метаданные/индексы/события**, а не тяжёлые blob’ы.
- JSON поля храним как `TEXT` (канонический JSON).
- Все записи append-only, где возможно (особенно trace events).

### 2.2 Таблицы (v1 minimum)

#### 2.2.1 `schema_version`
- `version INTEGER PRIMARY KEY`
- `applied_at TEXT NOT NULL`

#### 2.2.2 `runs`
- `trace_id TEXT PRIMARY KEY`
- `execution_id TEXT NOT NULL`
- `created_at TEXT NOT NULL`
- `state TEXT NOT NULL` (`PENDING|RUNNING|PAUSED|COMPLETED|FAILED|CANCELLED`)
- `objective TEXT NOT NULL`
- `config_snapshot_json TEXT NOT NULL` (канонический JSON)

Index:
- `idx_runs_created_at(created_at)`

#### 2.2.3 `artifacts`
- `artifact_id TEXT PRIMARY KEY`
- `artifact_type TEXT NOT NULL`
- `created_at TEXT NOT NULL`
- `produced_by_node_ref TEXT NOT NULL`
- `inputs_json TEXT NOT NULL` (список parent IDs)
- `quality_signals_json TEXT NOT NULL`
- `blob_key TEXT NOT NULL` (для FS CAS = относительный путь или `artifact_id`)
- `media_type TEXT NOT NULL`
- `size_bytes INTEGER NOT NULL`
- `deprecated_at TEXT NULL` (логическое устаревание при rollback; добавляется миграцией 0002)

Index:
- `idx_artifacts_created_at(created_at)`
- `idx_artifacts_type(artifact_type)`

#### 2.2.4 `lineage_edges`
- `parent_artifact_id TEXT NOT NULL`
- `child_artifact_id TEXT NOT NULL`
- `kind TEXT NOT NULL` (например `producedFrom|transformedFrom`)
- `created_at TEXT NOT NULL`
PRIMARY KEY: `(parent_artifact_id, child_artifact_id, kind)`

Index:
- `idx_lineage_child(child_artifact_id)`

#### 2.2.5 `trace_events`
Append-only события исполнения (основа для replay).
- `event_id INTEGER PRIMARY KEY AUTOINCREMENT`
- `trace_id TEXT NOT NULL`
- `ts TEXT NOT NULL` (ISO8601)
- `span_id TEXT`
- `parent_span_id TEXT`
- `kind TEXT NOT NULL` (например `decision|node_start|node_end|tool_call|critic_eval|error|checkpoint`)
- `payload_json TEXT NOT NULL` (канонический JSON)

Index:
- `idx_trace_events_trace_ts(trace_id, ts)`
- `idx_trace_events_kind(kind)`

#### 2.2.6 `node_executions`
Для статусов/ретраев:
- `node_execution_id TEXT PRIMARY KEY`
- `trace_id TEXT NOT NULL`
- `node_ref TEXT NOT NULL`
- `attempt INTEGER NOT NULL`
- `status TEXT NOT NULL` (`PENDING|READY|RUNNING|COMPLETED|FAILED|TIMEOUT|RETRYING|CANCELLED`)
- `started_at TEXT`
- `ended_at TEXT`
- `inputs_json TEXT NOT NULL`
- `outputs_json TEXT`
- `error_json TEXT`

Index:
- `idx_node_exec_trace(trace_id)`
- `idx_node_exec_node_ref(node_ref)`

#### 2.2.7 `critic_evaluations`
- `critic_evaluation_id TEXT PRIMARY KEY`
- `trace_id TEXT NOT NULL`
- `ts TEXT NOT NULL`
- `input_json TEXT NOT NULL`
- `verdict_json TEXT NOT NULL`

#### 2.2.8 `memory_chunks` (v1 optional, нужен для local embeddings)
Если включён local embeddings backend (см. `CONFIG_SPEC.md`), SQLite хранит chunks и embeddings.

- `chunk_id TEXT PRIMARY KEY`
- `source_artifact_id TEXT NOT NULL`
- `text TEXT NOT NULL`
- `metadata_json TEXT NOT NULL`
- `created_at TEXT NOT NULL`

Index:
- `idx_memory_chunks_source(source_artifact_id)`

#### 2.2.9 `memory_embeddings` (v1 optional, local embeddings)
- `chunk_id TEXT PRIMARY KEY` (FK → `memory_chunks.chunk_id`)
- `vector_json TEXT NOT NULL` (канонический JSON массива float; v1 bruteforce)
- `vector_dim INTEGER NOT NULL`
- `embedding_model TEXT NOT NULL`
- `created_at TEXT NOT NULL`

Binding: v1 использует **bruteforce top-k** по cosine similarity (в коде), без специальных индексов.

---

## 3. Index store — Postgres (production adapter) (binding)

### 3.1 Schema namespace
Все таблицы создаются в schema:
- `neuronium_agent` (по умолчанию; конфигурируемо).

### 3.2 Типы полей
- JSON — `JSONB`
- timestamps — `TIMESTAMPTZ`
- IDs — `TEXT` (до фиксации формата можно TEXT)

### 3.3 Таблицы
Структурно эквивалентны SQLite, но:
- `trace_events.payload` = JSONB
- индексы по JSONB допускаются (если нужно)

### 3.4 pgvector (опционально)
Если включён `memory.semantic_search.pgvector.enabled=true`:
- требуется `CREATE EXTENSION vector;` (опционально, не обязателен)
- таблица `memory_embeddings`:
  - `chunk_id TEXT PRIMARY KEY`
  - `embedding vector(<dim>) NOT NULL`
  - `metadata JSONB NOT NULL`
  - индекс `ivfflat/hnsw` (в зависимости от версии/настройки)

Binding: если extension недоступен, система должна:
- либо отключить semantic backend и деградировать,
- либо требовать явной настройки альтернативы (future).

### 3.5 Local embeddings в production (не рекомендуется по умолчанию)
Local embeddings backend в production допускается, но по умолчанию рекомендуем:
- pgvector (если доступен) или внешний vector store (future).

---

## 4. Migrations (binding)

### 4.1 Подход
В v1 используется **встроенный мигратор** без тяжёлых зависимостей:
- `schema_version` хранит текущую версию
- миграции — это последовательные SQL файлы:
  - `neuronium_agent/storage/migrations/sqlite/0001_init.sql`, `0002_artifact_deprecated_at.sql`
  - `neuronium_agent/storage/migrations/postgres/0001_init.sql`, `0002_artifact_deprecated_at.sql`

### 4.2 Правила
- миграции **идемпотентны** (или гарантировано применяются один раз по версии)
- применение в старте приложения (если `migrations_auto_apply=true`)
- все изменения схемы совместимы с immutability/append-only принципами (никаких destructive UPDATE данных артефактов)

---

## 5. Retention / Cleanup (v1 минимально)

В v1 допускается только ручная очистка (команда CLI в будущем).
Binding: автоматическое удаление артефактов и trace events **не делаем** до появления политики retention, чтобы не нарушить replay.

