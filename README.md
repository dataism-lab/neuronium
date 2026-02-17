# NEURONIUM Agent — OSS library + CLI

Commitment-aware AI Super Agent с планированием **Action Graph (DAG)**, гибридной памятью (GraphRAG + agentic retrieval), verification critics, typed contracts и audit/replay trace.

## Быстрый старт (локально, без внешних сервисов)

### 1. Установка

```bash
# Базовая установка (FS CAS + SQLite, OpenAI provider)
pip install -e .

# С Docker sandbox для CodeNode
pip install -e ".[docker]"

# Все extras (Postgres, Redis, pgvector, embeddings, dev)
pip install -e ".[all]"
```

### 2. Конфигурация

Создайте `neuronium.toml` в корне проекта (или используйте дефолты):

```toml
[project]
name = "neuronium"
data_dir = ".neuronium"

[determinism]
canonical_json = "neuronium-v1"
default_random_seed = 0
llm_temperature = 0.0

[storage]
blob_backend = "fs_cas"
index_backend = "sqlite"

[llm]
provider = "openai"
model = "gpt-4.1-mini"
```

### 3. API key

```bash
export NEURONIUM_OPENAI_API_KEY=sk-...
```

Альтернатива (рекомендуется для локальной разработки): создайте `.env` в корне проекта:

```dotenv
NEURONIUM_OPENAI_API_KEY=sk-...
```

CLI автоматически подхватит `.env` (не переопределяя уже заданные переменные окружения).

### 4. Запуск

```bash
# CLI
neuronium-agent run --objective "Write a fibonacci function in Python" \
    --trace-export ./trace.jsonl

# Python API
from neuronium_agent.api import create_runner
from neuronium_agent.types import RunRequest

runner = create_runner()
handle = runner.start(RunRequest(objective="Write fibonacci"))
status = runner.get_status(handle)
print(status.state)  # COMPLETED
runner.export_trace(handle, "jsonl", "trace.jsonl")
```

---

## Production: Postgres + Redis

### 1. Установка extras

```bash
pip install -e ".[postgres,redis]"
```

### 2. `neuronium.toml`

```toml
[storage]
index_backend = "postgres"
postgres_dsn = "postgresql+psycopg://user:pass@localhost:5432/mydb"
postgres_schema = "neuronium_agent"
migrations_auto_apply = true

[queue]
enabled = true
backend = "rq"
redis_url = "redis://localhost:6379/0"
queue_name = "neuronium"
```

### 3. Запуск worker

```bash
neuronium-agent worker
```

---

## Структура проекта

```
neuronium_agent/
├── api.py              # Публичный фасад: AgentRunner, create_runner
├── config.py           # Конфигурация (TOML + env + CLI)
├── types.py            # Публичные DTO
├── errors.py           # Иерархия ошибок
├── _canonical.py       # Канонический JSON, artifact ID
├── core/               # State machine, orchestrator
├── planning/           # HTN → Action Graph (DAG)
├── execution/          # Детерминированный DAG executor
├── nodes/              # ModelNode, CodeNode, McpToolNode, ...
├── storage/            # Blob + Index store (FS CAS, SQLite, Postgres)
│   └── migrations/     # SQL миграции (sqlite/, postgres/)
├── trace/              # Recorder, exporter, replay
├── verification/       # Simulated critics
├── memory/             # GraphRAG-lite (chunks + provenance, v0.2)
├── control/            # Control protocol
├── queue/              # Redis + RQ runner
└── cli/                # CLI entrypoints
tests/
├── test_canonical.py   # Canonical JSON
├── test_config.py      # Config loading
├── test_storage.py     # FS CAS + SQLite
├── test_determinism.py # Одинаковые входы → одинаковый trace
├── test_immutability.py# Артефакты не модифицируются
└── test_api.py         # Полный вертикальный срез
```

---

## Тесты

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## CLI команды

| Команда | Описание |
|---------|----------|
| `neuronium-agent run --objective "..."` | Запуск агента |
| `neuronium-agent status --trace-id ID` | Статус выполнения |
| `neuronium-agent control --trace-id ID --command pause` | Управление |
| `neuronium-agent replay --trace-id ID` | Replay (experimental) |
| `neuronium-agent worker` | Redis+RQ worker |

---

## Extras

| Extra | Зависимости | Назначение |
|-------|------------|------------|
| `[docker]` | docker | CodeNode sandbox |
| `[postgres]` | psycopg | Production index store |
| `[redis]` | redis, rq | Async queue runner |
| `[pgvector]` | pgvector | Semantic search в Postgres |
| `[embeddings]` | sentence-transformers | Локальные эмбеддинги |
| `[dev]` | pytest | Тесты |
| `[all]` | Всё вместе | Полная установка |

---

## Документация

- **Docs index**: `docs/README.md`
- **Demo walkthrough (RU)**: `docs/demos/DEMO_WALKTHROUGH_RU.md`
- **Conference demo script (RU)**: `docs/demos/CONFERENCE_DEMO_SCRIPT_RU.md`
- **Management status report (RU)**: `docs/reports/STATUS_REPORT_FOR_MANAGEMENT_RU.md`
- **Technical overview (RU)**: `docs/overview/PROJECT_TECH_OVERVIEW_RU.md`
- **Implementation Binding**: `docs/architecture/Implementation_Binding_Spec.md`
- **Roadmap**: `docs/roadmap/ROADMAP.md`
- **Roadmap status**: `docs/roadmap/ROADMAP_STATUS.md`
- **Config**: `docs/architecture/CONFIG_SPEC.md`
- **Public API**: `docs/architecture/PUBLIC_API_SPEC.md`
- **Storage schema**: `docs/architecture/STORAGE_SCHEMA_SPEC.md`
- **Presentation**: `docs/architecture/Super_Agent_presentation.md`
- **Full architecture spec**: `docs/architecture/AI_Super_Agent_Architecture_Implementation_Specification.md`
