# PUBLIC API SPEC — NEURONIUM (v1)

Версия: v0.1  
Дата: 2026-02-10  
Статус: Binding для v1

Цель: зафиксировать **публичный API** библиотеки и CLI, чтобы кодогенерация не создавала несовместимые/случайные интерфейсы.

---

## 1. Пакет и нейминг (binding)

Python package: `neuronium_agent`

Public modules (v1):
- `neuronium_agent.api`
- `neuronium_agent.config`
- `neuronium_agent.errors`
- `neuronium_agent.types`

Internal-only (не считается public API):
- `neuronium_agent.core.*`
- `neuronium_agent.planning.*`
- `neuronium_agent.execution.*`
- `neuronium_agent.nodes.*`
- `neuronium_agent.storage.*`
- `neuronium_agent.memory.*`
- `neuronium_agent.verification.*`
- `neuronium_agent.trace.*`

---

## 2. Типы публичного API (binding)

Все типы — Pydantic v2 модели (или `dataclass` только если явно указано).

### 2.1 Основные DTO
- `RunRequest`
  - `objective: str`
  - `constraints: list[str] = []`
  - `mode: Literal["batch","supervised"] | None`
  - `metadata: dict[str, Any] = {}`

- `RunHandle`
  - `trace_id: str`
  - `execution_id: str`
  - `created_at: datetime`

- `RunStatus`
  - `state: Literal["PENDING","RUNNING","PAUSED","COMPLETED","FAILED","CANCELLED"]`
  - `progress: float | None` (0..1)
  - `current_node_ref: str | None`
  - `message: str | None`

- `ControlCommand`
  - `type: Literal["continue","pause","revise","replan","stop"]`
  - `payload: dict[str, Any] = {}`

### 2.2 Trace/export
- `TraceExportFormat = Literal["jsonl","json","zip"]`

---

## 3. Главный фасад библиотеки (binding)

### 3.1 `AgentRunner`
Публичный класс, через который внешние приложения запускают агент.

Обязательные методы:
- `start(request: RunRequest) -> RunHandle`
- `get_status(handle: RunHandle) -> RunStatus`
- `control(handle: RunHandle, command: ControlCommand) -> RunStatus`
- `export_trace(handle: RunHandle, format: TraceExportFormat, path: str) -> None`

Опционально (v1 может быть stub):
- `replay(trace_id: str) -> RunHandle`

Binding: `AgentRunner` не должен требовать Postgres/Redis для базового использования.

### 3.2 Конструирование runner’а
Создание runner’а выполняется через фабрику:
- `create_runner(config: AppConfig) -> AgentRunner`

где `AppConfig` — pydantic-модель из `neuronium_agent.config`.

---

## 4. Storage/Queue adapters (public interfaces)

Чтобы production-интеграции не ломали API, минимальные интерфейсы считаются public:

### 4.1 Storage backends
- `BlobStore` (content-addressed)
  - `put(artifact_id: str, blob_bytes: bytes, media_type: str) -> None`
  - `get(artifact_id: str) -> bytes`
  - `exists(artifact_id: str) -> bool`

- `IndexStore`
  - `record_artifact_metadata(...)`
  - `record_lineage_edge(parent_id: str, child_id: str, kind: str) -> None`
  - `append_trace_event(trace_id: str, event: dict[str, Any]) -> None`
  - `get_trace_events(trace_id: str) -> Iterable[dict[str, Any]]`

Binding: SQLite и Postgres реализации должны быть взаимозаменяемыми через эти интерфейсы.

### 4.2 Queue runner (Redis + RQ)
Public “runner mode” API:
- `enqueue_run(request: RunRequest) -> RunHandle`
- `worker_main() -> None` (CLI entry)

В v1 допускается, что queue API находится в `neuronium_agent.api` как подмодуль `queue`.

---

## 5. Ошибки (binding)

Все публичные ошибки наследуются от:
- `NeuroniumError(Exception)`

Ключевые категории:
- `ConfigError(NeuroniumError)`
- `ValidationError(NeuroniumError)` (обёртка над pydantic/jsonschema boundary)
- `StorageError(NeuroniumError)`
- `McpError(NeuroniumError)`
- `SandboxError(NeuroniumError)`
- `ReplayError(NeuroniumError)`

Binding: ошибки должны быть **детерминированно сериализуемы** в trace (type + message + classification + node_ref/trace_id).

---

## 6. CLI (binding)

CLI entrypoint: `neuronium-agent`

Команды v1:
- `run --objective "..."`
  - `--config <path>`
  - `--mode batch|supervised`
  - `--trace-export <path>`
- `status --trace-id <id>`
- `control --trace-id <id> --command continue|pause|revise|replan|stop [--payload <json>]`
- `replay --trace-id <id>` (может быть experimental/stub)
- `worker` (если включён Redis+RQ)

Binding: все CLI команды должны писать события в trace и быть совместимыми с `Control Protocol` из архитектурной спеки.

---

## 7. SemVer и политика совместимости (binding)

До `v1.0.0` допускаются breaking changes.
После `v1.0.0`:
- public API (`neuronium_agent.api|config|errors|types`) — **SemVer**,
- internal modules могут меняться свободно.

