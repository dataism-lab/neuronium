# CONFIG SPEC — NEURONIUM (v1)

Версия: v0.1  
Дата: 2026-02-10  
Статус: Binding для v1

Цель: зафиксировать **точный формат конфигурации**, **приоритеты источников**, дефолты и примеры, чтобы генерация кода через Cursor/LLM не “додумывала” значения.

Связанные документы:
- `Implementation_Binding_Spec.md` (IBS)
- `../roadmap/ROADMAP.md`

---

## 1. Формат конфигурации (binding)

### 1.1 Основной формат
- **TOML** (Python 3.11+ имеет `tomllib`).

#### Почему вообще нужен конфиг
Конфиг определяет:
- какие backend’ы использовать (локальные vs production),
- лимиты/политики (детерминизм, параллелизм, таймауты, sandbox),
- провайдеры моделей (LLM/embeddings) и их параметры,
- подключения к внешним сервисам (MCP, Redis, Postgres).

Binding: конфиг — это **вход** для `create_runner(config)` и CLI (`neuronium-agent run ...`), а не “настройки IDE”.

### 1.2 Имя файла и расположение
Конфиг читается из первого найденного пути:
1) путь из CLI флага `--config <path>`
2) `./neuronium.toml`
3) `./config/neuronium.toml`

### 1.3 Приоритеты (highest → lowest)
1) **CLI flags**
2) **Environment variables** (префикс `NEURONIUM_`)
3) **Config file** (`neuronium.toml`)
4) **Built-in defaults**

### 1.4 Секреты (binding)
- Секреты **НЕ** должны храниться в `neuronium.toml` по умолчанию.
- Секреты передаются через env vars или секрет-менеджер (future).

---

## 2. Схема конфигурации (логическая)

Ниже — логическая структура. Реализация обязана валидировать конфиг через Pydantic-модели.

### 2.1 `project`
- `name: str` (default: `"neuronium"`)
- `data_dir: str` (default: `".neuronium"`) — корневая папка данных (FS CAS, SQLite, логи)

### 2.2 `determinism`
- `canonical_json: str` (enum, default: `"neuronium-v1"`) — версия canonicalization правил
- `default_random_seed: int` (default: `0`)
- `llm_temperature: float` (default: `0.0`)
- `strict: bool` (default: `false`) — при `true` ноды с `declared_non_deterministic` отклоняются при сборке реестра (Spec §1.2.1)
- `mcp_allow_non_deterministic_tool_ids: list[str]` (default: `[]`) — при `strict=true` разрешённые идентификаторы MCP-инструментов, считающихся недетерминистичными (allowlist)

### 2.3 `runtime`
- `mode: str` (enum: `"batch" | "supervised"`, default: `"batch"`)
- `max_parallel_nodes: int` (default: `4`)
- `checkpoint_policy: str` (enum: `"on_transition" | "periodic" | "node_boundary"`, default: `"on_transition"`)
- `checkpoint_interval_seconds: int|null` (default: `null`; используется если policy=`periodic`)
- `pause_grace_period_seconds: int` (default: `30`) — grace period для паузы: дать активным нодам доработать до checkpoint (Spec §9.1.2)
- `stop_grace_period_seconds: int` (default: `5`) — grace period для cooperative stop перед принудительным завершением (Spec §6.2.4)

### 2.4 `storage`
#### 2.4.1 Blob store (default OSS)
- `blob_backend: str` (enum: `"fs_cas" | "s3"`, default: `"fs_cas"`)
- `fs_cas_root: str` (default: `"{project.data_dir}/blobs"`)

#### 2.4.2 Index/metadata store (default OSS)
- `index_backend: str` (enum: `"sqlite" | "postgres"`, default: `"sqlite"`)
- `sqlite_path: str` (default: `"{project.data_dir}/index.sqlite3"`)

#### 2.4.3 Postgres (production adapter)
- `postgres_dsn: str|null` (default: `null`)  
  Пример: `postgresql+psycopg://user:pass@host:5432/dbname`
- `postgres_schema: str` (default: `"neuronium_agent"`)
- `migrations_auto_apply: bool` (default: `true`)

### 2.5 `queue` (Redis + RQ) — optional
- `enabled: bool` (default: `false`)
- `backend: str` (enum: `"rq"`, default: `"rq"`)
- `redis_url: str|null` (default: `null`)  
  Пример: `redis://localhost:6379/0`
- `queue_name: str` (default: `"neuronium"`)
- `job_timeout_seconds: int` (default: `900`)
- `result_ttl_seconds: int` (default: `86400`)

### 2.6 `llm`
В v1 поддерживается минимум 1 провайдер; точные поля зависят от выбранного провайдера.

Общий contract:
- `provider: str` (binding v1 default: `"openai"`; расширяемо)
- `model: str` (default: `"gpt-4.1-mini"`; можно поменять)
- `base_url: str|null` (default: `null`)
- `api_key_env: str` (default: `"NEURONIUM_OPENAI_API_KEY"`)
- `structured_output: bool` (default: `true`)
- `timeout_seconds: int` (default: `60`)
- `max_retries: int` (default: `2`)

Binding: провайдеры — это настройка **ModelNode** (LLM inference). Исполнение графа (executor) от провайдера не зависит.

Примечание: значение API key берётся из env var с именем `api_key_env`.

### 2.6.1 `model_catalog` (опционально, B13)

Каталог моделей для привязки model-нод к конкретной модели (Spec §5.2.1). Если секция не задана, используется внутренний дефолт (одна запись `default` из `llm`), поведение совпадает с текущим (все ноды — `llm.model` / `llm.provider`).

- `default_model_id: str` (default: `"default"`) — идентификатор модели по умолчанию при отсутствии или недоступности выбранной по `model_id` в параметрах ноды
- `models: list[entry]` (default: `[]`)

`entry`:
- `id: str` — идентификатор записи (например `"default"`, `"gpt4"`, `"critic"`)
- `provider: str` (default: `"openai"`)
- `model: str` — имя модели у провайдера (например `"gpt-4.1-mini"`)
- `api_key_env: str|null` (default: `null`) — если задан, используется этот env для API key; иначе наследуется `llm.api_key_env`
- `base_url: str|null` (default: `null`)
- `description: str|null` (default: `null`) — описание для документации

Поведение: при сборке реестра нод для каждой model-ноды читается `parameters.model_id`. Если в графе не задан `model_id`, используется дефолт стадии runbook (`ActionGraphStage.default_model_id`), если он задан. По итоговому id ищется запись в каталоге; если модель «доступна» (в env задан непустой ключ для `api_key_env`), используется она; иначе — запись с `default_model_id`; если и она недоступна — fallback на `llm.model` / `llm.provider` / `llm.api_key_env`.

Env (при необходимости): `NEURONIUM_MODEL_CATALOG_DEFAULT_MODEL_ID` для переопределения дефолтной модели каталога (если в TOML задана секция `model_catalog`).

### 2.7 `mcp`
- `enabled: bool` (default: `true`)
- `servers: list[server]` (default: `[]`)

`server`:
- `name: str`
- `url: str` (или launch config — future)
- `timeout_seconds: int` (default: `60`)
- `rate_limit_rps: float|null` (default: `null`)
- `policy`:
  - `fs_roots_allowlist: list[str]` (default: `[]`)
  - `network_allowlist: list[str]` (default: `[]`)
  - `require_approval_for: list[str]` (default: `["destructive", "exfiltration_risk", "high_cost"]`)

### 2.8 `code_node` (Docker binding)
- `enabled: bool` (default: `true`)
- `runtime: str` (binding: `"python"`)
- `docker`:
  - `enabled: bool` (binding: `true`)
  - `image: str` (default: `"python:3.11-slim"`, v1 можно уточнить позже)
  - `network_enabled: bool` (default: `false`)
  - `cpu_limit: str|null` (пример: `"1.0"`, default: `null`)
  - `mem_limit: str|null` (пример: `"512m"`, default: `null`)
  - `timeout_seconds: int` (default: `120`)
  - `fs_roots_allowlist: list[str]` (default: `[]`)

### 2.9 `memory`
- `enabled: bool` (default: `true`)
- `graphrag_backend: str` (enum: `"sqlite" | "postgres"`, default: `"sqlite"`)
- `semantic_search`:
  - `enabled: bool` (default: `false` для лёгкой установки)
  - `backend: str` (enum: `"pgvector" | "local"`, default: `"local"`)
  - `pgvector`:
    - `enabled: bool` (default: `false`)
    - `vector_dim: int` (default: `1536`) — зависит от embedding model
  - `local` (рекомендованный OSS путь):
    - `enabled: bool` (default: `false`)
    - `embedding_provider: str` (enum: `"openai" | "sentence_transformers"`, default: `"sentence_transformers"`)
    - `model: str` (default: `"sentence-transformers/all-MiniLM-L6-v2"`)
    - `vector_dim: int|null` (default: `null`) — если `null`, берём из модели
    - `index: str` (enum: `"bruteforce" | "hnsw"`, default: `"bruteforce"`)
    - `store_in_sqlite: bool` (default: `true`)

Binding: если `semantic_search.enabled=false`, запросы `mode=semantic` деградируют в keyword/structured с warning в trace.

### 2.10 `logging`
- `level: str` (enum: `"DEBUG"|"INFO"|"WARNING"|"ERROR"`, default: `"INFO"`)
- `json: bool` (default: `true`)
- `path: str` (default: `"{project.data_dir}/logs/neuronium.jsonl"`)

### 2.11 `recovery`
Политика повторов и эскалации (B1 Part 1/2, B2 verdict local fix).
- `max_node_retries: int` (default: `3`)
- `max_stage_retries: int` (default: `2`)
- `retry_backoff_base_seconds: float` (default: `1.0`)
- `retry_count_upgrade_threshold: int` (default: `2`) — после стольких повторов ноды классификация → PERSISTENT
- `repeated_rollback_threshold: int` (default: `3`) — при стольких повторных сбоях одной ноды → escalate
- `allow_auto_replan: bool` (default: `false`) — при `true` политика может вернуть REPLAN вместо ESCALATE
- `max_verdict_fix_attempts: int` (default: `1`) — макс. повторов стадии с контекстом verdict fix (gaps/suggestions)

Env (при необходимости): `NEURONIUM_RECOVERY_MAX_NODE_RETRIES`, `NEURONIUM_RECOVERY_MAX_STAGE_RETRIES`, `NEURONIUM_RECOVERY_MAX_VERDICT_FIX_ATTEMPTS`.

---

## 3. Environment variables (binding)

Префикс: `NEURONIUM_`

Примеры:
- `NEURONIUM_PROJECT_DATA_DIR`
- `NEURONIUM_STORAGE_INDEX_BACKEND=postgres`
- `NEURONIUM_STORAGE_POSTGRES_DSN=...`
- `NEURONIUM_QUEUE_ENABLED=true`
- `NEURONIUM_QUEUE_REDIS_URL=redis://...`
- `NEURONIUM_OPENAI_API_KEY=...` (или другое имя, указанное в `llm.api_key_env`)
- `NEURONIUM_RECOVERY_MAX_NODE_RETRIES`, `NEURONIUM_RECOVERY_MAX_STAGE_RETRIES`, `NEURONIUM_RECOVERY_MAX_VERDICT_FIX_ATTEMPTS`

---

## 4. Пример `neuronium.toml` (OSS default)

```toml
[project]
name = "neuronium"
data_dir = ".neuronium"

[determinism]
canonical_json = "neuronium-v1"
default_random_seed = 0
llm_temperature = 0.0

[runtime]
mode = "batch"
max_parallel_nodes = 4
checkpoint_policy = "on_transition"

[storage]
blob_backend = "fs_cas"
fs_cas_root = ".neuronium/blobs"
index_backend = "sqlite"
sqlite_path = ".neuronium/index.sqlite3"

[queue]
enabled = false

[mcp]
enabled = true
servers = []

[code_node]
enabled = true
runtime = "python"

[code_node.docker]
enabled = true
image = "python:3.11-slim"
network_enabled = false
timeout_seconds = 120
fs_roots_allowlist = ["./"]

[memory]
enabled = true
graphrag_backend = "sqlite"

[memory.semantic_search]
enabled = false
backend = "local"

[memory.semantic_search.local]
enabled = false
embedding_provider = "sentence_transformers"
model = "sentence-transformers/all-MiniLM-L6-v2"
index = "bruteforce"
store_in_sqlite = true

[logging]
level = "INFO"
json = true
path = ".neuronium/logs/neuronium.jsonl"
```

---

## 5. Пример (production)

```toml
[storage]
blob_backend = "fs_cas"
fs_cas_root = "/var/lib/neuronium/blobs"
index_backend = "postgres"
postgres_dsn = "postgresql+psycopg://user:pass@localhost:5432/mydb"
postgres_schema = "neuronium_agent"
migrations_auto_apply = true

[queue]
enabled = true
backend = "rq"
redis_url = "redis://localhost:6379/0"
queue_name = "neuronium"
job_timeout_seconds = 1800

[memory.semantic_search]
enabled = true
backend = "pgvector"

[memory.semantic_search.pgvector]
enabled = true
vector_dim = 1536
```

