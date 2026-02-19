# Implementation Binding Spec (IBS) — NEURONIUM (AI Super Agent)

Версия: **v0.1 (draft)**  
Дата: 2026-02-10  
Статус: **Binding** для реализации v1 (если не помечено как “Future/Extension”)

Документ фиксирует **необсуждаемые технические решения** (язык/контракты/форматы/хранилища/интеграции), чтобы генерация кода через LLM/Cursor была **детерминированной** и без “архитектурных выборов на лету”.

Основание: `AI_Super_Agent_Architecture_Implementation_Specification.md` + `Super_Agent_presentation.md`.

---

## 1. Scope v1 (что делаем обязательно)

### 1.1 Минимально жизнеспособные компоненты (MVP)
- **Cognitive Core** с явным `AgentState` и циклом **Commit → Execute → Control → Adapt**.
- **Planning**: HTN → **Action Graph (DAG)**.
- **Execution**: детерминированный DAG executor + статусы узлов + retry/timeout/failure classification.
- **Node system**: `ModelNode`, `McpToolNode`, `CodeNode` (Python runtime).
- **Verification layer**: simulated critic (минимально функциональный, но по контрактам спеки).
- **Audit/Trace**: запись решений/доказательств/результатов + replay-совместимость.
- **MCP integration**: capability discovery, sandbox/policy gates, audit logging.
- **CLI runtime**: локальный запуск (batch + supervised) и экспорт трейса.

### 1.2 “Production-grade” как цель, но без тяжёлых зависимостей по умолчанию
- Базовая установка должна работать “из коробки” **без Postgres/Redis**.
- Production-инфраструктура подключается через **опциональные адаптеры** и **extras**.

---

## 2. Language & Runtime Binding

### 2.1 Основной язык реализации
- **Python 3.11+** — основной язык core library и CLI.

### 2.2 Типизация и контракты
- Модели/контракты: **Pydantic v2** как “source of truth”.
- Валидация на границах: **JSON Schema**, сгенерированная из pydantic-моделей.
- Все межузловые сообщения — **typed JSON** (схемы обязательны).

### 2.4 Model provider binding (v1)
- v1 default LLM provider: **OpenAI** (через `ModelNode` provider adapter).
- Архитектура должна поддерживать **несколько моделей** (и потенциально нескольких провайдеров) через:
  - registry “capability-based matching” (model type, context window, structured output),
  - возможность биндить конкретный `ModelNode` к конкретной модели/провайдеру.

Binding: executor/планировщик **не зависят** от провайдера; провайдер влияет только на поведение `ModelNode`.

### 2.3 Code Node (v1)
- `CodeNode` исполняет **только Python** в v1.
- Исполнение **только в Docker** (см. раздел 7).
- Поддержка других runtime (`sql`, `node`, …) — **Future/Extension** (контракт раннера фиксируем заранее, реализацию — позже).

---

## 3. Determinism & Replay Binding (инварианты)

### 3.1 Канонический JSON (canonicalization)
Везде используется единый алгоритм canonical JSON:
- сортировка ключей объектов,
- запрет `NaN/Infinity`,
- нормализация чисел (стабильная сериализация),
- стабильная кодировка (UTF-8),
- без “плавающих” представлений.

**Binding**: canonicalization используется для:
- вычисления Artifact ID,
- записи/подписывания blob’ов,
- сравнения при replay.

### 3.2 Артефакты неизменяемы
- Любое “обновление” создаёт **новый артефакт** с новым ID и lineage-ребром.

### 3.3 Replay (запись недетерминированных входов)
Для всего внешнего/стохастического:
- LLM ответы,
- ответы инструментов (MCP),
- сетевые вызовы,

— система **записывает вход/выход** в trace/artifact storage, чтобы повторный прогон мог воспроизводиться **без внешних систем**.

---

## 4. IDs, References, Correlation Binding

### 4.1 Artifact ID
Artifact ID = content-addressed hash от:
- canonical JSON контента,
- сериализованного `creation_context` (timestamp, node_ref, input artifact IDs).

Формат: `sha256:<multibase-base58btc>` (точный префикс/кодек фиксируется в реализации).

### 4.2 Node Reference (`node_ref`)
Формат: `execution_id ":" plan_id "/" phase "/" node_id ["[" instance_index "]"]`

### 4.3 Correlation IDs (обязательные поля событий)
- `trace_id`
- `intention_id`
- `node_execution_id`
- `critic_evaluation_id`
- `span_id` (для иерархии операций)

---

## 5. Action Graph (DAG) Binding

### 5.1 Формат сериализации графа
JSON-структура Action Graph соответствует структуре из спеки (metadata/nodes/edges/conditionalBranches).

### 5.2 Node types (v1 contract)
- `model`
- `mcp`
- `code`
- `decision`
- `aggregate`

### 5.3 Edge types
- `data`
- `control`
- `resource`
- `conditional`

### 5.4 Детерминированное выполнение
Executor:
- топологический порядок,
- параллелизм ограничивается ресурсами,
- коммит результатов в детерминированном порядке (tie-breaker: node id / priority правила из спеки).

---

## 6. Unified Node Contracts Binding

### 6.1 Input/Output contracts
- Input: `inputs{...}`, `parameters{...}`, `context{executionId, traceId, retryCount, randomSeed}`
- Output: `outputs{...}` (artifact ids), `qualitySignals{...}`, `status`

### 6.2 Статусы и жизненный цикл узла
Статусы и допустимые переходы соответствуют спеки (`PENDING/READY/RUNNING/COMPLETED/FAILED/TIMEOUT/RETRYING/CANCELLED`).

### 6.3 Retry/Timeout/Failure classes
Binding-политики:
- retryable errors по умолчанию: `timeout`, `transient_failure`, `rate_limited`
- failure classes: `TRANSIENT | PERSISTENT | SYSTEMIC | CRITICAL`
- при `CRITICAL` — немедленная эскалация/policy gate.

---

## 7. Code Node Sandbox Binding (Docker, Windows-friendly)

### 7.1 Требование Docker
- В v1 `CodeNode` запускается **только** в Docker (Docker Desktop на Windows поддерживается).

### 7.2 Политики безопасности
- сеть **off by default** (включается явной настройкой),
- файловая система: allowlist roots,
- лимиты: CPU/RAM/wall-time, лимит размера вывода,
- логирование: stdout/stderr в structured logs.

### 7.3 Reproducibility
- base image pinned,
- зависимости раннера фиксируются (lockfile hash),
- окружение инъектируется декларативно (список переменных, секреты не хардкодим).

---

## 8. Storage Binding (OSS default + Production adapters)

### 8.1 Термины
- **Blob store (content-addressed)**: хранилище неизменяемых blob’ов, где путь/ключ выводится из Artifact ID.
- **Index store**: индекс/метаданные/связи/поиск.

### 8.2 Default (OSS / локально, без внешних сервисов)
- Blob store: **filesystem CAS** (папка проекта).
- Index store: **SQLite** (метаданные, lineage edges, trace events, быстрый доступ).

### 8.3 Production (адаптеры)
- Index store: **Postgres** (обязательно для production-режима).
- Blob store: **filesystem или S3/MinIO** (выбор зависит от окружения; в v1 реализуем минимум FS, S3 — extension).

### 8.4 Embeddings / Semantic Search (важно: лёгкая установка)
- Базовая установка **не требует** embeddings backend.
- Семантический поиск включается опционально через один из backend’ов:
  - **Postgres + pgvector** (опционально; не обязателен),
  - **Local embeddings** (binding: рекомендованный OSS путь) через `sentence-transformers` + локальный индекс.

**Binding**: если semantic backend не настроен, `mode=semantic` деградирует в:
- keyword/FTS + Graph traversal (с явным warning в trace).

### 8.5 Почему НЕ Neo4j в v1
Neo4j — мощно, но добавляет тяжёлую зависимость и усложняет OSS adoption. В v1 используем:
- SQLite/PG для граф-структур (entities/relations/lineage),
- векторный поиск — через pgvector (опционально) или future local backend.

Neo4j остаётся **Future adapter**, если появится реальная необходимость.

---

## 9. Memory / GraphRAG Binding (контракт, не обязательно полный scale)

### 9.1 Unified Query Interface
Контракт `memoryQuery` фиксируется как в спеки (mode: semantic/structured/hybrid/iterative; constraints; iteration; synthesis).

### 9.2 Entity/Relation model (v1)
Сущности/отношения и provenance поля фиксируются по спеки (минимум: `Entity`, `Relation`, confidence, provenance).

### 9.3 Agentic retrieval loop
State machine: `PLAN → RETRIEVE → VALIDATE → SYNTHESIZE → DECIDE`.

---

## 10. Verification Layer Binding (Simulated Critics)

### 10.1 Critic I/O contracts
Input/Output schemas критика соответствуют спеки.

### 10.2 Verdict semantics
`PASS | CONDITIONAL_PASS | FAIL | UNCERTAIN` с указанными правилами обработки.

### 10.3 Evidence requirements
Недостаток evidence ⇒ `UNCERTAIN` + явные gaps; при high-stakes ⇒ escalation.

---

## 11. Control Protocol Binding

### 11.1 Команды пользователя (CLI и будущие UI)
- `continue`, `pause`, `revise`, `replan`, `stop` — семантика по спеки.

### 11.2 NL feedback → control signals
Компонент преобразования NL feedback должен:
- классифицировать intent,
- извлекать scope/constraints,
- при низкой уверенности формировать `ClarificationRequest`.

---

## 12. Queue / Async Execution Binding (Production quality)

### 12.1 Выбранный v1 вариант
- **Redis + RQ** — рекомендованный production runner для long-horizon задач.

### 12.2 OSS default
- Без очереди: in-process execution через CLI (batch/supervised).
- Очередь — опционально (extras), но API спроектировано так, чтобы web-приложение могло легко включить worker mode.

---

## 13. Packaging & Repository Binding (для генерации проекта)

### 13.1 Структура пакета (binding)
Репозиторий должен быть организован модульно (не перегружать один файл):
- `neuronium_agent/` (core, python package)
  - `core/` (state machine, orchestration)
  - `planning/` (HTN, DAG model)
  - `execution/` (executor, retries, persistence)
  - `nodes/` (model/mcp/code + contracts)
  - `verification/` (critics)
  - `memory/` (GraphRAG + retrieval loop)
  - `storage/` (blob + index backends)
  - `control/` (protocol)
  - `trace/` (schemas, exporters, replay)
  - `cli/` (entrypoints)
- `examples/` (docker-compose, интеграции)
- `tests/` (contract tests, determinism/replay tests)

### 13.2 Extras (binding)
- `neuronium-agent[postgres]`
- `neuronium-agent[redis]`
- `neuronium-agent[pgvector]` (может зависеть от `postgres`)
- `neuronium-agent[docker]` (если хотим отдельной группой)

---

## 14. Open Questions (не блокируют старт, но требуют фиксации перед production)
- Local embeddings index: выбираем ли “bruteforce” (v1) → HNSW (v2), и где храним embeddings (SQLite/PG).
- pgvector policy: “если extension недоступен” — fallback стратегия в production (degrade vs alternative vector store).
- S3/MinIO adapter: нужен ли в v1 или оставить extension.

