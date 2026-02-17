# Статус проекта NEURONIUM Agent (NeuroniumAgent) — для руководства

Дата: 2026-02-16  
Источник фактов: код репозитория + тестовый прогон `pytest` (**142 passed**).

## 1) Коротко (executive summary)

Проект находится в состоянии **рабочего “вертикального среза” (vertical slice)**: есть CLI и библиотечный API, детерминированное выполнение **Action Graph (DAG)**, запись **audit trace**, строгий **offline replay**, контрольные команды (pause/continue/replan/revise/stop/escalate), базовые storage-адаптеры (SQLite/FS CAS, Postgres опционально) и 2 прикладных runbook’а для “бизнес-отчётов”.

Важно: часть заявленных в презентации/архит-спеке направлений реализована как **v0.2 каркас** (шаблонный planner, MCP только “local transport”, память — GraphRAG-lite на чанках без семантических эмбеддингов и без graph traversal).

## 2) Что уже можно делать (реально работает)

- **Запуск через CLI**: `neuronium-agent run ...` (по умолчанию runbook `autofix_demo`).
- **API-фасад**: `neuronium_agent.api:create_runner()` → `AgentRunner.start/get_status/control/export_trace/replay/resume_run`.
- **DAG выполнение**: узлы `model`, `code`, `mcp(local)`, `aggregate`, `decision`.
- **Trace / аудит**:
  - события `decision`, `node_start`, `node_end`, `critic_verdict`, `checkpoint`,
  - экспорт трейса в `jsonl/json/zip`.
- **Strict replay (offline)**:
  - `neuronium-agent replay --trace-id ...` воспроизводит run **без внешних вызовов**,
  - в строгом режиме требует `replay_data` для replay-способных узлов (тестируется).
- **Resume из checkpoint**:
  - `neuronium-agent run --trace-id <id>` продолжает выполнение с phase-boundary checkpoint,
  - для runbook’ов умеет best-effort продолжать **без повторного выполнения** (по gate snapshot).
- **Runbooks (прикладные “семейства планов”)**:
  - `autofix_demo`: generate → execute → critic → (если FAIL) fix → execute_fix → critic_fix.
  - `docs_report_v1`: чтение локальных документов → агрегация → черновик отчёта → критик.
  - `hybrid_memory_report_v1`: ingest internal/user docs в память → retrieval → draft → critic (демо “memory as component”).
- **Память (Stage 5, GraphRAG-lite)**:
  - ingestion файлов в таблицу `memory_chunks`,
  - retrieval (сейчас **keyword/hybrid**) + `EvidenceRef` (quote + quote_hash + locator),
  - фильтры `source_kind`/`visibility` (user/internal/audit_only) поддержаны.
- **Production add-ons (частично)**:
  - `IndexStore` на Postgres реализован,
  - Redis+RQ runner/worker реализован (опционально через extras).

## 3) Что реализовано частично / не соответствует “обещаниям” из презентации

- **HTN planner**: в `autofix_demo` планирование — это **детерминированные DAG-шаблоны**, не универсальная декомпозиция.
- **MCP**: протокол MCP как внешний транспорт **не реализован**; вместо этого есть “local transport” (`invoke_local_tool`) и policy-gates для FS.
- **Memory/GraphRAG**:
  - есть “GraphRAG-lite” на чанках и provenance,
  - **нет** entity/relation графа и multi-hop traversal,
  - режим `semantic` по флагу существует, но фактический retrieval пока **не embeddings-based** (используется тот же keyword путь).
- **Supervised mode**: параметр/флаг присутствует в конфиге/CLI, но интерактивного пошагового подтверждения как продукта пока нет.
- **S3/MinIO blob store, pgvector, local embeddings**: как направления/конфиг — есть, как реализация end-to-end — нет.

## 4) Риски (для демо и для продакшена)

- **Зависимости для live-демо**:
  - `autofix_demo` в live режиме требует **LLM ключ** и для `CodeNode` — **Docker**.
  - Надёжный план B: показывать **offline replay** (без ключей/сети).
- **Ограничения безопасности**:
  - FS-инструменты ограничены allowlist корня (по умолчанию CWD) — это хорошо, но важно запускать CLI из корня проекта.
- **Функциональные пробелы**:
  - MCP-интеграции с enterprise системами пока только “заглушка с интерфейсом”.
  - Memory semantic/iterative retrieval loop отсутствует как полноценная state machine.

## 5) Рекомендованные следующие шаги (1–3 недели)

- Зафиксировать “v0.2” позиционирование: **что демо показывает** (control loop + DAG + trace/replay + critics + policy-gated tools).
- Memory Stage 5:
  - реализовать реальный `semantic` (embeddings + top-k) или жёстко деградировать с warning,
  - добавить хотя бы минимальный “iterative retrieval loop” (PLAN→RETRIEVE→VALIDATE→SYNTHESIZE→DECIDE).
- MCP:
  - выделить интерфейс транспортного слоя и добавить хотя бы один реальный MCP-адаптер (или чётко обозначить roadmap).
- Supervised mode:
  - сделать реально интерактивным (подтверждение стадий/узлов, UI/CLI prompts).

