# ROADMAP — NEURONIUM (AI Super Agent)

Версия: v0.1  
Дата: 2026-02-10  
Цель: зафиксировать **последовательность реализации** (stage gates) и не забыть про GraphRAG/очереди/production-адаптеры.

Основание: staged подход из архитектурной спеки (6 стадий).

---

## Принципы поставки
- Каждый этап заканчивается **артефактами + тестами**, подтверждающими инварианты (детерминизм, неизменяемость, replay).
- По умолчанию — **лёгкая локальная установка** (FS+SQLite), “тяжёлое” — через extras и примеры.

---

## Stage 1 — Доменные модели и контракты
**Deliverables**
- Pydantic-модели для: AgentState, Intentions, Artifacts, ActionGraph, Node I/O, Critic I/O, Trace records.
- Генерация JSON Schema.

**Acceptance**
- Схемы валидируют reference payloads.
- Канонический JSON зафиксирован и протестирован.

---

## Stage 2 — State machines и lifecycle management
**Deliverables**
- Intention lifecycle: Commit/Execute/Control/Adapt.
- Meta-control actions: continue/revise/replan/escalate.
- Checkpoint model (лог + снапшоты).

**Acceptance**
- Exhaustive coverage переходов.
- Deterministic replay на синтетических сценариях.

---

## Stage 3 — Node execution infrastructure
**Deliverables**
- Unified node интерфейс + статусы.
- ModelNode (LLM provider abstraction + structured outputs + запись для replay).
- McpToolNode (capability discovery + policy gates + audit log).
- CodeNode (Python-only) с Docker sandbox.

**Acceptance**
- Contract tests для каждого узла.
- Запись и **strict replay** внешних ответов (replay использует только `replay_data`, без fallback на `node_end`, без внешних вызовов).
- Executor поддерживает передачу `initial_inputs` (например `objective/constraints`) в root-ноды.

---

## Stage 4 — Planning + Verification (end-to-end)
**Deliverables**
- HTN decomposition engine → Action Graph (DAG).
- DAG executor (детерминизм, параллельность).
- Simulated critics (минимум) + verdict pipeline.

**Acceptance**
- End-to-end демонстрация: “1 инструкция → DAG → выполнение → trace”.
- Replan/backtracking на ошибках инструментов.

---

## Stage 5 — Memory: GraphRAG + agentic retrieval (обязательный этап)
**Deliverables**
- GraphRAG entity/relation storage (локально: SQLite). *В v0.1 реализован вариант GraphRAG-lite: chunks + provenance + retrieval (keyword/hybrid); полный entity/relation граф — цель следующих итераций.*
- Unified Query Interface.
- Iterative retrieval loop state machine.
- Provenance/evidence linking.

**Acceptance**
- Retrieval работает в structured/hybrid режимах.
- Semantic mode: либо pgvector (если включено), либо явная деградация с trace warning.

---

## Stage 6 — CLI runtime + тестирование + интеграции
**Deliverables**
- CLI: batch + supervised, trace export, resume from checkpoints.
- Примеры (по мере появления): docker-compose для dev (Postgres/Redis), интеграционный пример “web app → worker”.
- Тесты: determinism/replay, failure simulation.

**Acceptance**
- Пользователь может: запустить, поставить на паузу, продолжить, пересобрать план, экспортировать trace.
- Пользователь может выполнить `replay` по `trace_id` в offline-режиме (strict), получив новый `trace_id`.

---

## Production add-ons (после v1 или параллельно, если нужно)
- Postgres index backend.
- pgvector backend (опционально).
- Redis+RQ runner (async execution).
- S3/MinIO blob store adapter.
- Local embeddings backend (sentence-transformers + local index).
- SQL runtime для CodeNode (DuckDB или PG) — только если появятся кейсы.


