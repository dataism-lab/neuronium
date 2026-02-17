# NeuroniumAgent — технический обзор (как устроено сейчас)

Этот документ — “конспект” для быстрого погружения: что где лежит, какие инварианты соблюдаются, что реально реализовано.

## 1) Ключевая идея

NEURONIUM — это **контрольная архитектура** вокруг LLM, а не “один большой промпт”.

- **Plan**: строим Action Graph (DAG)
- **Execute**: исполняем узлы графа (частично параллельно)
- **Control**: критик (и/или quality gate) валидирует результат
- **Adapt**: revise/replan/escalate/stop/continue

## 2) Главные инварианты (что система защищает)

- **Детерминизм**: фиксированные порядки, канонический JSON, стабильные tie-break правила.
- **Immutability артефактов**: результаты узлов сохраняются как immutable blobs (CAS).
- **Replay**: недетерминированные ответы (LLM/tools) пишутся в trace как `replay_data` и могут быть строго воспроизведены офлайн.

## 3) Структура модулей (high level)

- `neuronium_agent/api.py`: публичный фасад `AgentRunner`
- `neuronium_agent/cli/main.py`: CLI команды `run/status/control/replay/schema/worker`
- `neuronium_agent/core/orchestrator.py`: Cognitive Core (цикл + runbook runner)
- `neuronium_agent/planning/*`:
  - `htn.py`: autofix demo planner (DAG-шаблоны iter1/iter2)
  - `runbooks.py`: docs_report_v1 runbook (DAG-шаблон)
  - `memory_runbook.py`: hybrid_memory_report_v1 (2 стадии)
  - `runbook_contract.py`: контракт стадий и quality gate
- `neuronium_agent/execution/executor.py`: DAG executor (топологический порядок + ThreadPool)
- `neuronium_agent/nodes/*`: узлы `ModelNode/CodeNode/McpToolNode/Aggregate/Decision`
- `neuronium_agent/tools/*`: local tools + memory tools + ToolRuntime DI
- `neuronium_agent/trace/*`: recorder/exporter/checkpoints/replay provider
- `neuronium_agent/storage/*`: FS CAS + SQLite/Postgres index stores + migrations
- `neuronium_agent/memory/*`: GraphRAG-lite (chunks, provenance DTO, SQLite/PG store)

## 4) Что такое “runbook” в этой версии

Runbook = детерминированное семейство планов, оформленное как **последовательность стадий**, где каждая стадия — DAG + quality gate.

Сейчас доступны:
- `autofix_demo`: фиксированный 2-итерационный цикл в оркестраторе
- `docs_report_v1`: 1 стадия (read docs → merge → draft → critic)
- `hybrid_memory_report_v1`: 2 стадии (ingest+retrieve → synthesise+verify)

## 5) Trace и replay (что важно для аудита)

### Типовые события
- `decision`: объяснимые решения (выбор runbook, создание плана, strict_fail)
- `node_start` / `node_end`: исполнение узлов
- `critic_verdict`: нормализованный вердикт критика
- `checkpoint`: phase-boundary снимок для resume
- `replay_data`: записанные ответы, которые использует strict replay
- `stage_start` / `stage_end`: границы стадий runbook’а

### Strict replay
- При replay оркестратор загружает события исходного `trace_id`, создаёт новый `trace_id`, и через `ReplayProvider` инжектит recorded responses в узлы.
- В строгом режиме при отсутствии `replay_data` для любого replay-capable узла — **ошибка** (никакого “тихого” перехода к live вызовам).

## 6) Память (Stage 5: GraphRAG-lite)

Реализовано:
- ingestion локальных файлов в `memory_chunks` (чанки фиксированного размера + overlap)
- retrieval через keyword scoring + детерминированный tie-break
- provenance/evidence: `EvidenceRef` с `quote_hash` (проверяемость цитаты)
- фильтрация по `source_kind` и `visibility`

Не реализовано (пока):
- embeddings-based semantic retrieval
- entity/relation graph + multi-hop traversal
- iterative retrieval loop как state machine (iterative сейчас ведёт себя как hybrid)

