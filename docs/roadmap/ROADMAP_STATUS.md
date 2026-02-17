# ROADMAP Status — NEURONIUM (оценка текущей реализации)
Дата: 2026-02-16 (обновлено: runbooks + mcp(local) + SQLite thread-safety)

Это **не план**, а "сверка" `ROADMAP.md` с тем, что реально есть в репозитории на текущий момент.

Легенда:
- `[x]` — реализовано и используется
- `[~]` — частично (есть базовая реализация, но не дотягивает до acceptance или не интегрировано end-to-end)
- `[ ]` — нет / не подключено / только заглушка

---

## Stage 1 — Доменные модели и контракты

**Deliverables**
- [x] Pydantic-модели AgentState/Intention lifecycle — `neuronium_agent/core/state.py`
- [x] Pydantic-модели ActionGraph (DAG) — `neuronium_agent/planning/dag.py`
- [x] Typed Node I/O контракт — `neuronium_agent/nodes/base.py`
- [x] Critic I/O модели — `neuronium_agent/verification/critic.py` + `verification/demo_critic.py` (DemoCriticVerdict с hard rule: PASS requires evidence)
- [x] Trace records/event model — `neuronium_agent/trace/recorder.py` + `storage/*_store.py`
- [x] Генерация JSON Schema как отдельный deliverable/команда — `neuronium_agent/schemas/` (registry + export) + CLI `neuronium-agent schema export --out <dir>` (Phase A)

**Acceptance**
- [x] Канонический JSON зафиксирован — `neuronium_agent/_canonical.py` + `tests/test_determinism.py`
- [x] "Reference payloads" для всех схем (как набор эталонов) — `tests/reference_payloads/*.json` (21 модель), валидация: `tests/test_schema_reference_payloads.py` (16 тестов ✓) (Phase A)

**Вывод: 100% выполнено.**

---

## Stage 2 — State machines и lifecycle management

**Deliverables**
- [x] Commit/Execute/Control/Adapt фазы — полный цикл реализован в `orchestrator.py`: `COMMIT → EXECUTE → CONTROL → ADAPT → DONE` для каждой итерации
- [x] Meta-control actions: continue/revise/replan/escalate — **декларативный** `apply_control()` в orchestrator: команда → state transition → checkpoint → trace decision. Поддержка: pause, continue, stop, revise (constraints_add), replan, escalate. CLI: `neuronium-agent control --command <cmd>` (Phase B)
- [x] Checkpoint/лог снапшотов — phase-boundary checkpoints (`neuronium_agent/trace/checkpoints.py`): `build_checkpoint_payload` / `load_state_from_checkpoint` + `get_latest_phase_boundary_checkpoint`. Resume: `orchestrator.resume_run()` + CLI `neuronium-agent run --trace-id <id>` (Phase B)

**Acceptance**
- [x] Deterministic transition coverage — `tests/test_checkpoints_and_control.py::TestStateTransitions` — полное покрытие таблицы переходов RunState + проверка терминальных состояний (Phase B)
- [x] Deterministic replay на синтетических сценариях — реализовано через pre-recorded responses, тесты: `test_determinism.py`, `test_replay.py`, `test_autofix_demo.py`
- [x] Resume invariant — восстановление только из phase-boundary checkpoint (10 меток в `PHASE_BOUNDARIES`), mid-node не допускается; тесты: `TestResumeInvariant` (Phase B)

**Вывод: ~95% выполнено.** Осталось: интеграционный тест полного цикла pause → continue → resume → completion.

---

## Stage 3 — Node execution infrastructure

**Deliverables**
- [x] Unified node интерфейс + статусы — `neuronium_agent/nodes/base.py` (BaseNode, NodeInput, NodeOutput, QualitySignals, NodeContext)
- [x] ModelNode (OpenAI provider, structured output, запись для replay) — `neuronium_agent/nodes/model_node.py`
- [~] McpToolNode — **частично**: реального MCP протокола нет, но `node_type="mcp"` теперь **исполняется** через локальный transport (in-process tools), с policy allowlist и записью replay_data — `neuronium_agent/nodes/mcp_node.py`, `neuronium_agent/tools/local_tools.py`
- [x] CodeNode (Python-only) с Docker sandbox — `neuronium_agent/nodes/code_node.py`
- [x] DecisionNode — условное ветвление в DAG — `neuronium_agent/nodes/decision_node.py`
- [x] AggregateNode — слияние upstream выходов — `neuronium_agent/nodes/aggregate_node.py`

**Acceptance**
- [~] Contract tests для каждого узла — есть e2e и determinism тесты; нет отдельных contract suites на каждый тип узла
- [x] Запись и strict replay внешних ответов — **реализовано**: recording через `enable_recording()`, strict replay через `ReplayProvider` (не использует node_end fallback в strict mode); тесты: `test_replay.py`, `test_autofix_demo.py`
- [x] Executor поддерживает `initial_inputs` — реализовано, тест: `test_executor_initial_inputs.py`
- [x] Потокобезопасная запись trace events при параллельном выполнении DAG (SQLite) — `neuronium_agent/storage/sqlite_store.py` (lock + `check_same_thread=False`) + `neuronium_agent/trace/recorder.py` (lock), тест: `tests/test_sqlite_thread_safety.py`

**Вывод: ~85% выполнено.** MCP-протокол всё ещё заглушка, но tool layer стал “живым” через mcp(local). Contract test suites не выделены как отдельный набор.

---

## Stage 4 — Planning + Verification (end-to-end)

**Deliverables**
- [~] HTN decomposition engine → Action Graph — **шаблонный** (2 фиксированных DAG-шаблона: `plan_iter1` и `plan_iter2_fix`, не обобщённая декомпозиция), но покрывает демо-сценарий — `neuronium_agent/planning/htn.py`
- [x] DAG executor (детерминизм, параллельность) — `neuronium_agent/execution/executor.py` (topological order, ThreadPoolExecutor, deterministic sort)
- [x] Simulated critics + verdict pipeline — **реализовано**: `DemoCriticVerdict` с hard rule (PASS requires evidence), `CRITIC_SYSTEM_PROMPT`, `parse_critic_verdict`, LLM-critic интегрирован в DAG как model-нода — `verification/demo_critic.py`
- [~] Доп. детерминированный бизнес-runbook как ActionGraph-шаблон (docs→draft→critic) — `neuronium_agent/planning/runbooks.py`, критик: `neuronium_agent/verification/business_critic.py`

**Acceptance**
- [x] End-to-end демонстрация "1 инструкция → DAG → выполнение → trace" — **полный вертикальный срез**: objective → plan_iter1 (3 ноды) → execute → critic → verdict → (replan?) → plan_iter2_fix → execute → critic → COMPLETED/FAILED; тесты: `test_api.py`, `test_autofix_demo.py`
- [x] Replan/backtracking на ошибках инструментов — **реализовано** в orchestrator: iter1 FAIL → build_fix_context → plan_iter2_fix → execute → critic; тесты: `test_autofix_demo.py` (NameError → fix → PASS)

**Вывод: ~85% выполнено.** Работает end-to-end с replan. HTN — шаблонный, не обобщённый.

---

## Stage 5 — Memory: GraphRAG + agentic retrieval

**Deliverables**
- [~] Chunk-based GraphRAG-lite storage (SQLite/PG) — **есть**: `neuronium_agent/memory/sqlite_memory_store.py`, `memory/postgres_memory_store.py` (таблица `memory_chunks`)
- [~] Unified Query Interface — **частично**: pydantic-контракты `MemoryQuery/MemoryResult/EvidenceRef` в `neuronium_agent/memory/models.py` + tool `memory.query` (`neuronium_agent/tools/memory_tools.py`)
- [ ] Entity/relation storage + graph traversal (multi-hop) — нет (пока только чанки, без сущностей/отношений)
- [ ] Iterative retrieval loop state machine — нет (mode=`iterative` пока деградирует в тот же retrieval, что и hybrid)
- [x] Provenance/evidence linking — **есть**: `EvidenceRef` + `quote_hash` + `locator` (source_uri + span) в `memory.models` и выдаче `memory.query`

**Acceptance**
- [x] Retrieval structured/hybrid — **есть** (keyword scoring, deterministic tie-break) — `SqliteMemoryStore.search_keyword_topk` + `invoke_memory_query`
- [~] Semantic mode с pgvector или деградацией + trace warning — **частично**: fallback/ошибка при `require_exact_mode=True` реализованы и протестированы, но embeddings-based retrieval пока не реализован (`tests/test_memory_stage5.py`)

**Вывод: ~35–45% выполнено.** Есть GraphRAG-lite на чанках + provenance + runbook, но нет entity/relation графа, embeddings и iterative retrieval loop.

---

## Stage 6 — CLI runtime + тестирование + интеграции

**Deliverables**
- [x] CLI: run/status/control/replay/worker/schema — `neuronium_agent/cli/main.py`
- [~] CLI: выбор runbook для новых запусков — добавлен `--runbook` (по умолчанию `autofix_demo`), runbook_id передаётся через `RunRequest.metadata`
- [x] CLI resume from checkpoints — `neuronium-agent run --trace-id <id>` (Phase B); supervised mode — не реализован (runtime)
- [~] Примеры production зависимостей через extras — extras описаны, docker-compose примеров нет
- [x] Тесты determinism/replay — `test_determinism.py`, `test_replay.py`, `test_autofix_demo.py`, `test_api.py`
- [~] Failure simulation — ограниченно (strict_fail без LLM ключа, missing replay_data)

**Acceptance**
- [x] "Пользователь может: запустить, поставить на паузу, продолжить, пересобрать план, экспортировать trace" — запустить ✓, пауза/стоп ✓, continue ✓, replan ✓, revise ✓, escalate ✓, resume ✓, trace export ✓ (Phase B)
- [x] `replay --trace-id <id>` работает (strict offline) и создаёт новый `trace_id` — `cli/main.py`, `api.py`, тесты: `test_replay.py`

**Вывод: ~75% выполнено.** CLI работает для всех основных сценариев включая resume. Supervised mode, docker-compose примеры отсутствуют.

---

## Production add-ons

- [x] Postgres index backend — `neuronium_agent/storage/postgres_store.py` + `storage/migrations/postgres/*`
- [ ] pgvector backend — нет (есть зависимость/конфиг, реализации нет)
- [x] Redis+RQ runner (async execution) — `neuronium_agent/queue/rq_runner.py`
- [ ] S3/MinIO blob store adapter — нет (в конфиге `blob_backend="s3"` есть, реализации нет)
- [ ] Local embeddings backend — нет (есть зависимости/конфиг, реализации нет)
- [ ] SQL runtime для CodeNode (DuckDB/PG) — нет

---

## Сводка по стадиям

| Stage | Прогресс | Основные блокеры |
|-------|----------|-----------------|
| 1 — Доменные модели | **100%** | — |
| 2 — State machines | **~95%** | Интеграционный тест полного цикла pause→resume→completion |
| 3 — Node execution | ~85% | MCP реальный протокол, contract test suites |
| 4 — Planning + Verification | ~85% | Обобщённый HTN (не шаблонный) |
| 5 — Memory: GraphRAG | **~40%** | embeddings/semantic retrieval, entity+relation graph, iterative retrieval loop |
| 6 — CLI + тестирование | **~78%** | Supervised mode, docker-compose |
| Production add-ons | ~30% | pgvector, S3, embeddings |

