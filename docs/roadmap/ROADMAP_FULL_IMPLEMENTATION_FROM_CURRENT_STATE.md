# ROADMAP: Полная реализация IBS с текущего состояния

Версия: v1.0  
Дата: 2026-02-25  
Источник требований: `docs/architecture/AI_Super_Agent_Architecture_Implementation_Specification.md` (IBS)

---

## 1) Стартовая точка (as-is)

На текущий момент закрыт migration-трек tool-agnostic (phases 1-6) и ключевые дефекты clarification/control (включая BUG-4 hardening).  
Это означает, что базовый контур clarification/revise уже patch-first, schema-driven и покрыт регрессиями.

Остающийся разрыв до **полной реализации IBS** лежит в доведении системы до полноценных инвариантов и подсистем IBS (не только clarification-потока):

- строгий end-to-end детерминизм и replay как источник истины для всех недетерминированных входов;
- завершение формальных контрактов доменной модели и trace/artifact lineage;
- расширение verification/critic до контрактов IBS;
- доведение memory слоя до полного GraphRAG + agentic retrieval state machine;
- полноценный CLI runtime режимов batch/supervised/interactive с воспроизводимым restore/export;
- операционная готовность (observability, reliability gates, release criteria).

---

## 2) Цель (to-be)

Прийти к состоянию, где система соответствует IBS по всем ключевым разделам:

- `§1` инварианты (determinism, immutability, replay reproducibility);
- `§2-3` формальная доменная модель и lifecycle/decision audit;
- `§4-6` HTN+DAG исполнение с доказуемой корректностью и восстановлением;
- `§7` verification layer с формальными critic-контрактами;
- `§8` hybrid memory (GraphRAG + agentic loop + provenance);
- `§9-10` полный control protocol и формальный audit artifact schema.

---

## 3) План реализации (post-phase6 -> full IBS)

## Phase A — Invariants Hardening (Determinism/Replay/Immutability)

**Цель:** закрыть фундаментальные требования IBS `§1.2` на системном уровне.

**Scope**
- Зафиксировать canonical JSON serialization policy (sorted keys, numeric normalization).
- Провести аудит всех внешних/недетерминированных источников и гарантировать запись в replay trace.
- Закрыть “дыры replay”: запрет скрытых fallback-вызовов во время strict replay.
- Формализовать policy неизменяемости artifacts (append-only lineage + запрет in-place mutation).

**Acceptance**
- Набор deterministic/replay тестов зеленый в strict режиме.
- Для контрольных сценариев: одинаковые входы дают идентичные trace-структуры и outputs.
- Любая попытка replay без полного недетерминированного контекста приводит к явной ошибке полноты trace.

---

## Phase B — Domain & Contract Completion

**Цель:** довести типизацию и контракты до полного покрытия IBS `§2`, `§5`, `§10`.

**Scope**
- Нормализовать/расширить доменные типы AgentState, Intention, Artifact, DecisionRecord, Critic I/O.
- Финализировать JSON Schema для node input/output, critic input/output, trace artifacts.
- Ввести versioned schema policy (compatibility + migration notes).
- Закрыть contract tests на границах всех node типов и control commands.

**Acceptance**
- Все reference payloads проходят schema validation.
- Добавлен regression suite на backward compatibility ключевых артефактов.
- Контракты используются как единственный source of truth на I/O boundaries.

---

## Phase C — Planning & Execution Semantics Maturity

**Цель:** довести HTN/DAG контур до зрелости IBS `§4` и `§6`.

**Scope**
- Hardening HTN decomposition: method ranking, failure classification, partial invalidation.
- Детальный scheduling policy: critical path + deterministic tie-breaking under parallelism.
- Формализованный rollback scope (node/subgraph/intention) с сохранением valid branches.
- Проверка/поддержка conditional branching semantics без нарушения DAG-инварианта.

**Acceptance**
- End-to-end сценарии с backtracking/replan проходят без nondeterministic drift.
- Тесты подтверждают корректную partial invalidation и сохранение независимых ветвей.
- Trace отражает все scheduling/rollback решения с correlation IDs.

---

## Phase D — Verification Layer 2.0 (Critics)

**Цель:** реализовать контрактный verification pipeline по IBS `§7`.

**Scope**
- Ввести formal critic evaluation contracts (input schema, verdict schema, evidence requirements).
- Поддержать verdict modes: PASS / CONDITIONAL_PASS / FAIL / UNCERTAIN.
- Добавить deficiency severity (MINOR/MAJOR/CRITICAL) и policy mapping в adapt/escalate.
- Расширить uncertainty handling: threshold policy, ambiguity detection, disagreement handling.

**Acceptance**
- Critic pipeline покрыт unit/integration тестами на positive/negative/uncertain сценарии.
- Rejection triggers корректно маршрутизируются в revise/replan/escalate.
- В trace есть полная связка evaluation input -> verdict -> downstream decision.

---

## Phase E — Memory Full Stack (GraphRAG + Agentic Retrieval)

**Цель:** перейти от текущего GraphRAG-lite к полной IBS-модели `§8`.

**Scope**
- Реализовать entity/relation graph model с provenance и temporal validity.
- Ввести unified query interface (semantic/structured/hybrid/iterative).
- Реализовать retrieval loop state machine: PLAN -> RETRIEVE -> VALIDATE -> SYNTHESIZE -> DECIDE.
- Добавить contradiction/gap detection и stopping criteria policy.

**Acceptance**
- Multi-hop и hybrid retrieval сценарии воспроизводимы и покрыты тестами.
- Synthesis выводит результат с доказуемыми ссылками на provenance/evidence.
- Failure/uncertainty в retrieval корректно ведут к следующей итерации или escalation.

---

## Phase F — CLI Runtime Completion & Operational Readiness

**Цель:** довести runtime до полной эксплуатационной готовности по IBS `§1.1.2`, `§9`, `§10`.

**Scope**
- Закрыть CLI режимы batch/supervised/interactive как равноправные execution modes.
- Финализировать pause/continue/revise/replan/stop протокол с консистентным UX и state restore.
- Полный trace export/import/replay workflow (включая валидацию полноты перед replay).
- Операционная готовность: метрики, structured logs, failure drills, release gates.

**Acceptance**
- Пользователь может пройти полный цикл run -> pause/revise -> resume -> export -> replay offline.
- Проверены сценарии unexpected termination и deterministic recovery from checkpoint.
- Есть runbook готовности релиза с критериями pass/fail.

---

## 4) Сквозные треки (идут через все фазы)

- **Тестовый трек:** сначала red-tests на новые инварианты, затем реализация и regression.
- **Совместимость:** постепенная деактивация legacy-bridge путей только после подтвержденного adoption.
- **Наблюдаемость:** расширение decision/evidence telemetry в каждом фазовом PR.
- **Безопасность изменений:** мелкие обратимые шаги, без broad rewrite.

---

## 5) Порядок выполнения и зависимости

Рекомендуемый порядок:

1. `Phase A` (фундамент инвариантов)  
2. `Phase B` (контракты и схемы)  
3. `Phase C` + `Phase D` (планирование/исполнение и verification)  
4. `Phase E` (memory full stack)  
5. `Phase F` (runtime + production readiness)

Критические зависимости:

- Без завершения `Phase A` нельзя объективно подтверждать соответствие replay/trace требованиям IBS.
- `Phase B` должен завершиться до масштабного расширения critic/memory контрактов (`Phase D/E`).
- `Phase F` закрывается последним, когда стабилизированы все нижележащие подсистемы.

---

## 6) Definition of Done для “полной реализации IBS”

Считаем цель достигнутой, когда одновременно выполнено:

- все фазовые acceptance criteria (`A-F`) подтверждены тестами/сценариями;
- strict replay стабилен на контрольных e2e сценариях без внешних fallback;
- artifact lineage неизменяем и проверяем через integrity queries;
- verification слой принимает/отклоняет результаты по формальным контрактам и uncertainty policy;
- memory слой работает в structured/hybrid/iterative режимах с provenance и contradiction handling;
- CLI обеспечивает полный управляемый цикл исполнения и воспроизведения trace.

---

## 7) Ближайший следующий шаг (рекомендуемый)

Начать с **Phase A / Iteration 1**:

- зафиксировать deterministic serialization и replay completeness checklist;
- добавить red-tests на неполноту replay-trace и на скрытые недетерминированные вызовы;
- закрыть их минимальными правками без изменения внешнего API.
