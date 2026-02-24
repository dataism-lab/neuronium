# Task Memory

## 2026-02-22 — Serena usage rule

### Plan
- Add a persistent Cursor rule that enforces Serena-first workflow.
- Keep the rule concise and always active.

### Decisions
- Scope: always apply (`alwaysApply: true`).
- Rule file: `.cursor/rules/serena-first-workflow.mdc`.
- Focus: command/test execution via Serena, schema-first MCP usage, Serena-first navigation/editing, fallback policy.

### Verification
- Created `.cursor/rules/serena-first-workflow.mdc`.
- Confirmed file write success via Serena tool response.

### Outcome
- Persistent project rule added to bias future work toward Serena tools.

### Handoff note
- In the next session, keep using Serena MCP tools by default.
- If a task cannot be completed with Serena after reasonable attempts, use standard tools and document why.

## 2026-02-24 — Tool-agnostic migration phase 1 prep

### Plan
- Create a dedicated branch for phase 1 implementation.
- Keep `tasks/todo.md` tracked in git as task-level memory and handoff source.
- Start phase 1 with small reversible increments and evidence-first checks.

### Decisions
- Branch strategy: one branch per migration phase (`feat/tool-agnostic-phase1`).
- Git strategy for `tasks/`: do not ignore the directory globally; keep `tasks/todo.md` versioned.
- Commit strategy: checkpoint commit before coding, then incremental commits per completed sub-step.

### Verification
- Created branch `feat/tool-agnostic-phase1` from current working state.
- Confirmed `tasks/todo.md` exists and is currently the only file in `tasks/`.
- Confirmed `.gitignore` does not exclude `tasks/`.

### Outcome
- Repository is prepared to start phase 1 in an isolated branch with tracked task memory.

### Handoff note
- Begin implementation from the approved phase 1 plan and keep each step small/reversible.
- After each sub-step, attach evidence (tests/logs/scenario checks) before marking done.

## 2026-02-24 — ToolSchemaRegistry placement decision

### Plan
- Make ToolSchemaRegistry placement explicit in migration docs to avoid drift.
- Keep phase 1 implementation small and reversible.

### Decisions
- Phase 1 default: place `ToolSchemaRegistry` near planning implementation.
- Promotion rule: move to shared layer once there is a second non-planning consumer (e.g., orchestrator revise path).
- Refactor triggers: 2+ consumers or rising cross-module churn/import coupling.

### Verification
- Updated `docs/internal/TOOL_AGNOSTIC_MIGRATION_PLAN.md` with explicit placement and trigger bullets.

### Outcome
- Placement strategy is now explicit in the phased migration plan and can be tracked in later phases.

### Handoff note
- Re-check placement decision at start of phase 3 before revise integration work.

## 2026-02-24 — Tool-agnostic phase 1 implementation

### Plan
- Implement phase-1 generic primitives without changing active runtime behavior.
- Add evidence via unit and smoke tests, including legacy clarification flow.

### Decisions
- Added `ToolSchemaRegistry` in planning scope for phase 1 with API:
  - schema lookup by `tool_name` or `operator_id`,
  - required JSON Pointer extraction,
  - schema merge for candidate tools/operators.
- Added `MissingSlot` and `compute_missing_slots(state, input_schema)` as schema-driven building blocks with dedupe and nullable handling.
- Added `StatePatch` RFC6902 subset (`add`, `replace`, `remove`) with deterministic validation errors.
- Kept legacy contracts/flows intact; phase-1 smoke test asserts no breakage in legacy `missing_fields` schema shape.

### Verification
- Added tests:
  - `tests/test_tool_schema_registry.py`
  - `tests/test_missing_slots.py`
  - `tests/test_state_patch.py`
  - `tests/test_phase1_legacy_compat_smoke.py`
- Ran:
  - `uv run pytest tests/test_tool_schema_registry.py tests/test_missing_slots.py tests/test_state_patch.py tests/test_phase1_legacy_compat_smoke.py tests/test_supervised_clarification_flow.py -q`
- Result: `12 passed`.

### Outcome
- Phase 1 baseline primitives are implemented and covered by tests.
- Current clarification/revise runtime behavior remains backward-compatible.

### Handoff note
- Next step for phase 2: connect dynamic extraction schema generation to `ToolSchemaRegistry` behind a feature flag.
- Before phase 3 revise integration, re-check whether `ToolSchemaRegistry` should be promoted to a shared module.

## 2026-02-24 — Migration plan bug-track consolidation

### Plan
- Make the phase migration document self-contained for architecture + bug context.
- Add explicit mapping from known clarification bugs to migration phases and separate fixes.

### Decisions
- Added a dedicated `Bug Stabilization Track` section in `docs/internal/TOOL_AGNOSTIC_MIGRATION_PLAN.md`.
- Kept bugs as a parallel track (not additional migration phases) to preserve P0/P1 prioritization.
- Added explicit rollout gates (P0 closure, revise integrity, migration-ready criterion).

### Verification
- Verified the migration plan now includes:
  - bug-to-phase coverage matrix,
  - recommended execution order,
  - rationale for parallel tracks,
  - readiness gates.

### Outcome
- The migration plan can now be read as a single source of truth without losing critical bug context.

### Handoff note
- Before phase 2 implementation starts, confirm BUG-5 scheduling as a P0 hotfix.

## 2026-02-24 — BUG-5 hotfix + phase 2 start

### Plan
- Close P0 BUG-5 in interactive CLI pause/continue flow.
- Start phase 2 with schema-driven extraction under feature flag and legacy-safe fallback.
- Prove changes with targeted tests before completion.

### Decisions
- BUG-5 fix in `neuronium_agent/cli/main.py`:
  - `_interactive_supervised_loop` now checks pause type before sending `continue`;
  - `continue` is sent only for clarification pauses with `clarification_request_artifact_id`;
  - `_interactive_run_loop` now refreshes status from orchestrator after supervised loop.
- Phase-2 implementation (incremental):
  - `extraction_envelope_json_schema(...)` now supports dynamic `input_schema` injection while keeping legacy default behavior unchanged when not provided;
  - `HtnRecursivePlannerBackend` now resolves dynamic extraction schema from `ToolSchemaRegistry` using stage `allowed_tool_names`;
  - feature flag controls rollout:
    - metadata: `dynamic_extraction_schema` (+ optional runbook/stage allowlists),
    - env: `NEURONIUM_DYNAMIC_EXTRACTION_SCHEMA` (+ optional runbook/stage allowlists).
- Backward compatibility:
  - default path remains legacy schema (feature flag off),
  - no forced migration in existing runbooks.

### Verification
- Added tests:
  - `tests/test_cli_bug5_pause_flow.py`
  - `tests/test_phase2_dynamic_extraction_schema.py`
- Ran:
  - `uv run pytest tests/test_cli_bug5_pause_flow.py tests/test_phase2_dynamic_extraction_schema.py tests/test_phase1_legacy_compat_smoke.py tests/test_supervised_clarification_flow.py -q`
- Result:
  - `9 passed`.
- Confirmed lints for edited files:
  - no linter errors reported.

### Outcome
- P0 BUG-5 behavior corrected: no blind `continue` for non-clarification pauses; stale status is refreshed after supervised loop.
- Phase 2 baseline is active behind feature flag: extraction schema can be generated from tool contracts for selected runbook/stage.
- Legacy clarification contract shape remains covered by smoke test.

### Handoff note
- Next step: expand dynamic schema integration from extraction prompt/schema into downstream validation path (phase 2 acceptance hardening).
- Before phase 3, re-check `ToolSchemaRegistry` placement trigger if a second non-planning consumer appears in runtime paths.

## 2026-02-24 — Phase 2 hardening completion (tool-agnostic)

### Plan
- Complete phase-2 hardening only: keep patch/revise (phase 3) untouched.
- Extend dynamic extraction schema usage from extraction to downstream missing-field validation.
- Preserve legacy behavior when dynamic flag is disabled.

### Decisions
- `HtnRecursivePlannerBackend` now carries `extraction_input_schema` from extraction stage artifacts into validation path.
- `_compute_missing_fields(...)` now supports dual-path behavior:
  - dynamic path (only when dynamic input schema exists): compute missing via `compute_missing_slots(...)` and map to legacy `MissingField.field` using `slot_path_to_legacy_field(...)`;
  - legacy path (flag off / no dynamic schema): keep previous task-type-driven logic unchanged.
- Dynamic validation keeps backward compatibility for clarification contract shape (`missing_fields`), while avoiding new per-tool `if/elif`.
- Added tests for:
  - dynamic validation activation and required-field propagation to missing list,
  - legacy fallback behavior unchanged,
  - new catalog tool reflection in extraction schema and missing fields without backend branching.

### Verification
- Ran targeted unit+integration suite:
  - `uv run pytest tests/test_phase2_dynamic_extraction_schema.py tests/test_planner_backend_contract.py tests/test_htn_recursive_backend_integration.py tests/test_phase1_legacy_compat_smoke.py tests/test_supervised_clarification_flow.py -q`
- Result:
  - `19 passed in 0.53s`
- Lint check on edited files:
  - `neuronium_agent/planning/htn_recursive_backend.py`
  - `tests/test_phase2_dynamic_extraction_schema.py`
  - `tests/test_planner_backend_contract.py`
  - Result: no linter errors reported.

### Outcome
- Phase-2 acceptance hardening is implemented:
  1. dynamic extraction schema is now used in both extraction and validation (under existing runbook/stage feature-flag gating),
  2. legacy mode remains operational when flag is off,
  3. new catalog tools propagate into extraction schema and missing slots without new tool-specific branches,
  4. unit+integration tests are green.

### Handoff note
- Keep current dynamic required-field merge strategy (union across allowed tools) for phase 2; review strictness before wider rollout.
- Phase 3 should only address patch/revise integration and should not rework phase-2 compatibility bridge unless regression evidence appears.
