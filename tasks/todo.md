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
