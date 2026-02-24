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

## 2026-02-24 — Phase 3 preflight: ToolSchemaRegistry placement

### Plan
- Lock phase-3 precondition: move `ToolSchemaRegistry` to shared layer before/at start of revise-patch integration.
- Keep migration low-risk with a backward-compatible shim in planning namespace.
- Validate preflight baseline tests for import boundaries, legacy revise regression, and deterministic patch path.

### Decisions
- Added shared module:
  - `neuronium_agent/schemas/tool_schema_registry.py` (canonical location for registry + pointer helpers).
- Kept backward compatibility:
  - `neuronium_agent/planning/tool_schema_registry.py` converted to shim exporting the shared implementation.
- Updated internal consumers to shared import:
  - `neuronium_agent/planning/missing_slots.py`
  - `neuronium_agent/planning/htn_recursive_backend.py`
- Updated tests to validate shared import boundary while preserving planning shim compatibility:
  - `tests/test_tool_schema_registry.py`
- Updated migration doc with explicit phase-3 preflight placement lock and compatibility bridge policy:
  - `docs/internal/TOOL_AGNOSTIC_MIGRATION_PLAN.md`

### Verification
- Ran preflight baseline:
  - `uv run pytest tests/test_tool_schema_registry.py tests/test_missing_slots.py tests/test_state_patch.py tests/test_supervised_clarification_flow.py tests/test_phase2_dynamic_extraction_schema.py tests/test_planner_backend_contract.py -q`
- Result:
  - `27 passed in 0.54s`
- Lint check:
  - no linter errors in edited files.

### Outcome
- Tool schema contract logic is now shared-ready for phase-3 second consumer (`orchestrator/revise`) without cross-module planning coupling.
- Backward compatibility preserved through planning shim.
- Preflight acceptance baseline is green and documented.

### Handoff note
- Phase 3 implementation should consume `ToolSchemaRegistry` from `neuronium_agent.schemas.tool_schema_registry`.
- Keep legacy `answers` support via bridge during phase 3; remove legacy path only after dedicated compatibility acceptance.

## 2026-02-24 — Phase 3 implementation: revise/merge → patch-first

### Plan
- Switch `revise` to patch-first payload handling in orchestrator.
- Remove per-field answers merge from metadata inference and replace with `apply_patch`.
- Preserve backward compatibility via legacy `answers` → `patch` bridge and verify with targeted tests.

### Decisions
- `control(revise)` now normalizes payload patch operations via `_normalise_patch_ops(...)`.
- Legacy `answers` are accepted but converted by `_legacy_answers_to_patch(...)`; normalized patch is persisted and recorded as the canonical revise payload.
- Clarification response artifact now stores:
  - `patch` (canonical),
  - `legacy_answers` (only when provided, audit/compat bridge),
  - `answer_text` and `request_artifact_id`.
- `_infer_runbook_metadata(...)` now applies revise updates through `apply_patch(...)` and no longer contains per-field branches (`url/doc_paths/output_text/...`).
- Added helper methods in orchestrator for deterministic bridge behavior:
  - `_normalise_patch_ops(...)`,
  - `_legacy_answers_to_patch(...)`,
  - `_normalize_legacy_answer_value(...)`,
  - `_pointer_escape_token(...)`.

### Verification
- Added/updated tests:
  - `tests/test_supervised_clarification_flow.py`
    - asserts revise event contains canonical `patch`,
    - validates clarification response artifact stores `patch` (+ `legacy_answers` for old path),
    - adds patch-native revise scenario (`payload.patch` without `answers`).
  - `tests/test_state_patch.py`
    - adds escaped JSON pointer token coverage.
- Ran:
  - `uv run pytest tests/test_state_patch.py tests/test_supervised_clarification_flow.py tests/test_cli_bug5_pause_flow.py tests/test_resume_dispatch_and_docs_report_best_effort.py tests/test_phase2_dynamic_extraction_schema.py -q`
- Result:
  - `20 passed in 0.73s`
- Lint check on edited files:
  - `neuronium_agent/core/orchestrator.py`
  - `neuronium_agent/types.py`
  - `tests/test_supervised_clarification_flow.py`
  - `tests/test_state_patch.py`
  - Result: no linter errors.

### Outcome
- Phase 3 core acceptance implemented:
  - revise path is patch-first,
  - metadata/replay resume integration is patch-driven,
  - legacy `answers` remains supported via bridge.
- Orchestrator no longer relies on per-field answer merge logic for revise integration.

### Handoff note
- Next migration step can remove legacy `answers` bridge only after explicit compatibility deprecation window.
- For phase 4, reuse patch-native revise trail and proceed with full schema-driven missing-slots replacement.

## 2026-02-24 — Phase 4 implementation: schema-driven missing slots

### Plan
- Make schema-driven missing computation the primary runtime path in HTN backend.
- Remove active task-type/per-field missing logic branches and keep only a technical fallback.
- Prove compatibility with targeted acceptance/regression tests.

### Decisions
- `neuronium_agent/planning/htn_recursive_backend.py`:
  - `_compute_missing_fields(...)` now always goes through schema-based flow:
    - uses provided `dynamic_input_schema` when available;
    - otherwise builds runtime schema via `_build_runtime_input_schema(...)`.
  - Removed active task-type branch logic (`news_summary/docs_summary/write_file/generic_task`) from missing computation path.
  - Added minimal fallback path (`_compute_missing_fields_fallback(...)`) used only when schema path is unavailable/throws.
  - Missing dedupe is now path-oriented:
    - schema slots are keyed by JSON pointer path;
    - model-signaled critical fields are normalized to path with `_legacy_field_to_pointer(...)`;
    - deterministic output order is sorted by path.
  - Added helper methods:
    - `_build_runtime_input_schema(...)`,
    - `_default_schema_for_field(...)`,
    - `_metadata_has_value_for_key(...)`,
    - `_legacy_field_to_pointer(...)`,
    - `_compute_missing_fields_fallback(...)`.
- Compatibility behavior:
  - Legacy `missing_fields` output shape is preserved for consumers.
  - Generic no-source scenario remains represented via `source`.

### Verification
- Updated tests:
  - `tests/test_phase2_dynamic_extraction_schema.py`
    - renamed legacy fallback assertion to phase-4 schema-driven baseline behavior;
    - added deterministic dedupe test for duplicate `url` signals.
  - `tests/test_planner_backend_contract.py`
    - added test proving missing computation is no longer task-type-driven when schema is absent.
  - `tests/test_htn_recursive_backend_integration.py`
    - added integration test that run pauses with schema-driven missing `source` and emits clarification request.
- Ran acceptance tests:
  - `uv run pytest tests/test_phase2_dynamic_extraction_schema.py tests/test_planner_backend_contract.py tests/test_htn_recursive_backend_integration.py -q`
  - Result: `19 passed in 0.43s`
- Ran regression tests:
  - `uv run pytest tests/test_supervised_clarification_flow.py tests/test_cli_bug5_pause_flow.py -q`
  - Result: `6 passed in 0.42s`
- Lint check:
  - no linter errors in edited files.

### Outcome
- Phase 4 target reached for planner backend path:
  - schema-driven missing is primary;
  - active task-type/per-field missing branches removed;
  - deterministic path-based dedupe added;
  - clarification flow compatibility preserved and verified.

### Handoff note
- Next phase should focus on NL feedback to `StatePatch` (phase 5) without reintroducing field-specific merge rules.
- If stricter schema inference is needed, evolve `_build_runtime_input_schema(...)` incrementally and guard changes with regression tests first.

## 2026-02-24 — Phase 5 implementation: NL feedback -> StatePatch (IBS-driven)

### Plan
- Implement NL clarification answer to patch conversion in control/orchestrator path per IBS `§9.2` and `§11.2`.
- Keep backward compatibility (`answers -> patch` bridge and existing CLI question flow).
- Prove behavior with targeted clarification tests and regression suite.

### Decisions
- Clarification contracts expanded with patch-oriented hints:
  - `ClarificationQuestion.path`
  - `ClarificationQuestion.expected_schema`
- Planner clarification question generation now carries structural hints:
  - fallback path computes pointer + expected schema from field key,
  - model-generated questions accept optional `path`/`expected_schema` and are normalized deterministically.
- `Orchestrator.apply_control(revise)` now supports NL path:
  - if `payload.patch` absent and `payload.answers` absent but `payload.answer_text` present,
  - run model-step conversion (`control_nl_to_patch`) with strict JSON schema output,
  - apply confidence/clarification gate (`needs_clarification` or low confidence => no patch),
  - persist conversion result in decision payload and clarification response artifact.
- CLI supervised clarification loop now supports single free-form answer first (`answer_text`), with fallback to legacy per-question answers when left empty.
- Compatibility preserved:
  - legacy `answers` remains supported and bridged to patch,
  - existing clarify/revise flow remains operational.

### Verification
- Updated tests:
  - `tests/test_supervised_clarification_flow.py`
    - added `answer_text -> patch` revise integration scenario.
- Ran targeted tests:
  - `uv run pytest tests/test_supervised_clarification_flow.py tests/test_cli_bug5_pause_flow.py -q`
  - Result: `7 passed in 0.72s`
- Ran regression tests:
  - `uv run pytest tests/test_state_patch.py tests/test_phase2_dynamic_extraction_schema.py tests/test_planner_backend_contract.py tests/test_htn_recursive_backend_integration.py -q`
  - Result: `23 passed in 0.25s`
- Lint check:
  - no linter errors in edited files.

### Outcome
- Phase 5 core behavior is implemented:
  - NL clarification feedback can be converted into patch operations,
  - patch conversion is schema-constrained and confidence-gated,
  - trace/replay now includes model conversion step and conversion outcome metadata.
- Existing answers-based compatibility path remains intact.

### Handoff note
- Next hardening step: add explicit negative-path tests for low-confidence/needs-clarification conversion outcomes across more runbooks.
- Before removing legacy `answers` bridge, run compatibility window and migration metrics for clients still sending `answers`.

## 2026-02-24 — Phase 5 hardening follow-up (PR quality)

### Plan
- Close post-review gaps in Phase 5 implementation without broad refactor.
- Add confidence-threshold configurability and negative-path coverage for NL->patch conversion.
- Align control-flow semantics wording with actual bounded model conversion behavior.

### Decisions
- Runtime config:
  - added `runtime.nl_patch_min_confidence` with env override `NEURONIUM_RUNTIME_NL_PATCH_MIN_CONFIDENCE`.
- Control semantics wording:
  - `apply_control(...)` doc now explicitly states no user-plan DAG execution while allowing bounded internal conversion step for `revise` (`answer_text -> patch`).
- NL conversion gate:
  - replaced hardcoded `0.5` with `self.config.runtime.nl_patch_min_confidence`.
- Tests hardening in `tests/test_supervised_clarification_flow.py`:
  - invalid JSON from conversion model => `patch=[]`, status `invalid_json`;
  - low-confidence conversion => `patch=[]`, status `needs_clarification`;
  - configurable threshold respected (e.g. `0.95` blocks previous happy-path payload).

### Verification
- Targeted:
  - `uv run pytest tests/test_supervised_clarification_flow.py tests/test_cli_bug5_pause_flow.py -q`
  - Result: `10 passed in 0.92s`
- Regression:
  - `uv run pytest tests/test_state_patch.py tests/test_phase2_dynamic_extraction_schema.py tests/test_planner_backend_contract.py tests/test_htn_recursive_backend_integration.py -q`
  - Result: `23 passed in 0.29s`
- Lint:
  - no errors in updated files.

### Outcome
- Main review concerns are closed in-place:
  - threshold is policy-configurable,
  - negative cases are tested,
  - control semantics text is no longer contradictory to runtime behavior.

### Handoff note
- Next optional step: move NL->patch conversion out of `apply_control` into a dedicated control service if strict “purely declarative control” boundary is required by architecture governance.

## 2026-02-24 — Phase 6 implementation: clarification UX grouping + human prompts

### Plan
- Implement phase 6 incrementally without broad rewrites:
  - improve question UX in planner clarification generation,
  - improve grouped presentation in CLI pause flow,
  - keep backward compatibility of clarification contracts.
- Update phase-6 section in migration doc with explicit implementation/acceptance checklist.
- Prove behavior with targeted UX tests + clarification regressions.

### Decisions
- `neuronium_agent/planning/htn_recursive_backend.py`:
  - added deterministic UX helpers for clarification questions:
    - `_question_group_from_path(...)`,
    - `_sort_questions_for_presentation(...)`,
    - `_human_prompt_for_question(...)`,
    - `_examples_for_schema(...)`,
    - `_question_group_snapshot(...)`.
  - fallback question generation now creates human-friendly prompts and examples from `expected_schema`.
  - model-generated questions are normalized when fields are missing (`prompt/examples/path`), then deterministically sorted by `(group, key)`.
  - clarification request context now includes `question_groups` snapshot for presentation/audit.
- `neuronium_agent/cli/main.py`:
  - added presentation helpers:
    - `_question_group_from_path(...)`,
    - `_question_prompt_with_examples(...)`.
  - supervised interactive clarify flow now prints group headers and includes one short example in prompt.
  - pause help output now prints grouped questions (`inputs`, `tool_args`, `root`) instead of flat list.
- `docs/internal/TOOL_AGNOSTIC_MIGRATION_PLAN.md`:
  - phase 6 expanded with step-by-step implementation plan, explicit done criteria, and acceptance test set.
- Added new tests:
  - `tests/test_phase6_clarification_ux.py`.

### Verification
- Ran:
  - `uv run pytest tests/test_phase6_clarification_ux.py tests/test_supervised_clarification_flow.py tests/test_cli_bug5_pause_flow.py -q`
  - Result: `14 passed in 0.89s`
- Lint check:
  - `neuronium_agent/planning/htn_recursive_backend.py`
  - `neuronium_agent/cli/main.py`
  - `tests/test_phase6_clarification_ux.py`
  - Result: no linter errors.

### Outcome
- Phase 6 UX baseline is implemented:
  - clarification questions are human-friendly and include examples,
  - question ordering/grouping is deterministic and path-aware,
  - CLI pause presentation is grouped and easier to answer.
- Backward compatibility is preserved (`missing_fields`, revise flow, existing tests green).

### Handoff note
- Next iteration can optionally move grouping metadata from `context.question_groups` into a dedicated top-level clarification artifact field if external UI consumers need a stricter contract.
- Before deprecating any legacy formatting, run compatibility checks for existing clients parsing flat question lists.
