# Phase 6 PR9 Follow-up Note

This note records follow-up fixes applied after the initial phase 6 PR review.

## Scope

- Make CLI clarification answer parsing schema-driven (`expected_schema`) instead of key-only heuristics.
- Add tests for grouped pause-help rendering and schema-driven answer parsing.

## Why

- Phase 6 requires tool-agnostic UX behavior; schema-driven parsing avoids hardcoded field names for arrays and booleans.
- Grouped output should be regression-protected at the rendered-output level, not only via helper tests.

## Verification

- `uv run pytest tests/test_phase6_clarification_ux.py tests/test_cli_bug5_pause_flow.py tests/test_supervised_clarification_flow.py -q`
- `uv run pytest tests/test_state_patch.py tests/test_phase2_dynamic_extraction_schema.py tests/test_planner_backend_contract.py tests/test_htn_recursive_backend_integration.py -q`
