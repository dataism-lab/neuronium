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
