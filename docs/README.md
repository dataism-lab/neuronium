# Neuronium Agent — Documentation (public)

This folder contains the documentation that is part of the open-source repository.

## Architecture and specs

- [CONFIG_SPEC.md](architecture/CONFIG_SPEC.md) — Configuration format: TOML, env vars, priorities, defaults (CONFIG_SPEC).
- [PUBLIC_API_SPEC.md](architecture/PUBLIC_API_SPEC.md) — Public API: `AgentRunner`, `create_runner`, DTOs, trace export.
- [STORAGE_SCHEMA_SPEC.md](architecture/STORAGE_SCHEMA_SPEC.md) — Storage schema: runs, artifacts, trace events, migrations.
- [Implementation_Binding_Spec.md](architecture/Implementation_Binding_Spec.md) — Implementation binding: how specs map to code.
- [ADR_planner_backend_boundary.md](architecture/ADR_planner_backend_boundary.md) — ADR: planner backend boundary.
- [Super_Agent_presentation.md](architecture/Super_Agent_presentation.md) — High-level presentation (Plan → Execute → Control → Adapt).
- [AI_Super_Agent_Architecture_Implementation_Specification.md](architecture/AI_Super_Agent_Architecture_Implementation_Specification.md) — Full architecture and implementation specification.

## Roadmap

- [ROADMAP.md](roadmap/ROADMAP.md) — High-level product roadmap and stages.

---

## Internal documentation

Planning, status reports, demo scripts, and detailed task/analysis docs are kept in **`docs/internal/`**. That directory is not committed to the public repo (see `.gitignore`). It has its own [README](internal/README.md) with a full index of all internal documents.
