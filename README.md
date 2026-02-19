# NEURONIUM Agent — OSS library + CLI

Commitment-aware AI Super Agent with **Action Graph (DAG)** planning, hybrid memory (GraphRAG + agentic retrieval), verification critics, typed contracts, and audit/replay trace.

## Quick start (local, no external services)

### 1. Installation

```bash
# Base install (FS CAS + SQLite, OpenAI provider)
pip install -e .

# With Docker sandbox for CodeNode
pip install -e ".[docker]"

# All extras (Postgres, Redis, pgvector, embeddings, dev)
pip install -e ".[all]"
```

### 2. Configuration

Create `neuronium.toml` in the project root (or rely on defaults):

```toml
[project]
name = "neuronium"
data_dir = ".neuronium"

[determinism]
canonical_json = "neuronium-v1"
default_random_seed = 0
llm_temperature = 0.0

[storage]
blob_backend = "fs_cas"
index_backend = "sqlite"

[llm]
provider = "openai"
model = "gpt-4.1-mini"
```

### 3. API key

```bash
export NEURONIUM_OPENAI_API_KEY=sk-...
```

Alternatively (recommended for local development), create a `.env` file in the project root:

```dotenv
NEURONIUM_OPENAI_API_KEY=sk-...
```

The CLI loads `.env` automatically (without overriding already set environment variables).

### 4. Run

```bash
# CLI (human-readable output by default)
neuronium-agent run --objective "Write a fibonacci function in Python" \
    --trace-export ./trace.jsonl

# Verbose output (stdout/stderr previews, critic evidence)
neuronium-agent run -o "Write fibonacci" -v

# Execution summary at the end (plan, verdict, artifacts)
neuronium-agent run -o "Write fibonacci" --summary

# Raw logs for debugging
neuronium-agent run -o "Write fibonacci" --raw-logs

# Python API
from neuronium_agent.api import create_runner
from neuronium_agent.types import RunRequest

runner = create_runner()
handle = runner.start(RunRequest(objective="Write fibonacci"))
status = runner.get_status(handle)
print(status.state)  # COMPLETED
runner.export_trace(handle, "jsonl", "trace.jsonl")
```

---

## Production: Postgres + Redis

### 1. Install extras

```bash
pip install -e ".[postgres,redis]"
```

### 2. `neuronium.toml`

```toml
[storage]
index_backend = "postgres"
postgres_dsn = "postgresql+psycopg://user:pass@localhost:5432/mydb"
postgres_schema = "neuronium_agent"
migrations_auto_apply = true

[queue]
enabled = true
backend = "rq"
redis_url = "redis://localhost:6379/0"
queue_name = "neuronium"
```

### 3. Start worker

```bash
neuronium-agent worker
```

---

## Project structure

```
neuronium_agent/
├── api.py              # Public facade: AgentRunner, create_runner
├── config.py           # Configuration (TOML + env + CLI)
├── types.py            # Public DTOs
├── errors.py           # Error hierarchy
├── _canonical.py       # Canonical JSON, artifact ID
├── core/               # State machine, orchestrator
├── planning/           # HTN → Action Graph (DAG)
├── execution/          # Deterministic DAG executor
├── nodes/              # ModelNode, CodeNode, McpToolNode, ...
├── storage/            # Blob + Index store (FS CAS, SQLite, Postgres)
│   └── migrations/     # SQL migrations (sqlite/, postgres/)
├── trace/              # Recorder, exporter, replay
├── verification/       # Critics (demo, generic, business)
├── memory/             # GraphRAG-lite (chunks + provenance, v0.2)
├── artifacts/          # Artifact rendering, local index
├── recovery/           # Recovery policy, classifier
├── tools/              # MCP, web, export, memory tools
├── schemas/            # Export schemas, registry
├── control/            # Control protocol
├── queue/              # Redis + RQ runner
└── cli/                # CLI entrypoints
tests/
├── test_canonical.py   # Canonical JSON
├── test_config.py      # Config loading
├── test_storage.py     # FS CAS + SQLite
├── test_determinism.py # Same inputs → same trace
├── test_immutability.py# Artifacts are immutable
├── test_api.py         # Full vertical slice
└── ...                 # and other tests
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## CLI commands

| Command | Description |
|---------|-------------|
| `neuronium-agent run -o "..."` | Start agent (human-readable timeline by default) |
| `neuronium-agent run -o "..." --runbook ID` | Start with a specific runbook (default: `super_agent_v0`) |
| `neuronium-agent run -o "..." -v` | Verbose output (stdout/stderr, critic evidence) |
| `neuronium-agent run -o "..." --summary` | Print execution summary after run |
| `neuronium-agent run -o "..." --raw-logs` | Raw logs instead of timeline |
| `neuronium-agent run --trace-id ID` | Resume run from checkpoint |
| `neuronium-agent status --trace-id ID` | Check run status |
| `neuronium-agent control --trace-id ID --command pause` | Control (continue / pause / revise / replan / stop / escalate) |
| `neuronium-agent replay --trace-id ID` | Replay (experimental) |
| `neuronium-agent worker` | Redis+RQ worker |

---

## Extras

| Extra | Dependencies | Purpose |
|-------|--------------|---------|
| `[docker]` | docker | CodeNode sandbox |
| `[postgres]` | psycopg | Production index store |
| `[redis]` | redis, rq | Async queue runner |
| `[pgvector]` | pgvector | Semantic search in Postgres |
| `[embeddings]` | sentence-transformers | Local embeddings |
| `[dev]` | pytest | Tests |
| `[all]` | All of the above | Full install |

---

## Documentation

- **Docs index**: `docs/README.md`
- **Config**: `docs/architecture/CONFIG_SPEC.md`
- **Public API**: `docs/architecture/PUBLIC_API_SPEC.md`
- **Storage schema**: `docs/architecture/STORAGE_SCHEMA_SPEC.md`
- **Implementation Binding**: `docs/architecture/Implementation_Binding_Spec.md`
- **Roadmap**: `docs/roadmap/ROADMAP.md`
- **ADR (planner backend boundary)**: `docs/architecture/ADR_planner_backend_boundary.md`
- **Presentation**: `docs/architecture/Super_Agent_presentation.md`
- **Full architecture spec**: `docs/architecture/AI_Super_Agent_Architecture_Implementation_Specification.md`
