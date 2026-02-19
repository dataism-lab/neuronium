# Contributing

Thanks for your interest in Neuronium Agent. Below is how to set up your environment, run tests, and submit changes.

## Environment and setup

- **Python:** 3.11 or 3.12.
- A virtual environment is recommended (`venv` or `uv`).

```bash
# Clone
git clone https://github.com/dataism-lab/neuronium.git
cd neuronium

# Install in development mode (with tests)
pip install -e ".[dev]"

# Or with uv
uv venv
uv pip install -e ".[dev]"
```

For tests that use Postgres or Docker, install the corresponding extras:

```bash
pip install -e ".[dev,postgres]"   # storage tests with Postgres
pip install -e ".[dev,docker]"     # CodeNode tests in Docker
```

## Running tests

```bash
# All tests
pytest tests/ -v

# Specific file or test
pytest tests/test_config.py -v
pytest tests/test_api.py -v -k "test_run_completes"

# With coverage (if coverage is installed)
pytest tests/ -v --cov=neuronium_agent --cov-report=term-missing
```

Make sure all tests pass locally before submitting a PR.

## Code style and typing

- The project uses **type annotations**. New code should include them.
- Match the existing style: indentation, quotes, line length.
- Imports: standard library first, then third-party, then local; blank line between groups.

## How to propose a change

1. **Issue (optional)** — For bugs or ideas, you can open an issue to discuss the approach.
2. **Fork and branch** — Fork the repo and create a branch from the current `main`.
3. **Commits** — Keep commits logical, small, with clear messages.
4. **Pull Request** — Describe what changed and why; reference related issues if any. Ensure tests are green.

We’ll do our best to respond and give feedback in a reasonable time.

## Codebase layout

- **`neuronium_agent/`** — Main package: core, planning, execution, nodes, storage, trace, CLI.
- **`tests/`** — Pytest tests; file names `test_*.py`, function names `test_*`.
- **`docs/`** — Architecture, specs, demos; when changing API or config, update the relevant docs.

## Configuration and secrets

- Do not commit `.env`, API keys, or passwords. Use `.env.example` as a template.
- Project configuration: `neuronium.toml` and environment variables (see README and `docs/architecture/CONFIG_SPEC.md`).

## Questions

If something is unclear, ask in an issue or in the repository discussions.

Thanks for contributing.
