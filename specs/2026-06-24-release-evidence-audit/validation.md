# Validation

## Required Evidence

- `pytest tests/test_release_readiness.py tests/test_packaging.py -q` passes.
- `pytest -q` passes.
- `ruff check .` passes.
- `mypy src/adaptiveiv` passes.
- `uv build --offline` succeeds.
- No-editable release audit reports reconstructed evidence separately and
  remains `Ready: False` while canonical paper-table validation fails.

## Current Result

Passed for this release-evidence audit slice.

Commands and evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_release_readiness.py tests/test_packaging.py -q
```

Result: `7 passed`.

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build --offline
```

Result: built `dist/adaptiveiv-0.1.0.tar.gz` and
`dist/adaptiveiv-0.1.0-py3-none-any.whl`.

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --group dev python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit
```

Result: `Ready: False`; failed gates: `paper_table_validation`.

Refreshed audit rows:

```text
replication_validation: passed=True, required=True, 9 checks passed
inference_validation: passed=True, required=True, 3 checks passed
paper_table_validation: passed=False, required=True, scaled_mse_within_relative_tolerance max absolute relative error=0.999402
reconstructed_paper_table_evidence: passed=True, required=False, 4 checks passed
distribution_artifacts: passed=True, required=True, sdist and wheel present
```

Additional verification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/mypy src/adaptiveiv
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --group docs mkdocs build --strict --site-dir /tmp/adaptiveiv-docs-check
```

Results: `107 passed`; ruff passed; mypy passed; docs build passed. A direct
sdist scan found no `validation/outputs`, `specs`, `site`, `__pycache__`, or
`.pyc` paths.
