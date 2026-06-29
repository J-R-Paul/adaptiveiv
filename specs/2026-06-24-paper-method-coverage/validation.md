# Validation

## Required Evidence

- Focused tests for paper method coverage and release audit pass.
- Full tests pass.
- Ruff, mypy, docs build, and offline build pass.
- No-editable release audit reports `paper_method_coverage` as a failing
  required gate alongside the remaining canonical paper-table gate.

## Current Result

Passed for this coverage-audit slice.

Focused verification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_paper_method_coverage_marks_unimplemented_original_paper_methods tests/test_release_readiness.py -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check src/adaptiveiv/paper_benchmarks.py simulations/audit_release_readiness.py tests/test_replication_validation.py tests/test_release_readiness.py
```

Results: `6 passed`; ruff passed.

Release artifact refresh:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv build --offline
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --group dev python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit
```

Results: sdist and wheel built; release audit reports `Ready: False` with failed
required gates `paper_table_validation` and `paper_method_coverage`.

Original method coverage evidence from this slice:

```text
2SLS-P: implemented
2SLS-INT: implemented
2SLS-SSINT: implemented
2SLS-INF: implemented
2SLS-ADPT: implemented
LIML-INT: missing
2SLS-SSL: missing
UJIVE: missing
IJIVE: missing
```

Later update: `2026-06-24-liml-int-benchmark` added `LIML-INT` as implemented,
leaving LIML target transcription and `2SLS-SSL`, `UJIVE`, and `IJIVE`
implementation as blockers.

Full verification:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/mypy src/adaptiveiv
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --group docs mkdocs build --strict --site-dir /tmp/adaptiveiv-docs-check
```

Results: `109 passed`; ruff passed; mypy passed; docs build passed.
