# Validation

## Required Evidence

- Red release-audit tests fail before implementation.
- Focused release-audit tests pass after implementation.
- Full test suite passes.
- Ruff, mypy, docs build, uv build, clean package smoke, and refreshed release
  audit run successfully.

## Current Result

Done.

## Evidence

- Red focused tests:
  - `PYTHONPATH=. .venv/bin/pytest tests/test_release_readiness.py -q`
  - result before implementation: 6 failed because paper-replication gates were
    still required and the limitation row was absent.
- Focused release tests:
  - `PYTHONPATH=. .venv/bin/pytest tests/test_release_readiness.py -q`
  - result: 6 passed.
- Full test suite:
  - `PYTHONPATH=. .venv/bin/pytest -q`
  - result: 112 passed.
- Static checks:
  - `.venv/bin/ruff check .`
  - result: all checks passed.
  - `.venv/bin/mypy src/adaptiveiv`
  - result: no issues found in 8 source files.
- Docs:
  - `.venv/bin/mkdocs build --strict`
  - result: built successfully; emitted the upstream Material for MkDocs warning.
- Build and package smoke:
  - `UV_CACHE_DIR=/tmp/uv-cache uv build --offline`
  - result: built `adaptiveiv-0.1.0.tar.gz` and `adaptiveiv-0.1.0-py3-none-any.whl`.
  - `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --reinstall-package adaptiveiv --offline --group dev python -c "..."`
  - result: fresh installed package reported 330 paper targets and release audit
    readiness `True` with no failed required gates.
- Refreshed release audit:
  - `PYTHONPATH=. .venv/bin/python simulations/audit_release_readiness.py --output-dir validation/outputs/release_audit`
  - result: `Ready: True`; failed gates: none.
  - non-blocking paper-replication rows still report the numerical table
    discrepancy, stale exact-replication artifact target count, and missing
    external comparators.
