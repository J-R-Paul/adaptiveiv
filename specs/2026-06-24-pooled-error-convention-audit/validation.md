# Validation

## Required Evidence

- Focused tests for ranking and report rendering pass.
- Running the diagnostic for config indices 7, 18, and 19 writes CSV/Markdown
  artifacts under
  `validation/outputs/paper_tables/pooled_error_convention_audit_failed_rows`.
- Running the diagnostic for all DGP1/DGP2 chi-square configurations writes
  broader convention evidence under
  `validation/outputs/paper_tables/pooled_error_convention_audit`.
- `pytest tests/test_replication_validation.py -q` passes.
- `ruff check .` passes.
- The release audit remains explicit about any unresolved paper-table blocker.

## Current Result

Done.

Red evidence:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_pooled_error_convention_audit_ranks_conventions tests/test_replication_validation.py::test_pooled_error_convention_report_renders -q`
  initially failed with `ModuleNotFoundError` because
  `simulations.audit_pooled_error_conventions` did not exist.

Generated artifacts:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/audit_pooled_error_conventions.py --config-index 7 --config-index 18 --config-index 19 --output-dir validation/outputs/paper_tables/pooled_error_convention_audit_failed_rows`
  wrote 12 rows.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/audit_pooled_error_conventions.py --output-dir validation/outputs/paper_tables/pooled_error_convention_audit`
  wrote 48 rows.
- The failed-row artifact shows raw and centered chi-square conventions coincide
  for configs 7, 18, and 19; standardized conventions worsen all three failed
  rows.

Final verification on 2026-06-24:

- Focused tests passed: 2 tests.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py -q`
  passed: 42 tests.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
  passed: 104 tests.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .` passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src` passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-error-convention-docs`
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv build --offline` passed.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/audit_pooled_error_conventions.py --config-index 7 --repetitions 5 --output-dir /tmp/adaptiveiv-pooled-error-convention-smoke`
  passed and wrote 4 rows.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit`
  completed with `Ready: False` and failed gate `paper_table_validation`.
