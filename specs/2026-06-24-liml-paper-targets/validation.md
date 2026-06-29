# Validation

## Required Evidence

- Focused red tests fail before implementation for the missing LIML targets and
  missing release-audit freshness gate.
- Focused tests pass after implementation.
- Full test suite passes.
- Static checks pass with `ruff` and `mypy`.
- Documentation build passes.
- Distribution artifacts build with `uv`.
- Clean no-editable package smoke uses `--reinstall-package adaptiveiv` so the
  installed wheel reflects the current local code.
- A paper-table smoke run writes 11 comparison rows for one configuration:
  five existing estimators times two metrics plus one LIML-INT MAD row.

## Current Result

Done.

## Evidence

- Red tests first failed for the expected missing behaviors:
  - default validation methods did not include `liml_interacted`
  - `paper_table_targets()` still had 300 rows
  - `LIML-INT` was still marked as missing targets
  - release audit lacked `paper_table_artifact_freshness`
- Focused green tests:
  - `PYTHONPATH=. .venv/bin/pytest tests/test_replication_validation.py::test_estimate_methods_once_returns_all_validation_estimators tests/test_replication_validation.py::test_paper_table_targets_cover_implemented_section4_configs tests/test_replication_validation.py::test_paper_method_coverage_marks_unimplemented_original_paper_methods tests/test_packaging.py::test_package_imports_from_uv_environment tests/test_release_readiness.py::test_release_readiness_passes_when_required_gates_are_green tests/test_release_readiness.py::test_release_readiness_flags_missing_original_paper_methods tests/test_release_readiness.py::test_release_readiness_fails_when_paper_targets_artifact_is_stale -q`
  - result: 7 passed
- Paper-table smoke:
  - `PYTHONPATH=. .venv/bin/python simulations/validate_paper_tables.py --preset smoke --output-dir validation/outputs/paper_tables/smoke_liml_targets`
  - result: 12 simulation rows, 11 matched comparison rows, 3/3 checks passed
- Full tests:
  - `PYTHONPATH=. .venv/bin/pytest -q`
  - result: 112 passed
- Static checks:
  - `.venv/bin/ruff check .`
  - result: all checks passed
  - `.venv/bin/mypy src/adaptiveiv`
  - result: no issues found in 8 source files
- Docs:
  - `.venv/bin/mkdocs build --strict`
  - result: built successfully; emitted the upstream Material for MkDocs warning
- Build and package smoke:
  - `UV_CACHE_DIR=/tmp/uv-cache uv build --offline`
  - result: built `adaptiveiv-0.1.0.tar.gz` and `adaptiveiv-0.1.0-py3-none-any.whl`
  - `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --reinstall-package adaptiveiv --offline --group dev python -c "..."`
  - result: fresh installed package reported 330 targets and LIML-INT as implemented and target-transcribed
  - `PYTHONPATH=. .venv/bin/pytest tests/test_packaging.py -q`
  - result: 3 passed
- Release audit:
  - `PYTHONPATH=. .venv/bin/python simulations/audit_release_readiness.py --output-dir validation/outputs/release_audit --no-fail`
  - result: `Ready: False`
  - failed required gates: `paper_table_validation`, `paper_table_artifact_freshness`, `paper_method_coverage`
  - freshness detail: artifact rows=300; current target rows=330
