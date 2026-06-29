# Validation

## Required Evidence

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_pooled_rng_convention_audit_reproduces_current_convention tests/test_replication_validation.py::test_pooled_rng_convention_report_renders -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/audit_pooled_rng_conventions.py --config-index 7 --config-index 18 --config-index 19 --output-dir validation/outputs/paper_tables/pooled_rng_convention_audit_failed_rows`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/audit_pooled_rng_conventions.py --config-index 19 --repetitions 20 --output-dir /tmp/adaptiveiv-pooled-rng-convention-smoke`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`

## Current Result

Done.

Red check before implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_pooled_rng_convention_audit_reproduces_current_convention tests/test_replication_validation.py::test_pooled_rng_convention_report_renders -q`: failed at import because `simulations.audit_pooled_rng_conventions` did not exist.

Green checks after implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_pooled_rng_convention_audit_reproduces_current_convention tests/test_replication_validation.py::test_pooled_rng_convention_report_renders -q`: 2 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/audit_pooled_rng_conventions.py --config-index 7 --config-index 18 --config-index 19 --output-dir validation/outputs/paper_tables/pooled_rng_convention_audit_failed_rows`: wrote 9 rows.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/audit_pooled_rng_conventions.py --config-index 19 --repetitions 20 --output-dir /tmp/adaptiveiv-pooled-rng-convention-smoke`: wrote 3 rows.
- `UV_CACHE_DIR=/tmp/uv-cache uv build --offline`: successfully built `dist/adaptiveiv-0.1.0.tar.gz` and `dist/adaptiveiv-0.1.0-py3-none-any.whl`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_packaging.py -q`: 3 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`: 93 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`: all checks passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`: success, no issues in 8 source files.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-pooled-rng-convention-docs`: documentation built successfully; emitted the upstream MkDocs Material advisory warning.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit`: release audit still reports `Ready: False`, failed gate `paper_table_validation`.

Current convention-audit result:

- Config 7: `shuffled_coupled` reduces the pooled MSE relative error from `0.319201` to `0.156518`.
- Config 18: `fixed_order` changes the sign of the pooled MSE deviation and gives relative error `0.287893`; current convention has `-0.324359`.
- Config 19: all three tested conventions remain far below the paper target, with relative errors between `-0.999402` and `-0.997994`.
