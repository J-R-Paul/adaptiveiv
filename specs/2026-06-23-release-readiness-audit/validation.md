# Validation

## Required Evidence

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_release_readiness.py -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-release-readiness-docs`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit`

## Current Result

Done.

Red check before implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_release_readiness.py -q`: failed at collection because `simulations.audit_release_readiness` did not exist.

Green checks after implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_release_readiness.py -q`: 3 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit`: wrote the audit report, reported `Ready: False`, failed gate `paper_table_validation`.
- `UV_CACHE_DIR=/tmp/uv-cache uv build --offline`: successfully built `dist/adaptiveiv-0.1.0.tar.gz` and `dist/adaptiveiv-0.1.0-py3-none-any.whl`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`: 89 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`: all checks passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`: success, no issues in 8 source files.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-release-readiness-docs`: documentation built successfully; emitted the upstream MkDocs Material advisory warning.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit`: wrote the audit report, reported `Ready: False`, failed gate `paper_table_validation`.

Current release audit result:

- `replication_validation`: passed, 9 checks.
- `inference_validation`: passed, 3 checks.
- `paper_table_validation`: failed, `scaled_mse_within_relative_tolerance` has max absolute relative error `0.999402` against tolerance `0.25`.
- `distribution_artifacts`: passed, sdist and wheel present.
