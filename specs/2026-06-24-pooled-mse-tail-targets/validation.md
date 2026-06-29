# Validation

## Required Evidence

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_pooled_mse_target_diagnostic_classifies_failed_rows tests/test_replication_validation.py::test_pooled_mse_target_seed_scan_ranks_candidates_by_target_distance -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/diagnose_pooled_mse_targets.py --seed-count 1000 --random-seed 2026062401 --output-dir validation/outputs/paper_tables/pooled_mse_tail_targets_r1000`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/diagnose_pooled_mse_targets.py --seed-count 100 --random-seed 2026062401 --output-dir /tmp/adaptiveiv-pooled-mse-target-smoke`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`

## Current Result

Done.

Red check before implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_pooled_mse_target_diagnostic_classifies_failed_rows tests/test_replication_validation.py::test_pooled_mse_target_seed_scan_ranks_candidates_by_target_distance -q`: failed at import because `simulations.diagnose_pooled_mse_targets` did not exist.

Green checks after implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_pooled_mse_target_diagnostic_classifies_failed_rows tests/test_replication_validation.py::test_pooled_mse_target_seed_scan_ranks_candidates_by_target_distance -q`: 2 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/diagnose_pooled_mse_targets.py --seed-count 1000 --random-seed 2026062401 --output-dir validation/outputs/paper_tables/pooled_mse_tail_targets_r1000`: wrote 3 failed target rows and 20 candidate seed rows.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/diagnose_pooled_mse_targets.py --seed-count 100 --random-seed 2026062401 --output-dir /tmp/adaptiveiv-pooled-mse-target-smoke`: wrote 3 failed target rows and 20 candidate seed rows.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`: 91 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`: all checks passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`: success, no issues in 8 source files.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-pooled-mse-target-docs`: documentation built successfully; emitted the upstream MkDocs Material advisory warning.
- `UV_CACHE_DIR=/tmp/uv-cache uv build --offline`: successfully built `dist/adaptiveiv-0.1.0.tar.gz` and `dist/adaptiveiv-0.1.0-py3-none-any.whl`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_packaging.py -q`: 3 passed.

Current diagnostic result:

- Config 7, Table 2 / DGP1 / chi-square / G=100 pooled MSE is classified as `observed_exceeds_paper`; adding a larger single tail event cannot reconcile it.
- Config 18, Table 3 / DGP2 / chi-square / G=40 pooled MSE is classified as `missing_larger_tail`; the 1,000-seed scan found seed `3585863902` with `abs_beta_hat=126.252` versus target `124.800`.
- Config 19, Table 3 / DGP2 / chi-square / G=100 pooled MSE is classified as `missing_larger_tail`; the 1,000-seed scan's closest candidate had `abs_beta_hat=116.127` versus target `994.649`, so a larger search or better reconstruction of observation-level seeds is still needed.
