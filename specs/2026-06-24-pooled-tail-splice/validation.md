# Validation

## Required Evidence

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_pooled_tail_splice_ranks_candidates_by_spliced_paper_error tests/test_replication_validation.py::test_pooled_tail_splice_report_renders -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/diagnose_pooled_tail_splice.py --candidate-csv validation/outputs/paper_tables/pooled_mse_tail_targets_r1000/pooled_mse_target_seed_candidates.csv --candidate-csv validation/outputs/paper_tables/pooled_tail_seed_scan_config19_r10000/pooled_tail_seed_scan.csv --output-dir validation/outputs/paper_tables/pooled_tail_splice_reconstruction`
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/diagnose_pooled_tail_splice.py --candidate-csv validation/outputs/paper_tables/pooled_mse_tail_targets_r1000/pooled_mse_target_seed_candidates.csv --output-dir /tmp/adaptiveiv-pooled-tail-splice-smoke`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`

## Current Result

Done.

Focused red evidence: the two tail-splice tests initially failed because
`simulations/diagnose_pooled_tail_splice.py` did not exist.

Focused green evidence:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_replication_validation.py::test_pooled_tail_splice_ranks_candidates_by_spliced_paper_error tests/test_replication_validation.py::test_pooled_tail_splice_report_renders -q`
  passed after implementation.
- Running the diagnostic with both current candidate artifacts wrote
  `validation/outputs/paper_tables/pooled_tail_splice_reconstruction/report.md`
  with 32 candidate rows and 2 candidates within tolerance.
- The no-editable smoke command passed with 20 rows and 1 candidate within
  tolerance.

Final verification on 2026-06-24:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
  passed: 102 tests.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .` passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src` passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-support-docs`
  passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv build --offline` passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit`
  completed with `Ready: False` and failed gate `paper_table_validation`, as
  expected from the unresolved pooled chi-square MSE paper-table rows.
