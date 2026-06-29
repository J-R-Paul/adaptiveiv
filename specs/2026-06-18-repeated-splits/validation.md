# Validation

## Completion Standard

Repeated split-sample estimation is complete when `n_splits>1` is deterministic,
diagnosed, documented, and does not change default single-split behavior.

## Checks

- `pytest tests/test_model_results_api.py`
- `pytest tests/test_replication_validation.py`
- `pytest tests/test_docs_examples.py`
- full package verification before closeout

## Sign-Off

Done on 2026-06-18.

Final evidence:

- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest -q`
  passed with 42 tests.
- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev ruff check .`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev mypy src/adaptiveiv`
  passed with no issues in 7 source files.
- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group docs mkdocs build --strict --site-dir /tmp/adaptiveiv-repeated-splits-docs`
  passed with only the upstream Material for MkDocs warning.
- Recommended validation run with `--n-splits 3` produced 150 rows and 9/9
  qualitative checks.
- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv build` rebuilt wheel
  and sdist.
- `unzip -t dist/adaptiveiv-0.1.0-py3-none-any.whl` passed.
- Clean offline wheel smoke from `/tmp` fit `n_splits=3`, produced 6 finite
  split estimates, 6 component rows, and kept `inference_available=False`.
