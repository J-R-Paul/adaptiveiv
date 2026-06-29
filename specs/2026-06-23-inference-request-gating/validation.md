# Validation

## Required Evidence

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py::test_none_covariance_does_not_compute_hidden_homoskedastic_inference -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py::test_none_covariance_does_not_compute_hidden_homoskedastic_inference tests/test_model_results_api.py::test_homoskedastic_inference_is_available_for_supported_adaptive_fit tests/test_model_results_api.py::test_repeated_splits_require_explicit_no_inference_request -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-request-gating-docs`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/validate_inference.py --preset smoke --output-dir validation/outputs/inference`

## Current Result

Done.

Red check before implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py::test_none_covariance_does_not_compute_hidden_homoskedastic_inference -q`: failed because `cov_type="none"` still called `homoskedastic_split_inference`.

Green focused check after implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py::test_none_covariance_does_not_compute_hidden_homoskedastic_inference tests/test_model_results_api.py::test_homoskedastic_inference_is_available_for_supported_adaptive_fit tests/test_model_results_api.py::test_repeated_splits_require_explicit_no_inference_request -q`: 3 passed.

Full green checks:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`: 86 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`: all checks passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`: success, no issues in 8 source files.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-request-gating-docs`: documentation built successfully; emitted the upstream MkDocs Material advisory warning.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/validate_inference.py --preset smoke --output-dir validation/outputs/inference`: 5 simulation rows, 3/3 checks passed.
- `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/validate_inference.py --preset smoke --output-dir /tmp/adaptiveiv-inference-request-gating-smoke`: 5 simulation rows, 3/3 checks passed.
- `UV_CACHE_DIR=/tmp/uv-cache uv build --offline`: successfully built `dist/adaptiveiv-0.1.0.tar.gz` and `dist/adaptiveiv-0.1.0-py3-none-any.whl`.

Network note:

- `UV_CACHE_DIR=/tmp/uv-cache uv build` failed in the sandbox because DNS access
  to PyPI was unavailable for resolving `hatchling`. Escalated retries timed
  out at the approval layer, so the final build check used `uv build --offline`
  with the existing cache.
