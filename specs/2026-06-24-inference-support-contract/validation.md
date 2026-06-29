# Validation

## Evidence

- Red focused test:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py -q`
  failed with missing `AdaptiveIV.inference_support`.
- Green focused test:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py -q`
  passed: 30 tests.
- Full tests:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
  passed: 102 tests.
- Lint:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .` passed.
- Typing:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src` passed.
- Docs:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-support-docs`
  passed.
- Inference validation smoke:
  `UV_CACHE_DIR=/tmp/uv-cache PYTHONDONTWRITEBYTECODE=1 uv run --no-editable --group dev python simulations/validate_inference.py --preset smoke --output-dir validation/outputs/inference`
  passed with 5 simulation rows and 3/3 checks.
- Package build:
  `UV_CACHE_DIR=/tmp/uv-cache uv build --offline` passed.

## Release Audit

`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit`
completed with `Ready: False` and failed gate `paper_table_validation`. This is
expected and unchanged by the inference-support metadata work.
