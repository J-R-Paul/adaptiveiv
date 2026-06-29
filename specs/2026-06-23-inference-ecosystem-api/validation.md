# Validation

## Required Evidence

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py::test_results_expose_stats_ecosystem_inference_aliases tests/test_model_results_api.py::test_inference_aliases_raise_when_unavailable -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_docs_examples.py -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-ecosystem-docs`

## Current Result

Done.

Red checks before implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py::test_results_expose_stats_ecosystem_inference_aliases tests/test_model_results_api.py::test_inference_aliases_raise_when_unavailable -q`: 2 failed because `AdaptiveIVResults` had no `std_errors` attribute.

Green checks after implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py::test_results_expose_stats_ecosystem_inference_aliases tests/test_model_results_api.py::test_inference_aliases_raise_when_unavailable -q`: 2 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py -q`: 22 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_docs_examples.py -q`: 5 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`: all checks passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`: success, no issues in 8 source files.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-ecosystem-docs`: documentation built successfully; emitted the upstream MkDocs Material advisory warning.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`: 79 passed.
