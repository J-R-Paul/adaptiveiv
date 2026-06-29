# Validation

## Required Evidence

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py::test_inference_diagnostics_expose_component_and_final_variances tests/test_model_results_api.py::test_inference_diagnostics_raise_when_inference_unavailable tests/test_docs_examples.py::test_inference_validation_script_writes_outputs -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-diagnostics-docs`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/validate_inference.py --preset smoke --output-dir validation/outputs/inference`

## Current Result

Done.

Red checks before implementation:

- `pytest tests/test_model_results_api.py::test_inference_diagnostics_expose_component_and_final_variances tests/test_model_results_api.py::test_inference_diagnostics_raise_when_inference_unavailable -q` failed because `AdaptiveIVResults` had no `inference_diagnostics` property.
- `pytest tests/test_docs_examples.py::test_inference_validation_script_writes_outputs -q` failed because `inference_results.csv` did not include component and averaged variance columns.

Green checks after implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_model_results_api.py::test_inference_diagnostics_expose_component_and_final_variances tests/test_model_results_api.py::test_inference_diagnostics_raise_when_inference_unavailable tests/test_docs_examples.py::test_inference_validation_script_writes_outputs -q`: 3 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q`: 69 passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/ruff check .`: all checks passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mypy src`: success, no issues in 8 source files.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-diagnostics-docs`: documentation built successfully; emitted the upstream MkDocs Material advisory warning.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/validate_inference.py --preset smoke --output-dir validation/outputs/inference`: 5 simulation rows, 3/3 checks passed.
