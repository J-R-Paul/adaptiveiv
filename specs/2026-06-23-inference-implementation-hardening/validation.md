# Validation

## Required Evidence

- `pytest tests/test_estimators.py tests/test_model_results_api.py -q`
- `pytest tests/test_docs_examples.py::test_inference_validation_script_writes_outputs -q`
- `python simulations/validate_inference.py --preset smoke --output-dir validation/outputs/inference`

## Current Result

Done.

Red checks before implementation:

- `pytest tests/test_estimators.py::test_homoskedastic_split_inference_uses_component_specific_residuals -q` failed because component variances used the supplied averaged beta rather than component-specific coefficients.
- `pytest tests/test_model_results_api.py::test_conf_int_supports_arbitrary_normal_approximation_alpha -q` failed because `conf_int(alpha=0.10)` raised `ValueError`.

Green checks after implementation:

- `pytest tests/test_estimators.py::test_homoskedastic_split_inference_uses_component_specific_residuals -q`: 1 passed.
- `pytest tests/test_model_results_api.py::test_conf_int_supports_arbitrary_normal_approximation_alpha -q`: 1 passed.
- `pytest tests/test_estimators.py tests/test_model_results_api.py -q`: 28 passed.
- `pytest tests/test_docs_examples.py::test_inference_validation_script_writes_outputs -q`: 1 passed.
- `python simulations/validate_inference.py --preset smoke --output-dir validation/outputs/inference`: 5 simulation rows, 3/3 checks passed.
- `pytest -q`: 61 passed.
- `ruff check .`: all checks passed.
- `mypy src`: success, no issues in 8 source files.
- `mkdocs build --strict`: documentation built successfully; emitted the upstream MkDocs Material advisory warning.
- `uv build`: succeeded after network escalation for build-system dependency resolution; produced `dist/adaptiveiv-0.1.0.tar.gz` and `dist/adaptiveiv-0.1.0-py3-none-any.whl`.
- `uv run --no-editable --reinstall-package adaptiveiv python -c 'import adaptiveiv; from adaptiveiv.paper_benchmarks import paper_table_targets; print(adaptiveiv.__version__); print(len(paper_table_targets()))'`:
  rebuilt and installed the package, then printed `0.1.0` and `300`.
