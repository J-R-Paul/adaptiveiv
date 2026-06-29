# Plan

## Approach

Refactor the current single split path into a helper that returns one
repetition's estimate objects and diagnostics. Loop over that helper in
`fit(n_splits=...)`, using deterministic child seeds from the requested
`random_state`. Concatenate diagnostics and summarize stability in
`selection_summary`.

## Work Locations

- `src/adaptiveiv/model.py`
- `src/adaptiveiv/results.py`
- `src/adaptiveiv/validation.py`
- `tests/test_model_results_api.py`
- README/docs if needed

## Validation

- Targeted repeated-split tests.
- Full pytest, ruff, mypy, docs build.
