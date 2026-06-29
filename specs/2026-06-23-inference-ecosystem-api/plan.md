# Plan

## Approach

Use a narrow red-green loop. First add public API tests that encode the
ecosystem-facing contract. Then thread the accessors through `AdaptiveIVResults`
only, leaving model fitting and estimator covariance code unchanged.

## Work Locations

- `tests/test_model_results_api.py`
- `src/adaptiveiv/results.py`
- `docs/api.md`
- `docs/inference.md`
- `README.md`

## Next Concrete Step

Write failing tests for `cov_params()`, `std_errors`, and
`reference_distribution`, then implement the smallest result-object changes.
