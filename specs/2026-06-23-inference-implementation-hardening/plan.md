# Plan

## Approach

Make narrow corrections to the existing first inference slice, preserving the
current public eligibility rules.

## Work Locations

- `src/adaptiveiv/estimators.py`: component variance inputs.
- `src/adaptiveiv/results.py`: confidence interval quantiles and alpha
  validation.
- `tests/test_estimators.py`: component-specific variance regression test.
- `tests/test_model_results_api.py`: arbitrary alpha confidence interval test.
- `docs/inference.md` and README/docs references as needed.

## Next Concrete Step

Write failing tests for the two inference API contracts, then implement the
smallest production changes that make those tests pass.
