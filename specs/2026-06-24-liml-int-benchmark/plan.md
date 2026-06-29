# Plan

## Approach

Use a self-contained k-class LIML implementation. Residualize the outcome,
endogenous variable, and excluded instruments on exogenous controls, compute the
LIML kappa from the generalized eigenvalue problem, then form the k-class
coefficient. Build interacted excluded instruments by multiplying the scalar
instrument by group indicators.

## Work Location

- `src/adaptiveiv/estimators.py`
- `src/adaptiveiv/validation.py`
- `src/adaptiveiv/paper_benchmarks.py`
- `tests/test_estimators.py`
- `tests/test_replication_validation.py`

## Steps

1. Write failing tests for just-identified LIML and validation-row support.
2. Implement the estimator core and validation integration.
3. Update paper method coverage metadata.
4. Run focused tests, full tests, lint, typing, docs, build, and release audit.
