# Requirements

Spec: 2026-06-18-repeated-splits
Created: 2026-06-18
Status: done
Type: estimator generalization
Summary: Add repeated split-sample estimation and stability diagnostics without exposing unsupported inference.

## Goal

Make the adaptive estimator closer to the repeated split-sample procedure
described in Abadie, Gu, and Shen by allowing multiple random sample splits and
averaging the resulting split-specific estimates.

## Scope

- Add `n_splits` to `AdaptiveIV.fit`, defaulting to `1`.
- Preserve the current public output shape for `n_splits=1` where feasible.
- For `n_splits>1`, record per-repetition split estimates, thresholds, selected
  groups, and numerator/denominator diagnostics.
- Add stability summaries: number of repetitions, number of finite split
  estimates, mean and standard deviation of split estimates, and average
  selected/threshold-selected group counts.
- Keep inference unavailable until a separately validated covariance estimator
  exists.

## Out Of Scope

- Standard errors, p-values, confidence intervals, robust covariance, or
  cluster covariance.
- Cross-fitting beyond the current two-way split inside each repetition.
- Parallel execution.

## Success Looks Like

- Tests prove reproducibility for fixed `random_state` and `n_splits`.
- Tests prove `n_splits>1` averages all finite component estimates.
- Diagnostics expose repetition ids and selection stability.
- Existing tests and validation commands still pass.
