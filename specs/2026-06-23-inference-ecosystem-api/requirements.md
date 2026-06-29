# Requirements

Spec: 2026-06-23-inference-ecosystem-api
Created: 2026-06-23
Status: done
Type: inference implementation
Summary: Align supported inference results with familiar Python statistics result APIs without broadening the covariance contract.

## Goal

Make `AdaptiveIVResults` easier to use in normal `statsmodels` and
`linearmodels` workflows after the first `paper_homoskedastic` inference slice.

## Scope

- Add a `cov_params()` method matching the common `statsmodels` covariance
  accessor.
- Add a `std_errors` alias matching common `linearmodels` result naming.
- Expose the reference distribution used for p-values and confidence intervals.
- Document the accessors without claiming new covariance modes.

## Out Of Scope

- Robust, clustered, bootstrap, repeated-split, absolute-selection, pooled, or
  fully interacted inference.
- Changing the homoskedastic covariance formula.
- Changing paper-table validation.

## Success Looks Like

- Supported inference fits expose `cov_params()`, `std_errors`, and
  `reference_distribution == "normal"`.
- `cov_params()` returns the same labeled covariance matrix as `cov`.
- Point-estimate-only fits keep raising `NotImplementedError` through these
  inference aliases.
- Focused API tests and relevant docs tests pass.

## Outcome

Completed on 2026-06-23. `AdaptiveIVResults` now exposes `std_errors`,
`cov_params()`, and `reference_distribution` for supported inference fits,
while point-estimate-only results keep the same explicit unavailable-inference
state.
