# Requirements

Spec: 2026-06-24-inference-support-contract
Created: 2026-06-24
Status: done
Type: inference implementation
Summary: Expose a machine-readable inference support contract without changing the validated covariance formula.

## Goal

Make the implemented inference design easier to use in the Python statistics
ecosystem by allowing callers to query whether a proposed covariance request is
supported before fitting.

## Scope

- Add an `InferenceSupport` record with support status, resolved method,
  covariance request, covariance-estimator label, reference distribution, and a
  reason string.
- Add `AdaptiveIV.inference_support(...)` for detailed support metadata.
- Add `AdaptiveIV.supports_inference(...)` for a boolean check.
- Export `InferenceSupport` from the package root.
- Document the helpers in the API and inference docs.

## Out Of Scope

- Changing the `paper_homoskedastic` covariance formula.
- Adding robust, clustered, bootstrap, repeated-split, absolute-selection,
  pooled, or fully interacted inference.
- Changing paper-table validation status.

## Success Looks Like

- Supported `cov_type="homoskedastic"` and `"unadjusted"` select-and-interact
  requests report `supported=True`, `cov_estimator="paper_homoskedastic"`, and
  `reference_distribution="normal"`.
- Point-estimate-only and unsupported requests report `supported=False` with a
  clear reason rather than requiring downstream callers to catch fitting errors.
- Existing inference accessors and unsupported-inference failure modes remain
  unchanged.

## Outcome

Completed on 2026-06-24. The package now exposes `InferenceSupport`,
`model.inference_support(...)`, and `model.supports_inference(...)`. The change
adds queryable metadata around the existing inference design without broadening
the covariance contract.
