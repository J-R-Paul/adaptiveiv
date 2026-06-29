# Requirements

Spec: 2026-06-23-inference-request-gating
Created: 2026-06-23
Status: done
Type: inference implementation
Summary: Ensure analytic inference is only computed for explicitly requested and supported covariance fits.

## Goal

Tighten the first inference implementation so `cov_type="none"` is a genuine
point-estimate-only path. The model should not compute hidden
`paper_homoskedastic` quantities when users ask for diagnostics without
inference.

## Scope

- Gate split-repetition inference computation on the resolved covariance
  request.
- Preserve supported `cov_type="homoskedastic"` and `cov_type="unadjusted"`
  behavior.
- Keep repeated-split and absolute-selection inference unavailable unless a
  supported covariance mode is explicitly designed later.
- Add a regression test that fails if point-estimate-only fits compute analytic
  homoskedastic inference internally.

## Out Of Scope

- New covariance estimators.
- Changing the `paper_homoskedastic` variance formula.
- Changing point estimates, thresholds, or split diagnostics.

## Success Looks Like

- `cov_type="none"` returns the same point estimates and diagnostics without
  calling the analytic inference routine.
- Supported homoskedastic fits still expose `bse`, `cov`, p-values,
  confidence intervals, and inference diagnostics.
- Repeated-split fits with `cov_type="none"` remain point-estimate-only
  stability diagnostics.

## Outcome

Completed on 2026-06-23. Split repetitions now receive an explicit
`inference_requested` flag, and `homoskedastic_split_inference` is only called
for supported covariance requests.
