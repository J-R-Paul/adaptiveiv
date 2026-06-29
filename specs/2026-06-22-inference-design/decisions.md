# Decisions

## 2026-06-22 - First Public Inference Target

Decision: The first public inference target should be paper-baseline
homoskedastic large-sample inference for scalar select-and-interact estimators
under `n_splits=1` and `selection_rule="positive"`.

Rationale: This is closest to the paper's maintained assumptions and avoids
claiming validity for robust, clustered, mixed-sign, or repeated-split
extensions before validation exists.

## 2026-06-22 - Repeated Split Stability Is Not Inference

Decision: `selection_summary["split_beta_sd"]` should remain a stability
diagnostic and should not be exposed as `bse`.

Rationale: Repeated split components reuse the same observations. They are not
independent draws from the estimator's sampling distribution.

## 2026-06-22 - No Silent Covariance Fallbacks

Decision: Once homoskedastic inference is implemented, requesting
`cov_type="homoskedastic"` in an unsupported configuration should raise a clear
error rather than returning point estimates with no `bse`.

Rationale: In the Python statistics ecosystem, a supported `cov_type` request
normally means inferential quantities are available. Silent no-inference
fallbacks are too easy to misread.

## 2026-06-22 - Point-Estimate-Only Mode Should Be Explicit

Decision: The package should eventually support `cov_type=None` or
`cov_type="none"` for point estimates and diagnostics without inference.

Rationale: Users need a clean way to run repeated split-sample fits and
unsupported extensions without pretending inference is available.
