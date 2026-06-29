# Requirements

Spec: 2026-06-22-inference-design
Created: 2026-06-22
Status: done
Type: inference design
Summary: Define the statistical and public API contract for inference in `adaptiveiv` before implementing standard errors, p-values, or confidence intervals.

## Goal

Write a careful inference design specification for the Abadie-Gu-Shen adaptive
IV package. The spec should say what inferential quantities would mean, which
configurations can eventually report them, which assumptions must be visible to
users, and what validation evidence is required before the package exposes
`bse`, `cov`, `pvalues`, or confidence intervals.

## Why This Matters

The package already estimates the adaptive split-sample select-and-interact
coefficient and records rich diagnostics. It deliberately does not report
standard errors. That is the right default until inference is designed and
validated. The paper's central warning is that naive first-stage selection can
invalidate conventional IV inference; the package should not recreate that
problem by making unsupported standard errors look routine.

## Scope

- Define the inference target for:
  - pooled 2SLS benchmarks,
  - fully interacted 2SLS benchmarks,
  - split-sample fully interacted 2SLS,
  - fixed-threshold select-and-interact 2SLS,
  - adaptive split-sample select-and-interact 2SLS,
  - repeated split-sample averages.
- Specify the first public covariance mode and later covariance modes.
- Specify result-object behavior, naming, warnings, and failure modes.
- Specify validation requirements for coverage, test size, and API consistency.
- Clarify what must remain unsupported until separately justified.

## Out Of Scope

- Implementing inference code.
- Deriving a complete published proof beyond the paper-aligned estimator.
- Multiple endogenous regressors or multiple excluded instruments.
- Cluster-robust, heteroskedastic-robust, panel, or survey-weighted inference
  as public features.
- Treating repeated split estimates as independent draws.
- Reporting p-values from the current package before validation is complete.

## Success Looks Like

- `design.md` is detailed enough that implementation can proceed without
  inventing statistical promises ad hoc.
- The design distinguishes paper-baseline homoskedastic inference from robust,
  clustered, bootstrap, and repeated-split extensions.
- Unsupported inference remains an explicit result state rather than a silent
  fallback.
- The validation plan includes both algebraic comparisons and Monte Carlo
  coverage/size checks.
- The spec is aligned with the Python statistics ecosystem without copying
  misleading `statsmodels` or `linearmodels` behavior into unsupported cases.

## Outcome

Completed on 2026-06-22. The design spec is a terminal design artifact for the
current inference-design stage. The first paper-baseline inference slice was
implemented later on 2026-06-22; extended inference modes remain separate future
work.

## Source Notes

- Paper source: `../Abadie-Gu-Shen.pdf`, Journal of Econometrics 240 (2024).
- Current package state after implementation: `AdaptiveIVResults.inference_available`
  is true for supported `paper_homoskedastic` fits and false for explicit
  point-estimate-only or unsupported inference configurations.
- Previous related specs:
  - `specs/2026-06-18-replication-validation/`
  - `specs/2026-06-18-repeated-splits/`
