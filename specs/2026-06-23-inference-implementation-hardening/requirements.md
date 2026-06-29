# Requirements

Spec: 2026-06-23-inference-implementation-hardening
Created: 2026-06-23
Status: done
Type: inference implementation
Summary: Harden the first public inference implementation against the completed inference design spec.

## Goal

Bring the implemented `paper_homoskedastic` inference slice closer to the
inference design contract and to familiar Python statistics API behavior.

## Scope

- Ensure component-level homoskedastic variances use each split component's own
  select-and-interact coefficient when estimating structural residual variance.
- Support arbitrary two-sided normal-approximation confidence intervals through
  `results.conf_int(alpha=...)`.
- Keep repeated-split, absolute-selection, robust, clustered, bootstrap, pooled,
  and fully interacted inference unsupported unless separately validated.
- Update user-facing documentation for the supported confidence interval
  behavior.

## Out Of Scope

- New robust, clustered, bootstrap, or repeated-split covariance modes.
- Changing the inference eligibility set.
- Revalidating the entire paper-table replication suite.

## Success Looks Like

- Tests fail before implementation for component-specific residual variance and
  non-95 percent confidence intervals.
- Focused estimator and result API tests pass after implementation.
- Inference smoke validation still writes finite `paper_homoskedastic` outputs.

## Outcome

Completed on 2026-06-23. The implemented inference slice now uses
component-specific residual variance for each cross-fit component and supports
arbitrary two-sided normal-approximation confidence intervals through
`conf_int(alpha=...)`.
