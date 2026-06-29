# Requirements

Spec: 2026-06-18-replication-validation
Created: 2026-06-18
Status: done
Type: replication / validation
Summary: Build enough reproducible Monte Carlo validation around the Abadie-Gu-Shen estimator to support publishing the package.

## Goal

Create a reproducible validation layer showing that `adaptiveiv` behaves like the
Abadie, Gu, and Shen adaptive split-sample select-and-interact IV estimator in
the paper's central Monte Carlo settings.

## Why This Matters

The package should not be published on the strength of unit tests alone. The
core statistical promise is that adaptive selection exploits first-stage
heterogeneity and avoids weak or irrelevant groups. A publishable package needs
a rerunnable validation suite that shows this behavior across representative
DGPs, records random seeds and output provenance, and makes failures visible.

## Scope

- Encode the paper's Section 4 Monte Carlo DGP families:
  - DGP1: strong-or-zero first-stage groups.
  - DGP2: strong, weak, and zero first-stage groups.
  - DGP3: non-separated first-stage strengths drawn from the paper's
    untruncated normal mixtures.
- Provide a reproducible simulation runner that compares at least:
  - pooled 2SLS,
  - fully interacted 2SLS,
  - split-sample fully interacted 2SLS,
  - adaptive split-sample select-and-interact 2SLS,
  - oracle select-and-interact using known nonzero or strong groups where available.
- Produce machine-readable results and a short human-readable validation report.
- Add package tests for DGP construction, metric calculation, and simulation
  reproducibility.
- Document how to rerun the validation with `uv`.

## Out Of Scope

- Exact reproduction of all paper tables at 500 repetitions for every
  estimator, including LIML, UJIVE, IJIVE, and lasso.
- Empirical application replication using Stephens-Yang or Charles-Stephens data.
- Inference or covariance implementation.
- Claims that unavailable inferential quantities are validated.

## Inputs / Dependencies

- Local source package under `adaptiveiv/src/adaptiveiv`.
- Local paper PDF `../Abadie-Gu-Shen.pdf`.
- Extracted Section 4 details recorded in `scratch.md`.
- Python runtime managed by `uv`.
- Existing runtime dependencies: `numpy`, `pandas`, `statsmodels`, `patsy`.

## Open Questions

- Whether to add optional external estimators such as LIML or lasso later.
- Whether exact table reproduction should become a separate long-running spec.
- Whether validation outputs should be committed permanently or regenerated in
  release workflows only.

## Success Looks Like

- `uv run --group dev pytest -q` passes with validation tests included.
- A documented command runs the validation suite and writes CSV/Markdown outputs.
- The validation report records DGP settings, seeds, repetitions, metrics, and
  qualitative checks against the paper's Section 4 conclusions.
- Acceptance checks pass on a lightweight default run suitable for CI.
- The package documentation points users to the validation command and explains
  what is and is not replicated.
