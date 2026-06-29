# Requirements

Spec: 2026-06-23-paper-table-validation
Created: 2026-06-23
Status: in_progress
Type: replication / public-release validation
Summary: Add direct numerical comparison against the implemented-method rows of Abadie, Gu, and Shen Section 4 tables.

## Goal

Make the package's Monte Carlo validation more publication-ready by adding a
rerunnable artifact that compares simulated package output against the paper's
reported Tables 2-4 for the estimators implemented in `adaptiveiv`.

## Scope

- Transcribe the paper's reported `N x MSE` and `N x MAD` values for:
  - `2SLS-P`,
  - `2SLS-INT`,
  - `2SLS-SSINT`,
  - `2SLS-INF`,
  - `2SLS-ADPT`.
- Cover DGP1, DGP2, and DGP3 configurations in Tables 2-4.
- Provide a script that simulates the paper configurations, summarizes package
  estimates, and writes observed-vs-paper deviations.
- Keep smoke, release, and full presets separate so CI can check the pipeline
  while release work can run the paper's 500 repetitions.
- Allow the full preset to be run in auditable configuration chunks.
- Provide an aggregation script that combines completed chunks, recomputes
  summary/comparison artifacts, and checks configuration coverage.
- Document the command and limitations.

## Out Of Scope

- LIML, UJIVE, IJIVE, lasso, and empirical-application replications.
- Rejection-rate comparisons for every estimator.
- Claiming exact full-table confirmation before the full preset has been run and
  reviewed.

## Success Looks Like

- Tests verify that paper targets are complete for implemented methods.
- Tests verify that validation summaries can be matched to paper targets.
- The smoke preset writes targets, simulated summaries, comparisons, checks, and
  a Markdown report.
- Chunked runs preserve original paper configuration indices in their manifest
  and simulation rows.
- Aggregated chunk reports expose paper-target checks and missing configuration
  indices.
- Paper-table reports record DGP3 strength-vector provenance, including fixed
  versus redraw mode and any strength-seed controls used for reconstruction
  diagnostics.
- Summary and paper-comparison artifacts expose tail-error diagnostics so
  heavy-tailed MSE failures can be audited rather than treated as opaque
  aggregate deviations.
- Documentation explains that full release evidence requires the full preset.
