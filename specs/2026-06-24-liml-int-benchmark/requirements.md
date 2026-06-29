# Requirements

Spec: 2026-06-24-liml-int-benchmark
Created: 2026-06-24
Status: done
Type: estimator / paper-method coverage
Summary: Add the LIML-INT benchmark estimator needed for full paper method coverage.

## Goal

Reduce the original-paper method coverage blocker by adding a validated
fully-interacted LIML benchmark estimator for Section 4-style simulations.

## Scope

- Implement a point-estimate-only `liml_interacted` benchmark using the
  interacted-instrument specification.
- Make it available to validation through `estimate_methods_once(...,
  methods=[...])`.
- Update paper method coverage metadata so `LIML-INT` is marked implemented.
- Keep inference unsupported for the public model API unless separately
  designed.

## Out Of Scope

- Transcribing all LIML-INT paper targets in this slice.
- Adding LIML inference or public `AdaptiveIV.fit(method="liml_interacted")`.
- Implementing UJIVE, IJIVE, or 2SLS-SSL.

## Success Looks Like

- Unit tests verify just-identified LIML equals IV/2SLS.
- Validation tests verify `liml_interacted` can be requested and produces a
  finite row on a paper-style DGP.
- Method coverage moves from 5/9 to 6/9, leaving `2SLS-SSL`, `UJIVE`, and
  `IJIVE` as missing paper methods.

## Outcome

Completed on 2026-06-24. Added a point-estimate-only `liml_interacted`
benchmark estimator and validation-row support. Method coverage now reports
six of nine paper methods implemented. The release audit still fails
`paper_method_coverage` because LIML table targets are not transcribed and
`2SLS-SSL`, `UJIVE`, and `IJIVE` remain unimplemented.
