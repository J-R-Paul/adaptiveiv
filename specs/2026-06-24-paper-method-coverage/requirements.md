# Requirements

Spec: 2026-06-24-paper-method-coverage
Created: 2026-06-24
Status: done
Type: release validation
Summary: Make full original-paper estimator coverage explicit in release checks.

## Goal

Expose the gap between implemented-method paper-table confirmation and full
original-paper Monte Carlo estimator coverage. The package should not be called
fully paper-confirmed while Tables 2-4 methods such as LIML-INT, UJIVE, IJIVE,
and 2SLS-SSL remain unimplemented or unvalidated.

## Scope

- Add machine-readable paper method coverage metadata.
- Record which Section 4 Monte Carlo method labels are implemented by
  `adaptiveiv`.
- Add a required release-audit gate for full paper method coverage.
- Update tests and validation docs to make the subset/full distinction clear.

## Out Of Scope

- Implementing LIML-INT, UJIVE, IJIVE, or 2SLS-SSL in this slice.
- Transcribing target values for unimplemented methods.
- Relaxing the existing implemented-method paper-table comparison.

## Success Looks Like

- Tests prove that paper method coverage reports all nine paper method labels
  and marks the current missing methods.
- The release audit reports a failing required `paper_method_coverage` gate
  until all paper methods are implemented/validated or explicitly removed from
  the release objective.
- Documentation explains that `paper_targets_matched=300` is implemented-method
  coverage, not full original-table coverage.

## Outcome

Completed on 2026-06-24. `adaptiveiv.paper_benchmarks.paper_method_coverage()`
now reports all nine Section 4 method labels from the paper and marks the five
implemented/validated methods separately from the four missing competitor
methods. The release audit now has a required failing `paper_method_coverage`
gate until `LIML-INT`, `2SLS-SSL`, `UJIVE`, and `IJIVE` are implemented and
validated or the release objective is explicitly narrowed.

Update: the later `2026-06-24-liml-int-benchmark` slice added a point-estimate
implementation for `LIML-INT`. The coverage gate now reports six implemented
methods, but still fails because LIML table targets are not transcribed and
`2SLS-SSL`, `UJIVE`, and `IJIVE` remain unimplemented.
