# Plan

## Approach

Keep this as release-evidence metadata, not estimator implementation. Add method
coverage constants close to the transcribed paper targets, then have the release
audit consume that coverage table.

## Work Location

- `src/adaptiveiv/paper_benchmarks.py`
- `simulations/audit_release_readiness.py`
- `tests/test_replication_validation.py`
- `tests/test_release_readiness.py`
- `validation/README.md`

## Steps

1. Add full Section 4 method labels and coverage table.
2. Add release-audit coverage gate.
3. Add tests for coverage metadata and audit behavior.
4. Update validation documentation.
5. Run focused and full verification.
