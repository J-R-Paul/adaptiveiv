# Plan

## Approach

1. Add failing tests for LIML-INT target transcription, default paper-table
   validation estimator coverage, and release-audit artifact freshness.
2. Extend the paper target construction with a LIML-only MAD helper.
3. Include `liml_interacted` in default validation estimator runs.
4. Add a required release-audit gate comparing artifact `paper_targets.csv`
   row count with `paper_table_targets()`.
5. Update validation documentation, backlog, and spec log.

## Work Location

- `src/adaptiveiv/paper_benchmarks.py`
- `src/adaptiveiv/validation.py`
- `simulations/audit_release_readiness.py`
- `tests/test_replication_validation.py`
- `tests/test_release_readiness.py`
- `tests/test_packaging.py`
- `validation/README.md`
- `specs/backlog.md`
- `specs/log.md`

## Next Concrete Step

Write the red tests and verify that they fail for the expected missing behavior.
