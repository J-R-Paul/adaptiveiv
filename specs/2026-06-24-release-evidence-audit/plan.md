# Plan

## Approach

Update the release audit data model minimally. Add a `required` flag to audit
rows, compute readiness only over required rows, and add the reconstructed
paper-table artifact as an optional supporting-evidence row.

## Work Location

- `simulations/audit_release_readiness.py`
- `tests/test_release_readiness.py`
- `tests/test_packaging.py`
- `validation/README.md`
- `specs/log.md`

## Steps

1. Extend release-audit rows with `required`.
2. Add optional reconstructed paper-table evidence to the audit.
3. Update tests for required-gate semantics and packaging exclusions.
4. Refresh release audit output and validation docs.
5. Run tests, lint, build, and no-editable audit checks.
