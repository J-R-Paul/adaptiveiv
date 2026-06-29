# Plan

## Approach

1. Change release-audit tests so exact paper replication failures are
   non-blocking.
2. Update the audit implementation to mark paper replication gates optional and
   add a limitation gate.
3. Update current docs, backlog, and log with the revised release scope.
4. Run focused and full verification, then refresh the release audit.

## Work Location

- `simulations/audit_release_readiness.py`
- `tests/test_release_readiness.py`
- `README.md`
- `docs/limitations.md`
- `validation/README.md`
- `simulations/validate_replication.py`
- `specs/backlog.md`
- `specs/log.md`

## Next Concrete Step

Update docs and project notes to match the new audit contract.
