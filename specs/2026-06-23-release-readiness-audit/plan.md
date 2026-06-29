# Plan

Status: done

1. Add tests for green, failed-check, and missing-artifact release audit cases.
2. Implement `simulations/audit_release_readiness.py`.
3. Document the audit command in validation docs and README.
4. Run focused tests, full tests, lint, typing, docs, uv no-editable smoke, and
   the audit against current artifacts.
5. Record the final validation result and update `specs/log.md`.
