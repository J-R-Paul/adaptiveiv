# Decisions

## Qualitative Replication Scope

The validation suite targets a publishable package sanity layer, not exact
paper-table reproduction. It implements paper-style DGP1, DGP2, and DGP3 and
compares the estimators currently shipped by the package against an infeasible
oracle based on known simulated first-stage strengths.

Exact reproduction of all Tables 2-4 would require additional estimators that
are not part of the package surface yet: LIML, UJIVE, IJIVE, and lasso. Those
remain out of scope for this work item.

## Recommended Validation Tier

Very small smoke grids are useful for checking file generation, but their
qualitative checks are noisy. The recommended lightweight release check is:

```bash
uv run --no-editable --group dev python simulations/validate_replication.py \
  --repetitions 10 \
  --n-groups 40 \
  --n-per-group 120 \
  --output-dir validation/outputs/latest
```

On 2026-06-18 this produced 150 estimator rows and passed 9/9 qualitative
checks.
