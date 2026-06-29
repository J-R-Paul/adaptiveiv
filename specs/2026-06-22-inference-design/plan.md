# Plan

## Approach

This work item is a design specification, not an implementation plan. The
artifact should define the inference contract tightly enough that a later
implementation spec can be written without reopening basic statistical choices.

The design prioritizes:

- paper-baseline homoskedastic inference first,
- explicit no-inference behavior where assumptions are not met,
- separate treatment of repeated split-sample stability and standard errors,
- validation before exposing public inferential accessors.

## Work Locations

- `specs/2026-06-22-inference-design/design.md`
- `specs/2026-06-22-inference-design/requirements.md`
- `specs/2026-06-22-inference-design/validation.md`
- `specs/2026-06-22-inference-design/decisions.md`

Later implementation work would likely touch:

- `src/adaptiveiv/estimators.py`
- `src/adaptiveiv/model.py`
- `src/adaptiveiv/results.py`
- `src/adaptiveiv/validation.py`
- `simulations/validate_inference.py`
- `tests/`
- `docs/`

## Next Concrete Step

After review, create an implementation plan for the first inference slice:
paper-baseline homoskedastic analytic inference for `n_splits=1`,
`selection_rule="positive"`, and scalar select-and-interact estimators.
