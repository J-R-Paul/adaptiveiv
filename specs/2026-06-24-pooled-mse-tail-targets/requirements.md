# Requirements

Spec: 2026-06-24-pooled-mse-tail-targets
Created: 2026-06-24
Status: done
Type: numerical validation
Summary: Diagnose failed pooled chi-square MSE paper-table rows by scanning for seeds near the paper-implied tail event.

## Goal

Make concrete progress on the remaining paper-table numerical confirmation
blocker. The full combined validation currently fails only because a small
number of pooled chi-square `scaled_mse` rows differ from the paper. For rows
where the paper MSE is above the observed run, the comparison artifact already
computes the single-tail-event absolute error implied by the paper target. Add
a diagnostic that scans real observation-level seeds and ranks candidates by
closeness to that implied tail magnitude.

## Scope

- Read `paper_comparison.csv` from a full paper-table validation artifact.
- Identify pooled `scaled_mse` rows whose relative error exceeds the release
  tolerance.
- Separate rows that need a larger missing tail event from rows where the
  observed run already exceeds the paper target.
- For missing-tail rows, scan candidate seeds using the same fast pooled
  cross-product calculation as `diagnose_pooled_tail_seeds.py`.
- Write target diagnostics, candidate seed rankings, and a Markdown report.

## Out Of Scope

- Replacing full paper-table validation.
- Editing simulated results to force agreement with paper targets.
- Claiming original paper seeds have been recovered.
- Relaxing the release readiness audit.

## Success Looks Like

- Current failed rows are classified clearly.
- DGP2 chi-square missing-tail rows receive ranked seed candidates.
- DGP1 chi-square overshoot is reported as not fixable by adding a larger
  single tail event.
- The diagnostic is covered by focused tests and can be run through
  `uv run --no-editable`.

## Outcome

Completed on 2026-06-24. Added
`simulations/diagnose_pooled_mse_targets.py`, which reads failed pooled
`scaled_mse` comparison rows, classifies them by whether they need a missing
tail event or already exceed the paper target, and ranks real observation-level
seeds by closeness to the paper-implied tail magnitude. The current
1,000-seed diagnostic found a very close G=40 DGP2 chi-square candidate
(`abs_beta_hat=126.252` versus target `124.800`) but did not find a close G=100
DGP2 chi-square candidate, reinforcing that the largest paper MSE target
requires a rarer near-zero-denominator event.
