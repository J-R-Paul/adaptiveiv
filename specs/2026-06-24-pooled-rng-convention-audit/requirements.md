# Requirements

Spec: 2026-06-24-pooled-rng-convention-audit
Created: 2026-06-24
Status: done
Type: numerical validation
Summary: Audit whether DGP1/DGP2 group-strength RNG conventions explain failed pooled chi-square MSE paper-table rows.

## Goal

Test whether the remaining pooled chi-square paper-table MSE mismatches are
caused by recoverable simulation-state conventions rather than purely rare
observation-level tail events. In particular, compare pooled 2SLS MSE under
alternative DGP1/DGP2 group-strength assignment conventions while keeping the
paper-table seeds and pooled IV formula fixed.

## Scope

- Implement a fast pooled-only convention audit for selected paper
  configurations.
- Compare the current convention against fixed-order strengths and same-RNG
  shuffled strengths.
- Report scaled MSE, relative error to paper, max absolute estimate, and
  top-tail seed for each convention.
- Generate CSV and Markdown artifacts.

## Out Of Scope

- Changing the package's default DGP behavior.
- Replacing full estimator validation.
- Declaring paper confirmation based on pooled-only diagnostics.
- Relaxing release readiness.

## Success Looks Like

- The audit reproduces the current pooled scaled MSE under the current
  convention.
- Alternative conventions are reported side by side for failed pooled rows.
- The hard G=100 DGP2 chi-square mismatch remains visible if no convention
  generates a comparable tail event.

## Outcome

Completed on 2026-06-24. Added
`simulations/audit_pooled_rng_conventions.py`, which compares pooled 2SLS MSE
under the current `shuffled_separate` convention, deterministic fixed-order
strengths, and a same-RNG shuffled-strength convention. The failed-row audit
shows that simple DGP1/DGP2 strength-assignment conventions can move the
smaller failed rows but do not explain the Table 3 / DGP2 / chi-square / G=100
pooled MSE target, which remains orders of magnitude above all tested
conventions.
