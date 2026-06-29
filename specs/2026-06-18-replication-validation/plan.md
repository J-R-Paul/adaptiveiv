# Plan

Depends on: `adaptiveiv/plan.md`
Unblocks: package release confidence; future inference roadmap
Supersedes:

## Approach

Build a two-layer validation suite. The first layer is lightweight and runs in
CI: deterministic DGP checks, reproducibility checks, and a small Monte Carlo
that confirms adaptive selection has the expected qualitative behavior. The
second layer is a publishable rerunnable artifact: a script that can run a
larger grid, write results to `validation/outputs/`, and generate a Markdown
report summarizing the paper-aligned findings.

The suite will be honest about scope: it validates the estimators currently in
the package and compares them to an oracle based on known simulated first-stage
groups. It will not claim to reproduce LIML, UJIVE, IJIVE, lasso, empirical
applications, or inference.

## Where The Work Happens

- `src/adaptiveiv/simulation.py`: paper-style DGP definitions.
- `src/adaptiveiv/validation.py`: metric and scenario helpers if useful.
- `simulations/`: runnable validation scripts.
- `validation/`: committed README/report template and generated output location.
- `tests/`: fast tests for DGPs, reproducibility, metrics, and smoke validation.
- `README.md` and `docs/`: rerun instructions and validation scope.

## Steps

1. Record Section 4 DGP details from the paper in `scratch.md`.
2. Extend simulation helpers to generate DGP1, DGP2, and DGP3 with fixed seeds,
   normal or centered chi-squared errors, and known group strengths.
3. Add oracle/select helper support for simulation validation without expanding
   the public API unnecessarily.
4. Build a simulation runner that accepts grid size, repetitions, seed, and
   output directory, then writes CSV results and a Markdown report.
5. Add fast tests for DGP construction, reproducibility, metric aggregation, and
   a small validation run.
6. Document the validation command and the interpretation of the report.
7. Run full package verification plus a default validation run.

## Risks / Unknowns

- Exact paper table values will not match unless all estimators and all 500-rep
  grids are implemented; this spec targets qualitative package validation.
- Fully interacted 2SLS can be numerically unstable in very sparse first-stage
  scenarios; validation should report instability rather than hiding it.
- Simulation runtimes can grow quickly with `G=200`, `n_g=500`, and many
  repetitions; defaults must be CI-friendly while allowing larger release runs.

## Expected Outputs

- Paper-style DGP constructors.
- Validation runner script.
- CSV metrics file and Markdown report for the default validation run.
- Tests covering reproducibility and expected qualitative behavior.
- Documentation explaining the validation scope.
