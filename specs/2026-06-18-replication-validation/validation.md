# Validation

## Completion Standard

This work item is complete when the package has a rerunnable validation suite
that supports a publishable claim of qualitative replication for the implemented
estimators, with explicit scope limits.

## Checks

- Unit tests:
  - DGP1/DGP2/DGP3 group-strength distributions match the paper descriptions.
  - Reusing the same seed produces identical simulated data and validation
    metrics.
  - Metric aggregation computes bias, MSE, scaled MSE, MAD, finite share, and
    selected-group summaries correctly.
- Integration checks:
  - A small validation run completes from a clean `uv run --no-editable` command.
  - The output CSV and Markdown report are written with settings and seeds.
  - The recommended lightweight validation run
    (`--repetitions 10 --n-groups 40 --n-per-group 120`) passes all qualitative
    checks.
  - The report includes qualitative checks:
    - adaptive selects fewer groups than split-interacted in sparse DGPs,
    - adaptive is close to oracle in well-separated DGP1,
    - adaptive improves over pooled 2SLS in sparse DGPs,
    - diagnostics report selected and threshold-selected groups.
- Package checks:
  - `uv run --group dev pytest -q`
  - `uv run --group dev ruff check .`
  - `uv run --group dev mypy src/adaptiveiv`
  - `uv run --group docs mkdocs build --strict --site-dir /tmp/adaptiveiv-validation-docs`

## Human Review Needed?

Yes for release claims. The automated validation can establish that the suite is
reproducible and qualitatively consistent with the paper, but the user should
review the final Markdown report before using it in package release notes or
academic-facing documentation.

## Provenance Notes

- Paper source: `../Abadie-Gu-Shen.pdf`, Journal of Econometrics 240 (2024),
  Section 4.
- Current date: 2026-06-18.
- Simulation outputs should record seed, repetitions, grid, package version, and
  command-line arguments.
- Latest workspace validation run:
  - command: `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --group dev python simulations/validate_replication.py --repetitions 10 --n-groups 40 --n-per-group 120 --output-dir validation/outputs/latest`
  - output report: `validation/outputs/latest/report.md`
  - simulation rows: 150
  - qualitative checks: 9/9 passed

## Sign-Off

Done on 2026-06-18.

Final evidence:

- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest -q`
  passed with 38 tests.
- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev ruff check .`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev mypy src/adaptiveiv`
  passed with no issues in 7 source files.
- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group docs mkdocs build --strict --site-dir /tmp/adaptiveiv-validation-docs`
  passed with only the upstream Material for MkDocs warning.
- `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv build` rebuilt both
  wheel and sdist.
- Wheel integrity check passed with `unzip -t`.
- Sdist inspection showed source, docs, examples, `simulations/validate_replication.py`,
  and `validation/README.md`; internal `specs/` and generated
  `validation/outputs/` are excluded.
- Clean offline wheel smoke from `/tmp` imported `estimate_methods_once`,
  `simulate_paper_section4_dgp`, and `summarize_simulation_results`, then
  produced 5 estimator rows and 5 summary rows.
