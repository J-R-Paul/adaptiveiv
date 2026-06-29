# Validation

## Required Evidence

- Focused tests for map parsing and row replacement pass.
- The reconstruction script writes a separate output directory containing
  `replacement_manifest.csv`, `simulation_results.csv`, `summary.csv`,
  `paper_comparison.csv`, `checks.csv`, and `report.md`.
- The generated report clearly labels the artifact as reconstructed seed
  evidence rather than recovered original simulation state.
- `pytest tests/test_replication_validation.py -q` passes.
- `ruff check .` passes.

## Current Result

Passed for the reconstruction diagnostic.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python simulations/reconstruct_paper_table_seeds.py \
  --replacement 7:229:1 \
  --replacement 18:40:3585863902 \
  --replacement 19:369:2612616469 \
  --output-dir validation/outputs/paper_tables/full_combined_reconstructed_observation_seeds
```

Output:

```text
Reconstructed paper-table report: validation/outputs/paper_tables/full_combined_reconstructed_observation_seeds/report.md
Replacement rows: 15
Checks passed: 4/4
```

Generated checks:

```text
scaled_mad_within_relative_tolerance: True, max absolute relative error=0.187653
scaled_mse_within_relative_tolerance: True, max absolute relative error=0.236256
paper_targets_matched: True, matched rows=300; expected=300
config_coverage: True, missing=[]; unexpected=[]
```

Interpretation: replacing the three problematic observation-level replications
with real candidate seeds is sufficient for the full implemented estimator
table to pass the existing strict tolerance. This supports the rare-tail
simulation-state explanation for the final pooled chi-square MSE blocker, but
does not prove that these are the paper's original seeds.
