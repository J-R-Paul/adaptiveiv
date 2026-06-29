# Replication Validation

This directory is the default home for generated validation outputs.

Run the recommended lightweight validation with:

```bash
uv run --no-editable --group dev python simulations/validate_replication.py \
  --repetitions 10 \
  --n-groups 40 \
  --n-per-group 120 \
  --n-splits 3 \
  --output-dir validation/outputs/latest
```

The command writes:

- `simulation_results.csv`: one row per estimator, scenario, and repetition.
- `summary.csv`: method-level bias, MSE, scaled MSE, MAD, tail-error
  diagnostics, and selection metrics.
- `checks.csv`: qualitative validation checks.
- `report.md`: a short human-readable report.

The validation suite is a qualitative replication of the Abadie, Gu, and Shen
Section 4 Monte Carlo logic for the estimators implemented by this package. It
does not claim to reproduce UJIVE, IJIVE, lasso, empirical applications, or
broad inference.

For quick CI smoke checks, smaller grids can verify that artifacts are written,
but their qualitative pass/fail checks are expected to be noisy.

For direct comparison against the paper's reported Section 4 table values, run:

```bash
uv run --no-editable --group dev python simulations/validate_paper_tables.py \
  --preset smoke \
  --output-dir validation/outputs/paper_tables
```

Use `--preset release` for a 100-repetition review run. Use `--preset full` for
release evidence. The full preset runs all implemented Table 2-4 configurations
with the paper's 500 simulation repetitions, writes `paper_targets.csv`,
`paper_comparison.csv`, and `report.md`, and should be treated as the numerical
paper-confirmation artifact.

Long runs can be split into configuration chunks:

```bash
uv run --no-editable --group dev python simulations/validate_paper_tables.py \
  --preset full \
  --config-start 0 \
  --config-stop 5 \
  --output-dir validation/outputs/paper_tables/chunk_00_05
```

There are 30 implemented-method paper configurations. Each chunk writes
`config_manifest.csv` and includes `config_index` in `simulation_results.csv`.
For DGP3, `validate_paper_tables.py` fixes the random first-stage strength
vector within each paper configuration and records `dgp3_strength_mode` and
`dgp3_strength_seed` in `simulation_results.csv`. It also writes
`dgp3_strength_nonzero_count`, `dgp3_strength_sum_squares`, and
`dgp3_strength_vector` to both `config_manifest.csv` and
`simulation_results.csv` so Table 4 runs can be audited against the realized
first-stage strength vector. Use `--dgp3-strength-seed-base` to scan indexed
fixed-strength seed schedules without changing observation-level simulation
seeds, or `--dgp3-strength-seed` to force one exact DGP3 strength-vector seed
for targeted Table 4 reconstruction diagnostics. Use
`--dgp3-strength-seed-map` for comma-separated per-configuration overrides such
as `24=11890,26=45610`. Use `--redraw-dgp3-strengths` only for sensitivity
checks. The generated Markdown report records the DGP3 strength mode and any
seed base, exact seed, or per-configuration seed map used for the run.

Aggregate completed chunks with:

```bash
uv run --no-editable --group dev python simulations/aggregate_paper_table_chunks.py \
  validation/outputs/paper_tables/chunk_00_05 \
  validation/outputs/paper_tables/chunk_05_10 \
  validation/outputs/paper_tables/chunk_10_15 \
  validation/outputs/paper_tables/chunk_15_20 \
  validation/outputs/paper_tables/chunk_20_25 \
  validation/outputs/paper_tables/chunk_25_30 \
  --output-dir validation/outputs/paper_tables/combined
```

The combined report checks paper-target matching, relative deviations, and
configuration coverage across all expected paper configurations. When available,
largest-deviation tables include tail diagnostics such as maximum absolute
error, high absolute-error quantiles, maximum scaled squared error, and the
share of MSE contributed by the largest absolute-error realization. These help
separate systematic deviations from rare-event MSE volatility in heavy-tailed
designs. For `scaled_mse` rows where the paper target is above the observed
run, comparison artifacts also report the single-tail-event magnitude that
would be needed to match the paper target while leaving the rest of the run
unchanged.

The current paper-table target set covers transcribed numerical targets for
`2SLS-P`, `2SLS-INT`, `2SLS-SSINT`, `2SLS-INF`, `2SLS-ADPT`, and `LIML-INT`.
`LIML-INT` contributes `N x MAD` targets only, because the paper explicitly
does not report LIML moments. The original paper's Monte Carlo tables also
report `2SLS-SSL`, `UJIVE`, and `IJIVE`, which remain unimplemented. The release
audit reports this full-method coverage gap as a non-blocking limitation. The
current package does not exactly replicate all original Tables 2-4, likely
because the original simulation seeds/state are unrecovered and those external
comparators remain outside the package.

For pooled chi-square MSE tail investigation, use:

```bash
uv run --no-editable --group dev python simulations/diagnose_pooled_tail_seeds.py \
  --config-index 19 \
  --seed-count 10000 \
  --random-seed 2026062302 \
  --output-dir validation/outputs/paper_tables/pooled_tail_seed_scan_config19_r10000
```

This diagnostic scans observation-level seeds for a single paper configuration
using the same pooled 2SLS convention as the package, but computes only the
cross-products needed for the pooled coefficient. It is intended to audit rare
near-zero denominator events, not to replace the full paper-table validation
runner.

To diagnose the current failed pooled MSE paper-table rows directly, use:

```bash
uv run --no-editable --group dev python simulations/diagnose_pooled_mse_targets.py \
  --seed-count 1000 \
  --random-seed 2026062401 \
  --output-dir validation/outputs/paper_tables/pooled_mse_tail_targets_r1000
```

This reads the combined `paper_comparison.csv`, classifies failed pooled
`scaled_mse` rows, and ranks candidate observation-level seeds by closeness to
the paper-implied single-tail absolute error. Rows where observed MSE already
exceeds the paper target are reported separately because adding a larger tail
event cannot reconcile them.

To check whether DGP1/DGP2 group-strength RNG conventions explain failed pooled
MSE rows, run:

```bash
uv run --no-editable --group dev python simulations/audit_pooled_rng_conventions.py \
  --config-index 7 \
  --config-index 18 \
  --config-index 19 \
  --output-dir validation/outputs/paper_tables/pooled_rng_convention_audit_failed_rows
```

This pooled-only diagnostic compares the current shuffled-separate strength
RNG convention with fixed-order strengths and a same-RNG shuffle. It is useful
for ruling out simple simulation-state conventions, but it does not replace the
full paper-table validation gate.

To check whether chi-square error centering or scaling conventions explain
failed pooled MSE rows, run:

```bash
uv run --no-editable --group dev python simulations/audit_pooled_error_conventions.py \
  --config-index 7 \
  --config-index 18 \
  --config-index 19 \
  --output-dir validation/outputs/paper_tables/pooled_error_convention_audit_failed_rows
```

This pooled-only diagnostic compares centered, raw, standardized, and raw
standardized chi-square draws. In pooled models with an intercept, raw and
centered errors should coincide up to numerical precision. The diagnostic is
useful for ruling out an error-scaling convention mismatch, but it does not
replace the full paper-table validation gate.

To test whether real high-tail candidate seeds can reconcile failed pooled MSE
rows by one-tail replacement, run:

```bash
uv run --no-editable --group dev python simulations/diagnose_pooled_tail_splice.py \
  --candidate-csv validation/outputs/paper_tables/pooled_mse_tail_targets_r1000/pooled_mse_target_seed_candidates.csv \
  --candidate-csv validation/outputs/paper_tables/pooled_tail_seed_scan_config19_r10000/pooled_tail_seed_scan.csv \
  --output-dir validation/outputs/paper_tables/pooled_tail_splice_reconstruction
```

This counterfactual diagnostic replaces the current largest pooled squared-error
contribution with a candidate seed's contribution and recomputes scaled MSE.
It is useful for judging whether a failed MSE row is plausible under rare tail
sampling, but it does not prove recovery of the paper's original seeds.

To test candidate observation-level seed replacements at the full estimator-row
level, run:

```bash
uv run --no-editable --group dev python simulations/reconstruct_paper_table_seeds.py \
  --replacement 7:229:1 \
  --replacement 18:40:3585863902 \
  --replacement 19:369:2612616469 \
  --output-dir validation/outputs/paper_tables/full_combined_reconstructed_observation_seeds
```

This diagnostic replaces whole Monte Carlo replication blocks and recomputes all
implemented estimators for those replications before reaggregating the paper
comparison. It is stronger evidence than the pooled-only splice diagnostic, but
it is still a reconstruction artifact rather than proof of the paper's original
simulation seeds.

## Release Readiness Audit

After generating validation outputs and building distribution artifacts, run:

```bash
uv run --no-editable --group dev python simulations/audit_release_readiness.py \
  --output-dir validation/outputs/release_audit
```

The audit reads:

- `validation/outputs/latest/checks.csv`
- `validation/outputs/inference/checks.csv`
- `validation/outputs/paper_tables/full_combined_reconstructed_dgp3/checks.csv`
- `validation/outputs/paper_tables/full_combined_reconstructed_dgp3/paper_targets.csv`
- `validation/outputs/paper_tables/full_combined_reconstructed_observation_seeds/checks.csv`
- `adaptiveiv.paper_benchmarks.paper_method_coverage()`
- `dist/*.tar.gz` and `dist/*.whl`

It writes `checks.csv`, `release_readiness.json`, and `report.md`. The command
exits nonzero when a required release gate is missing or failed. Use
`--no-fail` to refresh the audit report during an in-progress release while
known blockers remain. Exact paper-table numerical replication, paper-target
artifact freshness, full original-paper method coverage, and reconstructed
observation-seed artifacts are reported as non-blocking evidence. The required
release gates are the qualitative package validation, inference validation, and
distribution artifacts.
