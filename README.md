# adaptiveiv

`adaptiveiv` implements adaptive split-sample select-and-interact IV estimators
for settings with group-level first-stage heterogeneity, following Abadie, Gu,
and Shen, "Instrumental variable estimation with first-stage heterogeneity",
Journal of Econometrics 240 (2024).

The package is designed to feel familiar to users of `pandas`, `statsmodels`,
and `linearmodels`: model inputs are dataframes or formulas, and results expose
`params`, `summary()`, diagnostic tables, and paper-baseline homoskedastic
inference for the first supported adaptive select-and-interact configuration.

## Installation

Development uses `uv`:

```bash
uv sync --group dev
```

Run tests:

```bash
uv run --group dev pytest -q
```

Run lint and type checks:

```bash
uv run --group dev ruff check .
uv run --group dev mypy src/adaptiveiv
```

Serve documentation locally:

```bash
uv run --group docs mkdocs serve
```

Build artifacts:

```bash
uv build
```

Run the recommended lightweight replication validation:

```bash
uv run --no-editable --group dev python simulations/validate_replication.py \
  --repetitions 10 \
  --n-groups 40 \
  --n-per-group 120 \
  --n-splits 3 \
  --output-dir validation/outputs/latest
```

Run a smoke check for the first inference slice:

```bash
uv run --no-editable --group dev python simulations/validate_inference.py \
  --preset smoke \
  --output-dir validation/outputs/inference
```

Run the paper-table comparison smoke check:

```bash
uv run --no-editable --group dev python simulations/validate_paper_tables.py \
  --preset smoke \
  --output-dir validation/outputs/paper_tables
```

Use `--preset release` for a 100-repetition review run. For release evidence
against the paper's reported Tables 2-4, use `--preset full`. The full preset
runs the paper's 500 repetitions for all implemented-method table
configurations, so it is intentionally much slower than the smoke and release
presets.

The full run can be split into chunks:

```bash
uv run --no-editable --group dev python simulations/validate_paper_tables.py \
  --preset full \
  --config-start 0 \
  --config-stop 5 \
  --output-dir validation/outputs/paper_tables/chunk_00_05
```

## Quick Start

```python
from adaptiveiv import AdaptiveIV, simulate_paper_dgp

data = simulate_paper_dgp(
    n_groups=20,
    n_per_group=150,
    beta=0.5,
    strong_fraction=0.3,
    weak_fraction=0.2,
    seed=321,
)

model = AdaptiveIV.from_formula(
    "Y ~ 1 + X + [W ~ Z]",
    data=data,
    groups="group",
)
results = model.fit(random_state=99)

print(results.params)
print(results.summary())
print(results.group_diagnostics.head())
```

The direct dataframe API is also available:

```python
model = AdaptiveIV(
    data=data,
    dependent="Y",
    endogenous="W",
    instruments="Z",
    exog=["X"],
    groups="group",
)
results = model.fit(random_state=99)
```

Use `cov_type="none"` when you want point estimates and diagnostics without
standard errors, for example with repeated split-sample fits:

```python
results = model.fit(random_state=99, n_splits=5, cov_type="none")
```

## Estimators

The package distinguishes:

- pooled 2SLS,
- fully interacted 2SLS,
- split-sample fully interacted 2SLS,
- fixed-threshold select-and-interact 2SLS,
- adaptive split-sample select-and-interact 2SLS.

The default `AdaptiveIV.fit()` method is the adaptive split-sample estimator.

## Diagnostics

Adaptive estimates should be audited through:

- `results.split_estimates`,
- `results.thresholds`,
- `results.selected_groups`,
- `results.group_diagnostics`,
- `results.component_diagnostics`,
- `results.selection_summary`.

These tables expose group-level `rho_hat`, `mu_hat`, residualized instrument
variance, selected status, split estimates, thresholds, numerators, and
denominators.

For the adaptive estimator, the split-a estimate uses split-b first-stage
statistics to choose groups, and vice versa. The selector first uses full-sample
order statistics to choose the paper's adaptive `k_hat`. It scales each ordered
group strength as `s_g = mu_hat_g / sqrt(kappa)` under the default positive
first-stage rule, or `abs(mu_hat_g) / sqrt(kappa)` when
`selection_rule="absolute"`. It chooses `k_hat` to minimize
`sigma_u^2 / n * sum_{j > k} s_(j)^2 + 2 * (sigma_u^2 * sigma_v^2 + sigma_uv^2) * k / n`.
Each cross-fit half then selects the top `k_hat` positive-strength groups in the
opposite split. `results.thresholds` records the split-specific selected groups
and the common risk-minimizing `k_hat`. `results.selected_groups` records the
groups actually used after requiring usability in both the selection and
estimation splits; `selection_summary` reports both counts and any post-threshold
drops.

## Inference

The first supported inference mode is `cov_type="homoskedastic"` for
`n_splits=1`, `selection_rule="positive"`, and select-and-interact estimators.
The result exposes `bse`, `std_errors`, `cov`, `cov_params()`, `tvalues`,
`pvalues`, and `conf_int()` with `cov_estimator="paper_homoskedastic"` and
`reference_distribution="normal"`. `conf_int(alpha=...)` returns two-sided
normal-approximation intervals for any `alpha` strictly between 0 and 1.

Use `cov_type="none"` or `cov_type=None` for point-estimate-only workflows.
Repeated split-sample fits (`n_splits>1`) and `selection_rule="absolute"` remain
diagnostic/point-estimate extensions and do not report standard errors.
`selection_summary["split_beta_sd"]` is a stability diagnostic, not a standard
error.

The maintained statistical interpretation follows the paper's baseline setup:
one scalar endogenous regressor, one scalar excluded instrument, group-level
first-stage heterogeneity, groupwise controls, independent groups, and the
paper's positive-threshold selection rule. Applications with mixed first-stage
signs can opt into `selection_rule="absolute"`, which selects on absolute
estimated strength and should be treated as an extension beyond the paper's
baseline selection rule.

## Replication Validation

The package includes a rerunnable qualitative replication suite for the paper's
Section 4 Monte Carlo logic:

```bash
uv run --no-editable --group dev python simulations/validate_replication.py \
  --repetitions 10 \
  --n-groups 40 \
  --n-per-group 120 \
  --n-splits 3 \
  --output-dir validation/outputs/latest
```

The script writes estimator-level simulation rows, summary metrics, qualitative
checks, and a Markdown report. It validates the implemented estimators against
paper-style DGP1, DGP2, and DGP3 scenarios; it does not claim to reproduce
UJIVE, IJIVE, lasso, empirical applications, or broad inference.
DGP3 follows the paper's untruncated normal mixture for nonzero first-stage
effects; the adaptive estimator still applies the paper's positive estimated
first-stage selection rule.

Very small smoke grids can be useful for checking that files are produced, but
their qualitative checks are noisy and should not be treated as release evidence.

For direct numerical comparison to the paper's reported Section 4 tables, run:

```bash
uv run --no-editable --group dev python simulations/validate_paper_tables.py \
  --preset full \
  --output-dir validation/outputs/paper_tables
```

This writes the transcribed paper targets, simulated summary metrics, and
observed-vs-paper deviations for the implemented methods in Tables 2-4:
`2SLS-P`, `2SLS-INT`, `2SLS-SSINT`, `2SLS-INF`, `2SLS-ADPT`, and `LIML-INT`
MAD targets. It does not exactly replicate all original paper tables, likely
because the original simulation seeds/state are unrecovered and `2SLS-SSL`,
`UJIVE`, and `IJIVE` remain external comparators. It does not compare paper
rejection-rate columns because broad benchmark inference is not yet implemented
for every estimator. Summary and comparison artifacts include tail-error
diagnostics so heavy-tailed MSE deviations can be audited separately from median
absolute-error behavior. For MSE rows where the paper target is above the
observed run, comparison artifacts also report the single-tail-event error
magnitude implied by the paper target. The auxiliary
`simulations/diagnose_pooled_tail_seeds.py` script can scan observation-level
seeds for rare pooled denominator events in a selected paper configuration.
The `simulations/diagnose_pooled_mse_targets.py` script reads failed pooled
MSE comparison rows and ranks candidate seeds by closeness to the paper-implied
tail magnitude. The `simulations/audit_pooled_rng_conventions.py` script checks
whether simple DGP1/DGP2 group-strength RNG conventions explain failed pooled
MSE rows. The `simulations/audit_pooled_error_conventions.py` script checks
whether chi-square centering or scaling conventions explain failed pooled MSE
rows. The `simulations/diagnose_pooled_tail_splice.py` script tests whether real
high-tail candidate seeds can bring failed pooled MSE rows inside paper
tolerance under a one-tail counterfactual replacement.

Use `--config-start` and `--config-stop` to run the 30 paper configurations in
auditable batches. Each run writes `config_manifest.csv` with the original
configuration indices included in `simulation_results.csv`.

For DGP3 paper-table validation, group first-stage strengths are fixed within
each paper configuration and reused across Monte Carlo repetitions. The output
records `dgp3_strength_mode` and `dgp3_strength_seed` so this convention is
auditable. Use `--redraw-dgp3-strengths` only for sensitivity checks that redraw
the DGP3 first-stage coefficients every repetition. The Markdown report records
the DGP3 strength mode and any supplied strength-seed controls.

After running chunks, aggregate them into one release report:

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

Audit release readiness from the current validation and packaging artifacts:

```bash
uv run --no-editable --group dev python simulations/audit_release_readiness.py \
  --output-dir validation/outputs/release_audit
```

Use `--no-fail` to write the report while release blockers are still open. The
audit exits nonzero unless required package gates pass: qualitative replication,
inference validation, and distribution artifacts. Exact paper-table numerical
replication, paper-target artifact freshness, and full original-paper method
coverage are reported as non-blocking limitations.

## Inference Validation

The package includes a separate inference validation runner for the first
`paper_homoskedastic` covariance slice:

```bash
uv run --no-editable --group dev python simulations/validate_inference.py \
  --preset release \
  --output-dir validation/outputs/inference
```

This runner checks finite standard-error production, empirical 95 percent
coverage, and 5 percent rejection in a paper-style DGP1 design. The release
preset is still an artifact for review, not a substitute for reading the report.

## Limitations

The first release does not support multiple endogenous regressors, multiple
excluded instruments, nonlinear models, or a scikit-learn-style prediction API.
Robust and clustered covariance estimators require additional theoretical and
simulation validation before being exposed.

## License

`adaptiveiv` is released under the MIT License. See `LICENSE`.

## Example

```bash
uv run --no-editable --group dev python examples/simulation_example.py
```
