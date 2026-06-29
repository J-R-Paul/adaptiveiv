# adaptiveiv Package Specification

## Purpose

`adaptiveiv` should be a research-grade Python package for estimating instrumental
variable models with group-level first-stage heterogeneity, centered on the
adaptive split-sample select-and-interact estimator of Abadie, Gu, and Shen.

The package should make the estimator usable by applied researchers who already
work with `pandas`, `statsmodels`, `linearmodels`, `patsy` or `formulaic`, and
`uv`-managed Python projects. It should provide transparent diagnostics rather
than a black-box coefficient.

## Scope

The core package should support:

- One scalar endogenous regressor.
- One scalar excluded instrument.
- Group-level first-stage heterogeneity.
- Exogenous controls.
- Group-specific first-stage and second-stage nuisance relationships consistent
  with the paper's setup.
- Adaptive split-sample select-and-interact 2SLS.
- Benchmark estimators needed for comparison and validation.
- Statsmodels-like result objects with inferential quantities where supported.
- Formula and explicit-array/dataframe APIs.

The package does not initially need to support:

- Multiple endogenous regressors.
- Multiple excluded instruments.
- Nonlinear models.
- Panel-specific estimators beyond what can be expressed through groups and
  controls.
- A scikit-learn-first prediction API.

## Statistical Specification

The package should clearly distinguish the following estimators:

- Pooled 2SLS: ignores group-level first-stage heterogeneity.
- Fully interacted 2SLS: interacts the excluded instrument with group indicators.
- Split-sample fully interacted 2SLS: uses sample splitting without adaptive
  thresholding.
- Select-and-interact 2SLS: uses a fixed first-stage threshold.
- Adaptive split-sample select-and-interact 2SLS: chooses the threshold using
  the MSE criterion in Abadie, Gu, and Shen.

The adaptive estimator should expose, at minimum:

- The two split-specific estimates.
- The final averaged estimate.
- The split-specific thresholds.
- The tuning sequence used for kappa.
- The selected groups in each split.
- Group-level `rho_hat`, `mu_hat`, transformed strength measures, and selection
  status.
- Counts of selected, skipped, weak, and unusable groups.
- Denominator and numerator components for auditability.

The package should document the maintained assumptions:

- Observations are independent across groups and conditionally i.i.d. within
  groups, unless a covariance option explicitly relaxes this.
- The first-stage direction is one-sided unless the user opts into an absolute
  strength rule.
- Exogenous controls enter in the way required for the fully interacted
  interpretation.
- Inference is only reported for covariance estimators that the package can
  justify.

## Public API Specification

The package should provide a direct dataframe API:

```python
from adaptiveiv import AdaptiveIV

model = AdaptiveIV(
    data=df,
    dependent="Y",
    endogenous="W",
    instruments="Z",
    exog=["X1", "X2"],
    groups="group",
)

results = model.fit(random_state=123, kappa="log2", cov_type="homoskedastic")
```

The package should also provide a formula API resembling `linearmodels`:

```python
from adaptiveiv import AdaptiveIV

model = AdaptiveIV.from_formula(
    "Y ~ 1 + X1 + X2 + [W ~ Z]",
    data=df,
    groups="group",
)

results = model.fit(random_state=123)
```

The results object should feel familiar to `statsmodels` and `linearmodels`
users:

```python
results.params
results.bse
results.tvalues
results.pvalues
results.conf_int()
results.summary()
results.first_stage
results.group_diagnostics
results.selected_groups
```

Naming should follow Python statistics conventions:

- `dependent`, `endogenous`, `exog`, `instruments`, and `groups` for model
  inputs.
- `params`, `bse`, `cov`, `tvalues`, `pvalues`, and `conf_int` for estimates.
- `fit`, `from_formula`, `summary`, and `predict` only where the behavior is
  meaningful.

## Results and Diagnostics Specification

`AdaptiveIVResults` should include:

- `params`: a `pandas.Series` indexed by endogenous variable name.
- `bse`: a `pandas.Series` when standard errors are available.
- `cov`: a `pandas.DataFrame` covariance matrix when available.
- `tvalues`, `pvalues`, and confidence intervals when inference is supported.
- `nobs`, `ngroups`, `df_model`, and `df_resid` where meaningful.
- `cov_type` and covariance configuration.
- `method` identifying the estimator.
- `random_state`, split identifiers, and reproducibility metadata.
- `group_diagnostics`: one row per group and split.
- `selection_summary`: selected/skipped counts and threshold information.
- `warnings`: structured warnings produced during fitting.

The summary output should include:

- Model formula or explicit variable specification.
- Estimator name.
- Number of observations and groups.
- Number of selected groups per split.
- Kappa and threshold choices.
- Coefficient table.
- Notes on inference assumptions and any unsupported inferential quantities.

## Formula Specification

The primary formula syntax should be:

```text
Y ~ 1 + X1 + X2 + [W ~ Z]
```

The formula parser should:

- Preserve pandas indexes.
- Preserve variable names in output tables.
- Support quoted variable names through the selected formula backend.
- Treat the endogenous variable and excluded instrument explicitly.
- Reject ambiguous formulas with clear errors.
- Handle intercepts consistently with the direct API.

The package should choose either `patsy` or `formulaic` as the formula backend and
document the choice. Compatibility with `linearmodels` formula conventions is a
priority.

## Inference Specification

The package should not present placeholder inferential quantities as if they are
real.

Supported covariance modes should be explicit. Candidate modes:

- `cov_type="homoskedastic"` for the baseline paper setting where implemented
  and validated.
- `cov_type="none"` or `cov_type=None` for point-estimate-only workflows and
  unsupported inference configurations.
- `cov_type="robust"` if justified and tested.
- `cov_type="clustered"` only after the estimand, grouping, and finite-sample
  behavior are documented.

If standard errors are not available for a configuration, `results.summary()`
should say so plainly, and inferential fields should be absent or marked as not
available with a clear exception or warning.

## Validation Specification

The package should be validated against:

- Algebraic unit tests for residualization, group statistics, threshold
  selection, and split-sample aggregation.
- Edge-case tests for missing values, tiny groups, zero instrument variance,
  all groups selected, no groups selected, negative first-stage estimates, and
  unbalanced group sizes.
- Monte Carlo designs based on the Abadie, Gu, and Shen paper.
- Comparisons against `statsmodels` or `linearmodels` for pooled and fully
  interacted 2SLS in cases where the estimators should coincide.
- Reproducibility tests for sample splitting.
- Smoke tests for installed wheels and source distributions.

Validation should prioritize statistical correctness over snapshot-style output
tests.

## Documentation Specification

The documentation should include:

- A README with installation, quick start, estimator overview, and citation.
- A worked simulation example that uses a valid instrument.
- A comparison example showing pooled 2SLS, fully interacted 2SLS, and adaptive
  select-and-interact 2SLS.
- A formula API example.
- A diagnostics example showing selected groups and first-stage strengths.
- A limitations page explaining supported assumptions and unsupported cases.
- API reference pages for the model and results classes.

The examples should be runnable as tests or notebooks so they do not drift.

## Packaging Specification

The package should be managed with `uv`.

Required packaging properties:

- `pyproject.toml` is the single source of package metadata.
- `uv.lock` is committed for reproducible development.
- Runtime dependencies are minimal and justified.
- Development dependencies are separated from runtime dependencies.
- Test, lint, type-check, docs, and build commands are exposed through uv-friendly
  commands or documented scripts.
- Source distribution and wheel builds should include only package code and
  intended metadata.
- Tests should not be shipped inside the runtime package unless deliberately
  exposed as package data.
- Generated artifacts such as `.venv`, `dist`, caches, and `__pycache__` should
  not be part of source control.

Suggested dependency groups:

```toml
[dependency-groups]
dev = [
    "pytest",
    "ruff",
    "mypy",
]
docs = [
    "mkdocs",
    "mkdocs-material",
]
examples = [
    "linearmodels",
    "matplotlib",
]
```

The package should support a current, realistic Python range. Python 3.10 or 3.11
as a lower bound is preferable unless a dependency or project constraint requires
3.12+.

## Ecosystem Fit

The package should interoperate cleanly with:

- `pandas`: primary user-facing data container.
- `numpy`: internal numerical arrays.
- `statsmodels`: familiar result conventions and summaries.
- `linearmodels`: formula style and IV comparison benchmarks.
- `patsy` or `formulaic`: formula parsing.
- `pytest`: test runner.
- `ruff`: linting and formatting.
- `mypy` or `pyright`: type-checking where practical.
- `uv`: environment, dependency, build, and publishing workflow.

The package should avoid introducing a custom data abstraction unless a standard
Python statistics object cannot represent the required information.

## Quality Bar

A release candidate should satisfy:

- The public API is documented.
- The estimator has been checked against the paper's formulas.
- The formula and direct APIs agree on equivalent specifications.
- The package can be installed from a built wheel in a clean environment.
- The test suite passes through `uv`.
- The examples run through `uv`.
- No placeholder standard errors or p-values appear in normal summaries.
- Diagnostics make selection decisions auditable.
- The README tells users when the estimator is appropriate and when it is not.

## Open Design Questions

- Should the default first-stage selection rule require positive `mu_hat`, or
  should the package expose an absolute-strength option for applications where
  first-stage signs differ?
- Should the formula backend be `formulaic` for closer `linearmodels`
  compatibility, or `patsy` for familiarity with `statsmodels`?
- Which additional covariance estimators are theoretically defensible after the
  implemented paper-baseline homoskedastic slice?
- Should benchmark estimators live in the main namespace or under a comparison
  submodule?
- Should the package include paper-style Monte Carlo utilities as public API or
  keep them in examples/tests?
