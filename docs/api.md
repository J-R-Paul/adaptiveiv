# API Reference

## `AdaptiveIV`

```python
AdaptiveIV(
    data,
    dependent,
    endogenous,
    instruments,
    exog=None,
    groups=None,
)
```

Use the direct dataframe API when variable names are already available as
columns. The first public release supports one endogenous regressor and one
excluded instrument.

## `AdaptiveIV.from_formula`

```python
AdaptiveIV.from_formula("Y ~ 1 + X1 + X2 + [W ~ Z]", data=data, groups="group")
```

Formula syntax follows the `linearmodels` IV convention for one endogenous
variable and one excluded instrument. Quoted names through `Q("name")` are
supported. No-intercept formulas are rejected because the estimator residualizes
within group against a group intercept.

## `AdaptiveIVResults`

Core fields:

- `params`
- `bse`
- `std_errors`
- `cov`
- `cov_params()`
- `tvalues`
- `pvalues`
- `conf_int(alpha=0.05)`
- `split_estimates`
- `thresholds`
- `selected_groups`
- `group_diagnostics`
- `component_diagnostics`
- `inference_diagnostics`
- `selection_summary`
- `cov_type`
- `cov_estimator`
- `reference_distribution`
- `inference_available`
- `inference_notes`
- `summary()`

Capability helpers:

- `model.inference_support(...)`
- `model.supports_inference(...)`

`thresholds` stores the split-specific threshold result, including `k_hat`,
`delta`, selected groups, and risk values for adaptive selection. For adaptive
fits, `k_hat` is chosen from the full-sample order statistics and each cross-fit
half uses the top `k_hat` positive-strength groups from the opposite split.
`selected_groups` stores the groups actually used in each split estimate after
groups unusable in either relevant split have been dropped. `selection_summary`
reports both threshold-selected and actually-used counts.

For supported homoskedastic select-and-interact fits, inferential fields return
pandas objects with the same index as `params`. `conf_int(alpha=...)` returns
two-sided normal-approximation intervals for any `alpha` strictly between 0 and
1. `std_errors` aliases `bse`, `cov_params()` aliases `cov`, and
`reference_distribution` records `"normal"`. `inference_diagnostics` returns
one row for each split component plus one
row for the averaged estimator, with component variances, standard errors,
residual degrees of freedom, and the covariance estimator label. For
point-estimate-only fits
(`cov_type="none"` or `None`) and unsupported inference configurations,
inferential fields such as `bse`, `cov`, `pvalues`, and `conf_int()` raise
`NotImplementedError` or `fit()` raises a clear unsupported-inference error.

`model.inference_support(...)` returns an `InferenceSupport` record describing
whether a proposed inference request is supported before fitting. It reports the
resolved method, covariance request, internal covariance estimator label,
reference distribution, and a reason for unsupported requests. Use
`model.supports_inference(...)` when only a boolean answer is needed.

## Benchmark Helpers

- `fit_pooled_2sls`
- `fit_fully_interacted_2sls`

These helpers are intended for validation, comparison, and examples.
