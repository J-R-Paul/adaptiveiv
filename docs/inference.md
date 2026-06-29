# Inference

`adaptiveiv` exposes inferential quantities only for configurations that match
the first validated inference slice.

## Supported Mode

The supported public covariance request is:

```python
results = model.fit(cov_type="homoskedastic")
```

This is available for scalar select-and-interact fits with:

- `n_splits=1`,
- `selection_rule="positive"`,
- one endogenous variable,
- one excluded instrument,
- no robust or clustered covariance request.

When available, results expose:

- `results.bse`
- `results.std_errors`
- `results.cov`
- `results.cov_params()`
- `results.tvalues`
- `results.pvalues`
- `results.conf_int(alpha=0.05)`
- `results.inference_diagnostics`
- `results.cov_estimator == "paper_homoskedastic"`
- `results.reference_distribution == "normal"`
- `results.inference_notes`

Before fitting, the model can report the inference contract for a proposed
request:

```python
support = model.inference_support(cov_type="homoskedastic")
support.supported
support.cov_estimator
support.reason
```

For code paths that only need a boolean, use
`model.supports_inference(cov_type="homoskedastic")`.

The covariance is computed from the select-and-interact moment structure. For
each cross-fit component, residual variance is estimated using that component's
own select-and-interact coefficient. For the averaged split estimate, the first
implementation treats the
cross-component covariance as first-order negligible under the paper-baseline
assumptions and records that convention in `results.inference_notes`.
`results.inference_diagnostics` exposes the component-level variances and the
final averaged-estimator variance used to construct `results.bse`.

`conf_int(alpha=...)` returns two-sided normal-approximation intervals for any
`alpha` strictly between 0 and 1.
`std_errors` and `cov_params()` are aliases for the same validated quantities as
`bse` and `cov`; they are included for compatibility with common Python
statistics result APIs.

## Point-Estimate-Only Mode

Use `cov_type="none"` or `cov_type=None` when you want estimates and diagnostics
without inference:

```python
results = model.fit(n_splits=5, cov_type="none")
```

This is required for repeated split-sample fits and for
`selection_rule="absolute"`. In these cases, `results.bse`, `results.cov`, and
`results.conf_int()` raise `NotImplementedError`. The inference aliases
`results.std_errors` and `results.cov_params()` follow the same unavailable
state.

## Repeated Splits

Repeated split-sample estimation is a stability and generalization diagnostic.
The repeated split components reuse observations, so
`selection_summary["split_beta_sd"]` is not a standard error and is never
reported as `results.bse`.

## Unsupported Covariance Modes

The package does not yet support:

- robust covariance,
- clustered covariance,
- repeated-split analytic covariance,
- bootstrap covariance,
- inference for `selection_rule="absolute"`,
- benchmark estimator covariance for pooled or fully interacted comparison
  methods.

Unsupported inference requests raise clear errors rather than returning
placeholder values.

## Validation Command

Run the inference validation smoke preset with:

```bash
uv run --no-editable --group dev python simulations/validate_inference.py \
  --preset smoke \
  --output-dir validation/outputs/inference
```

For release review, use:

```bash
uv run --no-editable --group dev python simulations/validate_inference.py \
  --preset release \
  --output-dir validation/outputs/inference
```

The report checks finite inference, 95 percent interval coverage, and 5 percent
test rejection for the first `paper_homoskedastic` inference slice.
