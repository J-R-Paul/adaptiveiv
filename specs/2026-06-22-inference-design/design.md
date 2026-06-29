# Inference Design Specification

## Purpose

Inference in `adaptiveiv` should be conservative, explicit, and paper-aligned.
The package should expose standard errors, covariance matrices, p-values, and
confidence intervals only when the reported quantities have a clear sampling
interpretation and have passed validation checks.

The immediate design target is not "make `results.bse` return something." It is
to define when `results.bse` should be meaningful.

The first public inference feature should be the paper-baseline homoskedastic
large-sample covariance for select-and-interact estimators under the maintained
Abadie-Gu-Shen setup:

- one scalar endogenous regressor,
- one scalar excluded instrument,
- group-level first-stage heterogeneity,
- groupwise residualization against intercept and exogenous controls,
- independent observations with the paper's common homoskedastic error
  structure,
- one-sided nonnegative first-stage direction unless an extension is explicitly
  selected,
- sample splitting for adaptive selection,
- no claim of robust or clustered validity.

Everything else should be staged as an extension.

## Paper-Aligned Statistical Contract

The paper separates three issues that the package must also keep separate.

First, pooled 2SLS ignores first-stage heterogeneity and can be inefficient.
Fully interacted 2SLS uses group-specific instrument interactions and is
first-order efficient under the paper's homoskedastic setup, but may suffer
many-IV bias when the number of groups is large relative to sample size.

Second, selecting groups based on first-stage strength and then running pooled
2SLS on the selected sample is not a valid shortcut. The paper shows that this
select-and-pool approach can violate the exclusion restriction at a rate that
breaks conventional inference. The package should not implement select-and-pool
as a public estimator, and validation should include it only as a negative
control if useful.

Third, select-and-interact is different. It keeps the interacted-instrument
structure while selecting groups with strong first-stage relationships. Under
the paper's assumptions, fixed-threshold select-and-interact and split-sample
select-and-interact have conventional first-order large-sample inference. The
adaptive estimator chooses the threshold to optimize higher-order asymptotic MSE
for the split-sample select-and-interact estimator. Inference should therefore
be attached to the select-and-interact moment structure, not to a naive selected
pooled regression.

## Inferential Target

The package should define inferential quantities around the scalar coefficient
reported in `results.params[endogenous]`.

For the adaptive estimator with one split repetition, the point estimate is the
average of two cross-fit components:

```text
beta_hat = (beta_hat_a + beta_hat_b) / 2
```

Each component uses one split for threshold selection and first-stage weights,
then uses the other split for estimation. Conditional on the selection split,
the estimation split component is a just-identified IV ratio with generated
group weights. A paper-baseline first-order standard error should use the
select-and-interact moment structure and should not treat selected groups as if
they had been chosen exogenously by the analyst.

For fixed-threshold select-and-interact, the target is the same moment structure
with a user-specified threshold.

For split-interacted estimation, the threshold is effectively `-inf`, so all
usable groups enter the select-and-interact ratio.

For fully interacted 2SLS, the target is the full-sample interacted 2SLS
coefficient. This benchmark can be checked against `linearmodels` or an
equivalent explicit 2SLS calculation.

For pooled 2SLS, the target is the conventional pooled 2SLS coefficient.
Pooled inference is useful mostly as a benchmark and should not be used to
justify the adaptive estimator.

## Conditioning Convention

The public documentation must state the conditioning convention.

The first analytic covariance should be a first-order, paper-baseline covariance
conditional on the realized sample split schedule and using the package's
selection rule. It should be described as large-sample inference for the
select-and-interact estimator under the maintained assumptions, not as finite
sample post-selection exact inference.

This convention is acceptable because the paper's argument is asymptotic: the
selection-induced exclusion problem is first-order negligible for
select-and-interact under the stated conditions, while it is not negligible for
select-and-pool.

The package should not claim uniform post-selection validity across weak-group
configurations, mixed-sign first stages, heavy tails, heteroskedasticity,
clustering, or repeated resampling algorithms unless those cases are separately
validated.

## First Public Covariance Mode

The first public covariance mode should be named plainly. Recommended public
names:

- `cov_type="homoskedastic"` as the user-facing name.
- `cov_type="unadjusted"` as an alias if compatibility with statistics package
  naming becomes useful.

The result object should record a more specific internal label, for example:

```text
cov_estimator = "paper_homoskedastic"
```

This avoids an ambiguity: many packages use "homoskedastic" for ordinary IV
standard errors, but here the covariance is attached to a select-and-interact
split estimator.

### Required Assumptions

`cov_type="homoskedastic"` should only be available when all of the following
are true:

- one endogenous variable,
- one excluded instrument,
- scalar coefficient result,
- no observation weights,
- no cluster correction,
- no heteroskedastic-robust request,
- groups have enough observations after residualization,
- selected estimation components have nonzero denominators,
- `selection_rule="positive"` unless the absolute-sign extension has its own
  validation flag,
- `n_splits=1` for the first release of analytic inference.

The `n_splits=1` restriction is important. Repeated split estimates reuse the
same observations across repetitions. The repeated components are not
independent Monte Carlo draws, so the package must not compute a repeated-split
standard error by taking `split_beta_sd / sqrt(number_of_components)`.

## Analytic Variance Object

The spec does not require a final implementation formula, but it does require
the variance object to be built from estimator moments rather than from a
generic OLS regression summary.

For each cross-fit component, define an estimation-split instrument score of the
form:

```text
q_i = rho_hat_from_selection_split[g(i)] * z_resid_i * selected_g
```

where `z_resid_i` is the instrument residualized within group on the intercept
and exogenous controls in the estimation split. The component estimator solves:

```text
sum_i q_i * (Y_i - beta * W_i) = 0
```

A homoskedastic first-order covariance should be based on the sample analog of:

```text
Var(beta_hat_component | selection split)
  = sigma_u^2 * sum_i q_i^2 / (sum_i q_i * W_i)^2
```

with a denominator convention and degrees-of-freedom adjustment specified and
tested.

For the one-repetition split estimator, the final covariance must account for
the fact that the reported coefficient averages two cross-fit components. The
two components use disjoint observations for their own outcome moments, but each
component's selection weights are estimated from the other split. Disjoint
estimation samples are therefore not, by themselves, enough to justify treating
the two component estimates as independent unconditionally.

A later implementation spec must choose and justify one of the following:

- derive that the cross-component covariance is first-order negligible under
  the paper-baseline assumptions and use:

```text
Var((beta_hat_a + beta_hat_b) / 2)
  = (Var(beta_hat_a) + Var(beta_hat_b)) / 4
```

- include a sample analog of the cross-component covariance term; or
- expose only component-level analytic inference until the average-estimator
  covariance is validated.

The residual variance estimate should be tied to the select-and-interact
estimation sample, not to unselected groups. This follows the paper's warning
that unselected groups should not affect the second-stage standard error
through residual-variance estimation.

The design must document the exact degrees-of-freedom rule before
implementation. A conservative first choice is to use the number of estimation
observations contributing to selected groups minus the number of group-specific
control/intercept nuisance terms and the endogenous coefficient, with fallback
to no inference when the degrees of freedom are nonpositive. If this proves too
fragile, use a no-small-sample-correction large-sample variance and report that
choice explicitly.

## Repeated Split-Sample Inference

Repeated split-sample estimation is valuable for reducing dependence on one
random split and for diagnosing stability. It complicates inference.

The package should treat repeated-split inference as unsupported until one of
the following is designed and validated:

1. An influence-function covariance for the average over the fixed split
   schedule, accounting for reuse of observations across repetitions.
2. A bootstrap or resampling covariance that reruns the entire adaptive
   procedure, including splitting, selection, and averaging, with a fixed and
   documented split-randomness convention.
3. A documented recommendation that users use `n_splits=1` for analytic
   standard errors and `n_splits>1` for sensitivity/point-estimate stability
   only.

The first package release with inference should choose option 3 unless option 1
or 2 is validated. This is conservative and honest.

`selection_summary["split_beta_sd"]` should remain a stability diagnostic, not
a standard error.

## Robust and Clustered Extensions

`cov_type="robust"` should not be exposed simply by swapping
`sigma_u^2 * sum(q_i^2)` for `sum(q_i^2 * u_i^2)`. That may be a plausible
sandwich direction, but the interaction between group selection, generated
weights, groupwise residualization, and weak groups needs separate validation.

Before robust inference is public, the spec should be extended to define:

- whether robustness is to observation-level heteroskedasticity within groups,
- whether group-level heteroskedasticity is allowed,
- whether selected and unselected groups affect residual variance,
- whether finite-G behavior is acceptable,
- which Monte Carlo designs demonstrate coverage.

`cov_type="clustered"` is a later extension. If clustering is by the same
variable as the first-stage heterogeneity groups, the asymptotic sequence is
different from treating observations as independent within group. If clustering
is by a different variable, the moment structure becomes more complex. Clustered
inference should therefore remain unsupported until the estimand, asymptotic
sequence, and validation design are written down.

## Absolute-Strength Selection Extension

The paper baseline assumes a one-sided first-stage direction. The current
package allows `selection_rule="absolute"` as an explicit practical extension
for applications with mixed signs.

Inference for `selection_rule="absolute"` should not be automatically inherited
from the paper-baseline covariance. It may be supportable, but it should require
separate validation because the selection event and weighted estimand differ.

Recommended first release behavior:

- `selection_rule="positive"`: eligible for `paper_homoskedastic` inference.
- `selection_rule="absolute"`: point estimates and diagnostics available;
  inferential accessors remain unavailable until extension validation passes.

## Result Object Contract

When inference is unavailable:

- `results.inference_available` is `False`.
- `results.bse`, `results.cov`, and `results.conf_int()` raise
  `NotImplementedError`.
- `results.summary()` prints the coefficient table without standard errors and
  includes a short inference note.
- `results.cov_type` records the request only if the request was accepted as a
  valid no-inference configuration; otherwise `fit()` raises.

When inference is available:

- `results.inference_available` is `True`.
- `results.params` is a `pandas.Series`.
- `results.bse` is a `pandas.Series` with the same index.
- `results.cov` is a `pandas.DataFrame` with matching row and column labels.
- `results.tvalues`, `results.pvalues`, and `results.conf_int()` are computed
  from `params`, `bse`, and a documented reference distribution.
- `results.cov_type` is the public covariance request, such as
  `"homoskedastic"`.
- `results.cov_estimator` records the specific estimator, such as
  `"paper_homoskedastic"`.
- `results.inference_notes` lists assumptions, restrictions, and finite-sample
  warnings.

The default reference distribution should be normal approximation for the first
release unless a finite-sample `t` reference is explicitly justified. If a
finite-sample correction is used, the result object should expose `df_resid` and
the confidence interval calculation should use it consistently.

## Fit API Contract

The fit API should avoid pretending that covariance modes are available before
they are validated.

Recommended target behavior:

```python
result = model.fit(cov_type="homoskedastic")
```

If the requested estimator and options are eligible, this returns inference.

If the requested estimator is not eligible, the package should raise a clear
error by default:

```text
Inference for n_splits > 1 is not yet available. Use cov_type=None for point
estimates and stability diagnostics, or fit with n_splits=1 for
paper_homoskedastic inference.
```

To support point-estimate-only workflows, the package should accept a clear
no-inference request:

```python
model.fit(cov_type=None)
model.fit(cov_type="none")
```

Backward compatibility may require a transition period because the current
package accepts `cov_type="homoskedastic"` but returns no inference. During the
transition, the deprecation path should be:

1. Introduce `cov_type=None` / `"none"` as the explicit point-estimate-only
   mode.
2. Once homoskedastic inference is implemented, make
   `cov_type="homoskedastic"` mean inference is requested.
3. If a configuration cannot supply homoskedastic inference, raise rather than
   silently returning no standard errors.

## Summary Output

The text summary should be boring and explicit.

For inference-enabled results, include:

- estimator name,
- covariance estimator name,
- number of observations,
- number of groups,
- split repetition count,
- selected and threshold-selected group counts,
- coefficient table with standard error, test statistic, p-value, and interval,
- notes on maintained assumptions.

For no-inference results, include:

- estimator name,
- `cov_type: none` or equivalent,
- coefficient table without inferential columns,
- a short note explaining why inference is unavailable.

The summary should not mix stability diagnostics into inferential columns.

## Python Statistics Ecosystem Fit

The package should feel familiar to users of `statsmodels` and `linearmodels`
without implying that every covariance feature those packages expose is valid
here.

Naming should follow common conventions:

- `params`
- `bse`
- `cov`
- `tvalues`
- `pvalues`
- `conf_int(alpha=0.05)`
- `cov_type`
- `df_resid`
- `summary()`

Compatibility should be semantic, not cosmetic. If an inferential quantity is
not valid, the familiar accessor should raise or be absent rather than return a
placeholder. Users should not need to remember that `split_beta_sd` is not a
standard error.

Benchmark estimators can be validated against `linearmodels` where their
specifications coincide. The adaptive estimator should not delegate inference
to `linearmodels` because its selection, cross-fit weighting, and diagnostics
are package-specific.

## Validation Requirements

Inference should not be public until all applicable checks pass.

### Algebraic Checks

- In no-selection cases, fully interacted estimates and unadjusted covariance
  should match an explicit interacted 2SLS calculation within numerical
  tolerance.
- Split-interacted estimates with all groups selected should match the
  select-and-interact moment formula.
- The analytic covariance should be invariant to groupwise rescaling of the
  instrument in cases where the coefficient estimate is invariant.
- Standard errors should be unchanged by row order for fixed `random_state`.
- Components with zero denominator, no selected groups, or nonpositive residual
  degrees of freedom should disable inference with a clear warning/error.

### Monte Carlo Coverage and Size Checks

Use paper-style DGPs with `beta=0` and known truth.

Minimum acceptance checks for paper-baseline homoskedastic inference:

- In DGP1 with well-separated strong/zero groups and normal errors, empirical
  95 percent confidence interval coverage for adaptive `n_splits=1` should be
  close to nominal on a release-sized simulation grid.
- Wald rejection at 5 percent under `beta=0` should be close to nominal for the
  adaptive estimator in well-separated normal-error designs.
- Fully interacted 2SLS should show the expected many-IV size distortion in
  difficult cases, so the validation suite can detect known finite-sample
  problems rather than only pass easy cases.
- Select-and-pool, if included as a private validation comparator, should
  over-reject in the paper's warning design. This is a negative control.
- Heteroskedastic or heavy-tailed variants should be reported separately and
  should not be used to validate homoskedastic inference unless they pass for a
  theoretically defensible reason.

### Repeated-Split Checks

Until repeated-split inference is implemented, tests should assert that:

- `n_splits>1` with `cov_type="homoskedastic"` raises a clear unsupported
  inference error once homoskedastic inference exists.
- `n_splits>1` with `cov_type=None` or `"none"` still returns point estimates
  and stability diagnostics.
- `selection_summary["split_beta_sd"]` is never exposed as `bse`.

### API Checks

- `results.inference_available` switches to `True` only for supported
  configurations.
- `results.bse`, `results.cov`, `results.tvalues`, `results.pvalues`, and
  `conf_int()` have stable pandas indexes.
- `summary()` includes inference notes and covariance labels.
- Unsupported covariance requests raise useful `ValueError` messages.
- Documentation examples show both inference-enabled and point-estimate-only
  usage.

## Documentation Requirements

Documentation should include a dedicated inference page with:

- the maintained assumptions,
- which estimators support inference,
- which covariance modes are unsupported,
- why select-and-pool is not the same as select-and-interact,
- why repeated split stability is not a standard error,
- how to request point-estimate-only fits,
- how to interpret `selection_summary` alongside standard errors.

The README should remain short and should point to the inference page rather
than trying to explain all asymptotics inline.

## Release Gate

Inference should be treated as release-blocking once exposed. Before any release
that returns finite `bse`, the following should be run and recorded:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest -q
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev ruff check .
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev mypy src/adaptiveiv
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group docs mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-docs
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --group dev python simulations/validate_inference.py --preset release
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv build
```

The inference validation report should be reviewed before release notes claim
coverage, size control, or valid p-values.

## Open Design Questions

- Should the first public inference release support benchmark estimator
  inference before adaptive inference, or should all inference arrive together?
- Should `cov_type=None` become the default until inference is implemented, or
  should the current `cov_type="homoskedastic"` default remain for API
  continuity with a deprecation note?
- Should the first analytic covariance use a small-sample degrees-of-freedom
  correction, or a pure large-sample normal approximation?
- Should repeated-split inference be pursued through an influence-function
  covariance or through a full-procedure bootstrap?
- Is `selection_rule="absolute"` important enough to validate in the first
  inference release, or should it remain point-estimate-only?
