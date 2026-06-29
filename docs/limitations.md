# Limitations

The current release is deliberately narrow.

Supported:

- One scalar endogenous regressor.
- One scalar excluded instrument.
- Group-level first-stage heterogeneity.
- Groupwise residualization against an intercept and exogenous controls.
- One-sided positive first-stage selection.
- Adaptive split-sample select-and-interact estimation.
- Pooled and fully interacted comparison estimators.
- Paper-baseline homoskedastic inference for `n_splits=1`,
  `selection_rule="positive"` select-and-interact fits.

Not yet supported:

- Multiple endogenous regressors.
- Multiple excluded instruments.
- Robust or clustered covariance estimators.
- Repeated split-sample inference for `n_splits>1`.
- Inference for `selection_rule="absolute"`.
- Panel-specific estimators beyond group labels and controls.
- A scikit-learn-style prediction API.

Use `cov_type="none"` or `cov_type=None` for point-estimate-only workflows.
Unsupported inference requests raise clear errors rather than returning
placeholder values.

The replication validation suite is intentionally limited to the estimators
implemented in this package. It checks paper-style DGP1, DGP2, and DGP3 Monte
Carlo behavior for pooled, fully interacted, split-interacted, adaptive, and
oracle validation estimators. It does not reproduce UJIVE, IJIVE, lasso, the
empirical applications, or broad inference results.
DGP3 uses the paper's untruncated normal mixture for nonzero first-stage
effects, so simulated relevant groups can occasionally have negative realized
first-stage coefficients.
In paper-table validation, DGP3 first-stage strengths are fixed within each
configuration and reused across Monte Carlo repetitions. The validation outputs
record the strength mode, seed, nonzero count, sum of squares, and realized
strength vector. The paper-table runner can vary DGP3 strength seeds separately
from observation-level simulation seeds through `--dgp3-strength-seed-base` or
`--dgp3-strength-seed`, and can use `--dgp3-strength-seed-map` for
per-configuration reconstruction diagnostics. `--redraw-dgp3-strengths` is
available only for sensitivity checks. The original paper does not publish the
realized DGP3 strength vectors, so exact Table 4 replication remains
conditional on unrecovered simulation state.

The paper-table validation suite transcribes the reported Section 4 targets for
the implemented methods and compares simulated `N x MSE` and `N x MAD` values
against Tables 2-4. It does not exactly replicate all original paper tables,
likely because the original simulation seeds/state are unrecovered and
`2SLS-SSL`, `UJIVE`, and `IJIVE` remain external comparators. It does not
compare rejection-rate columns until benchmark inference is implemented for
every relevant estimator. Heavy-tailed chi-square-error MSE rows can be
dominated by one or two pooled-IV denominator events. Validation reports
therefore include tail diagnostics and, when the paper MSE target is above the
observed run, the single-tail-event error magnitude implied by the paper target.
The repository also includes
`simulations/diagnose_pooled_tail_seeds.py` for bounded seed scans of these
pooled tail events.

The adaptive selector follows the paper's positive-threshold selection rule by
default: it chooses `k_hat` from full-sample order statistics, then each
cross-fit half uses the top `k_hat` positive estimated-strength groups from the
opposite split. Groups with nonpositive estimated first-stage strength are not
selected by the default adaptive rule. `selection_rule="absolute"` is available
as an explicit extension for applications where first-stage signs differ, but it
does not correspond to the paper's baseline selection rule and does not
currently support inference.
