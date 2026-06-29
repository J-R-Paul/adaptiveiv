# Scratch

## Paper Section 4 Extraction

Source: `../Abadie-Gu-Shen.pdf`, pages around Section 4, extracted with
`pdftotext -layout`.

The paper's simulations use the simultaneous-equation model with:

- `X_i`, transformed instrument `Z_i`, `v_i`, and `e_i` drawn iid standard
  normal in the normal-error designs.
- `u_i = rho_uv * v_i + sqrt(1 - rho_uv^2) * e_i`.
- `beta = 0`, `theta = 1`, and `gamma = 1`.
- The group first-stage parameter `rho_g` controls instrument relevance.
- Group size fixed at `n_g = 500`.
- `G` varies from 40 to 200 in reported tables.
- Tuning rule `kappa = (log G)^2`.

The paper compares many estimators, but this package currently implements the
following relevant ones:

- pooled 2SLS (`2SLS-P`),
- fully interacted 2SLS (`2SLS-INT`),
- split-sample fully interacted 2SLS (`2SLS-SSINT`, equivalent to threshold
  `-inf`),
- adaptive split-sample select-and-interact (`2SLS-ADPT`),
- fixed-threshold select-and-interact.

For this validation spec, the oracle comparator will be implemented using known
simulated group strengths. It is not a public estimator and should be labeled as
an infeasible validation benchmark.

## DGP Families

DGP1:

- Proportion `p_s` of groups have strong first stage `rho_g = 1`.
- Remaining groups have irrelevant instruments `rho_g = 0`.
- Paper reports `p_s = 0.05` and `p_s = 0.25`.

DGP2:

- Proportion `p_s` of groups have strong first stage `rho_g = 1`.
- Proportion `p_w` of groups have weak first stage `rho_g = 0.2`.
- Remaining groups have `rho_g = 0`.
- Paper reports `p_s = p_w = 0.025` and `p_s = p_w = 0.125`.

DGP3:

- Ninety percent of groups have irrelevant instruments.
- Among the remaining ten percent:
  - half have `rho_g ~ N(0.2, 0.1^2)`,
  - half have `rho_g ~ N(1, 0.25^2)`.
- The paper treats this as a non-separated first-stage design.

Reported metrics:

- empirical MSE, usually scaled as `N * MSE`,
- MAD, reported in the paper as a scaled absolute-error quantity,
- rejection proportions for second-stage t-tests.

Inference is intentionally out of scope for this package release, so validation
will focus on coefficient error, MSE/MAD, finite shares, and selected-group
behavior.
