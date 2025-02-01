# %%
import numpy as np
import pandas as pd
# from scipy import stats
# from tqdm import tqdm
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


'''Function: `adaptive_split_sample_estimator`

This is the top-level function that ties everything together.
	•	Data Splitting:
The full data is randomly split into two halves (dataₐ and dataᵦ).

	•	Processing Each Split:
The nested helper process_split is applied to each half. For each group
within a split, it performs the residualization of Z̃, computes the
first-stage coefficient and μ values, and then performs both a first-stage
regression (to obtain residuals v) and a second-stage regression (to obtain residuals u).
These residuals and μ values are stored for later use.

	•	Variance Estimation and Adaptive Threshold:
Using the collected residuals, variance estimates are computed. Then, the adaptive
threshold δ is computed for each split by using the other split’s μ values.

	•	Final Estimation:
Finally, using the split-sample IV estimator (via select_and_interact_from_splits),
two estimates are obtained (one from each split) and averaged to yield the final adaptive estimator β̂ₐdₚₜ.

'''
def adaptive_split_sample_estimator(data, groups_col, y_col, w_col, z_col, x_cols, random_state=None):
    """
    Compute the adaptive split-sample select-and-interact estimator β.

      1. Randomly split the full data into two halves (data_a and data_b).
      2. For each split, compute group-level instrument strength (mu_hat) and collect
         residuals from a first-stage and second-stage regression.
      3. Use the residuals to estimate sigma_u^2, sigma_v^2, and sigma_uv.
      4. Compute an adaptive threshold δ_hat for each split using the other split's mu_hat values.
      5. For each split, run the second-stage regression using only groups with tilde_mu ≥ δ_hat
         (the selection rule) and weight by the opposite split’s first-stage coefficient.
      6. Return the average of the split–sample estimates.

    Parameters
    ----------
    data : pandas.DataFrame
        Full dataset containing all variables.
    groups_col : str
        Column name identifying groups.
    y_col : str
        Outcome variable column name.
    w_col : str
        Endogenous regressor column name.
    z_col : str
        Instrument column name.
        TODO: Generalise to multiple instruments.
    x_cols : list of str
        List of exogenous regressor column names.
    random_state : int
        Seed for reproducible random splitting.

    Returns
    -------
    float
        The adaptive split-sample select-and-interact estimator for β.
    """
    # Split the data into two equal parts.
    data_a, data_b = train_test_split(data, test_size=0.5, random_state=random_state)

    # Determine number of groups (G) and set kappa = (log G)^2.
    N_G = data[groups_col].nunique()
    kappa = (np.log(N_G)) ** 2

    # Function to process each split:
    def process_split(split_data):
        group_mu = {}   # dictionary: group -> mu_hat
        u_resids = []   # collect second-stage residuals
        v_resids = []   # collect first-stage residuals

        for group, df in split_data.groupby(groups_col):
            X = df[x_cols].values
            Z_tilde = df[z_col].values
            W = df[w_col].values
            Y = df[y_col].values
            n = len(df)
            d = X.shape[1] if X.ndim > 1 else 1

            if n <= d + 1:
                continue  # Skip groups with insufficient observations

            # Residualize Z_tilde on X (with intercept)
            X_const = sm.add_constant(X, has_constant='skip')
            model_Z = sm.OLS(Z_tilde, X_const)
            res_Z = model_Z.fit()
            Z = res_Z.resid

            if np.allclose(np.dot(Z, Z), 0):
                continue

            # Compute first-stage coefficient (for instrument strength)
            rho_hat = np.dot(Z, W) / np.dot(Z, Z)
            mu_hat = rho_hat * np.sqrt(np.dot(Z, Z))
            group_mu[group] = mu_hat

            # --- For variance estimation: run a “first-stage” regression of W on Z (with intercept) ---
            X_first = sm.add_constant(Z, has_constant='skip')
            res_first = sm.OLS(W, X_first).fit()
            v = res_first.resid
            W_hat = res_first.predict()

            # --- Second-stage regression: regress Y on predicted W and controls (X_const) ---
            # Note: The design matrix has W_hat as the first column (coefficient of interest)
            # and X_const (which already includes a constant) as additional controls.
            X_second = np.column_stack((W_hat, X_const))
            res_second = sm.OLS(Y, X_second).fit()
            u = res_second.resid

            # Collect residuals (all observations across groups)
            u_resids.extend(u)
            v_resids.extend(v)

        return group_mu, u_resids, v_resids

    # Process each split separately.
    group_mu_a, u_a, v_a = process_split(data_a)
    group_mu_b, u_b, v_b = process_split(data_b)

    # Compute variance estimates from residuals in each split.
    sigma_u_sq_a, sigma_v_sq_a, sigma_uv_a = compute_variance_estimates(u_a, v_a)
    sigma_u_sq_b, sigma_v_sq_b, sigma_uv_b = compute_variance_estimates(u_b, v_b)

    # Prepare for adaptive threshold calculation.
    mu_hat_a = list(group_mu_a.values())
    mu_hat_b = list(group_mu_b.values())
    N_a = len(data_a)
    N_b = len(data_b)

    # Compute delta_hat for each split using the mu_hat from the other split.
    delta_hat_a = compute_adaptive_delta(mu_hat_b, sigma_u_sq_a, sigma_v_sq_a, sigma_uv_a, N_a, kappa)
    delta_hat_b = compute_adaptive_delta(mu_hat_a, sigma_u_sq_b, sigma_v_sq_b, sigma_uv_b, N_b, kappa)

    # Compute split-sample estimates.
    beta_a = select_and_interact_from_splits(data_a, data_b, groups_col, y_col, w_col, z_col, x_cols, delta_hat_a, kappa)
    beta_b = select_and_interact_from_splits(data_b, data_a, groups_col, y_col, w_col, z_col, x_cols, delta_hat_b, kappa)

    #  Average
    return (beta_a + beta_b) / 2




"""
Function: `select_and_interact_from_splits`

This function computes the IV estimator using data from two splits.
	•	Residualization and Group Stats:
For each group in the provided split, it first residualizes the instrument Z̃ so that the resulting
instrument Z is orthogonal to X. Then, it computes the group-level first-stage
coefficient ρ̂₉ by regressing W on Z and calculates the statistic μ̂₉ = ρ̂₉ √(Z′Z).
This is scaled by the tuning parameter κ to yield μ̃₉.

	•	Selection Rule:
The function then uses the “selection split” (which has its own computed group statistics)
to decide which groups are strong. Only those groups with μ̃₉ ≥ δ are used in the second-stage estimation.

	•	Second-Stage Estimation:
For each selected group, it computes the inner products Z′Y and Z′W from the estimation split.
These are then weighted by the first-stage coefficient from the selection split.
Finally, the estimator β̂ is formed by taking the ratio of the weighted sums.

"""
def select_and_interact_from_splits(estimation_data, selection_data, groups_col, y_col, w_col, z_col, x_cols, delta, kappa):
    """
    Split-sample select-and-interact estimator β.
    One split (selection_data) is used to compute the instrument strength
    (rho_hat and tilde_mu) for each group, and the other split (estimation_data) is used to compute
    the second-stage IV components.

    For each group g copmute:
       - Z_g: the residualized instrument (Z_tilde regressed on X)
       - rho_hat_g: estimated slope from regressing W on Z (using only the estimation split)
       - mu_hat_g: = rho_hat_g * sqrt(Z_g'Z_g)
       - tilde_mu_g: = mu_hat_g / sqrt(kappa)

    Then, for groups with tilde_mu (from the selection split) >= delta,
    we form:
       β̂ = [∑_g ρ̂_g (Z_g'W_g)]⁻¹ ∑_g ρ̂_g (Z_g'Y_g)
    where the weight ρ̂_g is taken from the selection split.

    Parameters
    ----------
    estimation_data : pandas.DataFrame
        Data used for computing the second-stage (estimation) IV regression.
    selection_data : pandas.DataFrame
        Data used for computing instrument strength statistics.
    groups_col : str
        Name of the column that indicates group membership.
    y_col : str
        Name of the outcome variable column.
    w_col : str
        Name of the endogenous regressor column.
    z_col : str
        Name of the instrument column.
    x_cols : list of str
        List of exogenous regressor column names.
    delta : float
        The threshold value (δ) used for group selection.
    kappa : float
        The tuning parameter κ used to form tilde_mu.

    Returns
    -------
    float
        The computed split-sample select-and-interact estimator for β.
    """
    # Helper function: compute group-level statistics
    def compute_group_stats(data):
        group_stats = {}
        # Group data by groups_col
        for group, df in data.groupby(groups_col):
            X = df[x_cols].values
            Z_tilde = df[z_col].values  # original instrument
            W = df[w_col].values
            Y = df[y_col].values
            n = len(df)
            d = X.shape[1] if X.ndim > 1 else 1

            # Require sufficient degrees of freedom (n > d+1)
            if n <= d + 1:
                continue

            # --- Residualize the instrument: regress Z_tilde on X (with intercept) ---
            X_const = sm.add_constant(X, has_constant='add')
            model_Z = sm.OLS(Z_tilde, X_const)
            res_Z = model_Z.fit()
            Z = res_Z.resid  # This Z is orthogonal to X

            # If no variation in Z, skip this group
            if np.allclose(np.dot(Z, Z), 0):
                continue

            # --- Compute the first-stage coefficient for this group ---
            rho_hat = np.dot(Z, W) / np.dot(Z, Z)
            mu_hat = rho_hat * np.sqrt(np.dot(Z, Z))
            tilde_mu = mu_hat / np.sqrt(kappa)

            group_stats[group] = {
                'Z': Z,
                'W': W,
                'Y': Y,
                'rho_hat': rho_hat,
                'tilde_mu': tilde_mu
            }
        return group_stats

    # Compute statistics on the estimation and selection splits
    stats_est = compute_group_stats(estimation_data)
    stats_sel = compute_group_stats(selection_data)

    # Use the selection split to decide which groups have strong first-stage signals.
    numerator, denominator = 0.0, 0.0
    for group in stats_est:
        if group in stats_sel and stats_sel[group]['tilde_mu'] >= delta:
            # Use stats from the estimation split for the second-stage IV components,
            # but use the selection split’s rho_hat as the weight.
            Z_est = stats_est[group]['Z']
            W_est = stats_est[group]['W']
            Y_est = stats_est[group]['Y']
            rho_hat_sel = stats_sel[group]['rho_hat']

            numerator += rho_hat_sel * np.dot(Z_est, Y_est)
            denominator += rho_hat_sel * np.dot(Z_est, W_est)

    if np.isclose(denominator, 0):
        raise ValueError("No valid groups selected. Adjust delta or check data.")

    return numerator / denominator


"""Function: `compute_adaptive_delta`

This function computes the adaptive threshold δ̂ for selecting strong instruments.
	•	Scaling and Sorting:
It first scales all group-level μ values by √κ to form the tilde values and sorts them in descending order.
	•	Cost Function R(K):
The function then loops over possible numbers of selected groups K and computes a
cost function R(K), which is a combination of two terms: one that depends on the sum of squared
tilde values for the groups that are not selected, and a second that is proportional to the number of
selected groups (weighted by variance estimates).
	•	Threshold Determination:
The optimal K is the one that minimizes R(K), and the corresponding μ̃ at that order statistic
is returned as the adaptive threshold δ̂.
  """
def compute_adaptive_delta(mu_hat_values, sigma_u_sq, sigma_v_sq, sigma_uv, N, kappa):
    """
    Compute the adaptive threshold delta_hat based on minimizing the criterion:

       R(K) = (sigma_u^2/N)*sum_{g=K+1}^{G} (tilde_mu_{(g)})^2 + 2*(sigma_u^2*sigma_v^2 + sigma_uv^2)*(K/N),

    where tilde_mu_{(g)} = mu_hat_{(g)}/sqrt(kappa) and the mu_hat values are sorted
    in descending order.

    Parameters
    ----------
    mu_hat_values : list or array of float
        The group-level mu_hat values (without the kappa scaling) from one split.
    sigma_u_sq : float
        Estimate of the variance of u (second-stage residuals).
    sigma_v_sq : float
        Estimate of the variance of v (first-stage residuals).
    sigma_uv : float
        Estimate of the covariance between u and v.
    N : int
        Sample size of the split used for variance estimation.
    kappa : float
        The tuning parameter κ.

    Returns
    -------
    float
        The adaptive threshold delta_hat.
    """
    # Compute tilde_mu values and sort in descending order
    tilde_mu_sorted = sorted([mu / np.sqrt(kappa) for mu in mu_hat_values], reverse=True)
    R_values = []
    G_eff = len(tilde_mu_sorted)  # effective number of groups in this split

    # Loop over K = 0,1,...,G_eff (K indicates how many groups are "selected")
    for K in range(G_eff + 1):
        # Sum the squares of tilde_mu for groups not selected (i.e., from K to end)
        sum_tilde_mu_sq = sum(mu_val**2 for mu_val in tilde_mu_sorted[K:])
        term1 = (sigma_u_sq / N) * sum_tilde_mu_sq
        term2 = 2 * (sigma_u_sq * sigma_v_sq + sigma_uv**2) * (K / N)
        R = term1 + term2
        R_values.append(R)

    # Find K that minimizes R(K)
    K_hat = int(np.argmin(R_values))
    # Set delta_hat as the K_hat-th order statistic if available; otherwise 0.
    delta_hat = tilde_mu_sorted[K_hat] if K_hat < G_eff else 0.0 # TODO: Maybe 0 is not the most appropriate value
    return delta_hat

def compute_variance_estimates(u_residuals, v_residuals):
    """
    Compute the sample variances sigma_u^2 and sigma_v^2 and the covariance sigma_uv.

    Parameters
    ----------
    u_residuals : list or array of float
        Collected second-stage residuals.
    v_residuals : list or array of float
        Collected first-stage residuals.

    Returns
    -------
    tuple of (sigma_u_sq, sigma_v_sq, sigma_uv)
    """
    u = np.array(u_residuals)
    v = np.array(v_residuals)
    sigma_u_sq = np.var(u, ddof=1)
    sigma_v_sq = np.var(v, ddof=1)
    sigma_uv = np.cov(u, v, ddof=1)[0, 1]
    return sigma_u_sq, sigma_v_sq, sigma_uv





# %%
# # %%
def gen_data(seed=None):
    np.random.seed(seed)

    # Sample size
    n = 10_000

    # Create n groups
    n_g = 40
    group = np.random.choice(range(n_g), n)

    # Generate instrument
    Z = np.random.normal(0, 1, n)

    # Create endogenous variable D with group-dependent relationships
    D = np.zeros(n)
    U = np.random.normal(0, 1, n)  # Unobserved confounder

    # Randomly assign group-dependent relationships. Halft with strong correlation, half with weak correlation
    for i in range(n):
        if group[i] % 2 == 0:
            if group[i] % 4 == 0:
                D[i] = Z[i] + 0.3*U[i] + np.random.normal(0, 0.5)
            else:
                D[i] = 0.2*Z[i] + 0.3*U[i] + np.random.normal(0, 0.5)
        else:
            D[i] = np.random.normal()*Z[i] + 0.3*U[i] + np.random.normal(0, 0.5)

    # Generate outcome variable Y with group heterogeneity on D
    treatment_effects = np.where(group % 2 == 0, 0.8, 0.4)
    Y = np.multiply(treatment_effects, D) + 1.5*U + np.random.normal(0, 1, n)

    # Create DataFrame
    df = pd.DataFrame({'Y': Y, 'D': D, 'Z': Z, 'group': group})

    # Add observed covariates
    # TODO: Change DGP so that X1 and X2 are relevant for IV estimation
    df['X1'] = np.random.normal(0, 1, n)
    df['X2'] = np.random.binomial(1, 0.5, n)

    return df

df = gen_data()
beta_aiv = adaptive_split_sample_estimator(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'])
beta_aiv
