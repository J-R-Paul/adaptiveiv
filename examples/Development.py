# %%
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adaptiveiv import AdaptiveIV

from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from scipy.stats import t

def gen_data(seed=42):
    np.random.seed(seed)  # For reproducibility

    # Sample size
    n = 1000

    # Create three distinct groups
    group = np.random.choice([0, 1, 2], size=n, p=[1/3, 1/3, 1/3])

    # Generate instrument Z (exogenous)
    Z = np.random.normal(0, 1, n)

    # Create endogenous variable D with group-dependent relationships
    D = np.zeros(n)
    U = np.random.normal(0, 1, n)  # Unobserved confounder

    # Group 0: Strong correlation with Z (high relevance)
    # Group 1: Weak correlation with Z (low relevance)
    # Group 2: No correlation with Z (exclusion restriction)
    for i in range(n):
        if group[i] == 0:
            D[i] = 0.3*Z[i] + 0.3*U[i] + np.random.normal(0, 0.5)
        elif group[i] == 1:
            D[i] = 0.03*Z[i] + 0.3*U[i] + np.random.normal(0, 0.5)
        else:
            D[i] = 0.0*Z[i] + 0.3*U[i] + np.random.normal(0, 0.5)

    # Generate outcome variable Y
    Y = 0.8*D + 1.5*U + np.random.normal(0, 1, n)

    # Create DataFrame
    df = pd.DataFrame({'Y': Y, 'D': D, 'Z': Z, 'group': group})

    # Add observed covariates (optional)
    df['X1'] = np.random.normal(0, 1, n)
    df['X2'] = np.random.binomial(1, 0.5, n)

    return df


def selective_iv_estimator(data, groups_col, y_col, w_col, z_col, x_cols, alpha_fs=0.05):
    """
    Calculate the selective IV estimator using groups with significant first-stage instrument strength.

    Parameters:
    - data: pandas DataFrame containing the data.
    - groups_col: Column name for the group identifiers.
    - y_col: Column name for the outcome variable Y.
    - w_col: Column name for the endogenous variable W.
    - z_col: Column name for the instrument Z_tilde.
    - x_cols: List of column names for the covariates X.
    - alpha_fs: Significance level for the first-stage t-test (default: 0.05).

    Returns:
    - beta_selp: The computed selective IV estimator.
    """
    numerator = 0.0
    denominator = 0.0

    for group_name, group_data in data.groupby(groups_col):
        X_g = group_data[x_cols].values
        Z_tilde_g = group_data[z_col].values
        W_g = group_data[w_col].values
        Y_g = group_data[y_col].values
        n_g = len(group_data)
        d = X_g.shape[1]

        if n_g <= d + 1:  # Ensure enough observations for degrees of freedom
            continue

        # Residualize Z_tilde on X_g
        model_Z = sm.OLS(Z_tilde_g, sm.add_constant(X_g, has_constant='skip'))  # Ensure constant
        res_Z = model_Z.fit()
        Z_g = res_Z.resid

        # First stage regression of W on Z_g and X_g (with constant)
        X_first = np.column_stack((Z_g, sm.add_constant(X_g, has_constant='skip')))
        model_first = sm.OLS(W_g, X_first)
        res_first = model_first.fit()

        # Extract t-statistic for Z_g (first coefficient)
        rho_g = res_first.params[0]
        se_rho_g = res_first.bse[0]
        t_g = rho_g / se_rho_g

        df = n_g - X_first.shape[1]  # Degrees of freedom
        if df <= 0:
            continue

        c_alpha = t.ppf(1 - alpha_fs, df)

        if t_g > c_alpha:
            # print("group", group_name, "selected")
            zw = np.dot(Z_g, W_g)
            zy = np.dot(Z_g, Y_g)
            denominator += zw
            numerator += zy

    if denominator == 0:
        raise ValueError("No groups selected; adjust significance level or check instrument strength.")

    beta_selp = numerator / denominator
    return beta_selp

def adaptive_select_and_interact(data, groups_col, y_col, w_col, z_col, x_cols, delta):
    """
    Compute the adaptive select-and-interact estimator β_sel,int(δ).

    Parameters:
    - data: pandas DataFrame containing the data.
    - groups_col: Column name for group identifiers.
    - y_col: Column name for the outcome variable Y.
    - w_col: Column name for the endogenous variable W.
    - z_col: Column name for the raw instrument Z_tilde.
    - x_cols: List of column names for covariates X.
    - delta: Threshold for μ_hat_g to select groups.

    Returns:
    - beta_sel_int: Estimated coefficient β_sel,int(δ).
    """
    numerator = 0.0
    denominator = 0.0

    for group_name, group_data in data.groupby(groups_col):
        X_g = group_data[x_cols].values
        Z_tilde_g = group_data[z_col].values
        W_g = group_data[w_col].values
        Y_g = group_data[y_col].values
        n_g = len(group_data)
        d = X_g.shape[1]

        # Skip small groups (insufficient degrees of freedom)
        if n_g <= d + 1:
            continue

        # Residualize Z_tilde on X_g (with constant)
        X_with_const = sm.add_constant(X_g, has_constant='add')
        model_Z = sm.OLS(Z_tilde_g, X_with_const)
        res_Z = model_Z.fit()
        Z_g = res_Z.resid

        # Compute instrument strength metrics
        z_z = np.dot(Z_g, Z_g)
        if z_z == 0:  # Avoid division by zero
            continue
        z_w = np.dot(Z_g, W_g)
        rho_hat_g = z_w / z_z
        mu_hat_g = rho_hat_g * np.sqrt(z_z)

        # Check selection criterion
        if mu_hat_g > delta:
            weight = rho_hat_g ** 2
            numerator += weight * np.dot(Z_g, Y_g)
            denominator += weight * z_w

    if denominator == 0:
        raise ValueError("No groups selected. Adjust delta or check instrument strength.")

    beta_sel_int = numerator / denominator
    return beta_sel_int


# %%
betas = []
for _ in range(100):
    df = gen_data(seed = None)
    beta_selp = selective_iv_estimator(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'])
    betas.append(beta_selp)

print(np.mean(betas))
print("true beta: 0.8")
# %%
betas = []
for delta in range(0, 1):
    betas_delta = []
    for _ in range(100):
        df = gen_data(seed = None)
        try:
            beta_sel_int = adaptive_select_and_interact(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'], delta)
        except ValueError:
            beta_sel_int = np.nan

        betas_delta.append(beta_sel_int)

    betas.append(np.mean(betas_delta))
# %%
# Plot
plt.plot(range(0, 30), betas)
plt.xlabel("Threshold (δ)")
plt.ylabel("Estimator (β̂)")
# %%
def split_sample_select_and_interact(data, groups_col, y_col, w_col, z_col, x_cols, delta, random_state=42):
    """
    Compute the split-sample select-and-interact estimator β_ssel,int(δ).

    Parameters:
    - data: pandas DataFrame containing the data.
    - groups_col: Column name for the group identifiers.
    - y_col: Column name for the outcome variable Y.
    - w_col: Column name for the endogenous variable W.
    - z_col: Column name for the raw instrument Z_tilde.
    - x_cols: List of column names for the covariates X.
    - delta: Threshold for μ_hat_g to select groups.
    - random_state: Seed for reproducible data splitting (default: 42).

    Returns:
    - beta_ssel_int: Estimated coefficient β_ssel,int(δ).
    """
    # Split the data into two halves
    data_a, data_b = train_test_split(
        data, test_size=0.5, shuffle=True, random_state=random_state
    )

    # Process each split to compute group-level statistics
    def compute_group_stats(split_data):
        group_stats = {}
        for group_name, group_df in split_data.groupby(groups_col):
            X = group_df[x_cols].values
            Z_tilde = group_df[z_col].values
            W = group_df[w_col].values
            Y = group_df[y_col].values
            n = len(group_df)
            d = X.shape[1]

            if n <= d + 1:  # Insufficient observations for regression
                continue

            # Residualize Z_tilde on X (with constant)
            X_const = sm.add_constant(X, has_constant='add')
            model_Z = sm.OLS(Z_tilde, X_const)
            res_Z = model_Z.fit()
            Z = res_Z.resid

            # Compute instrument strength metrics
            z_z = np.dot(Z, Z)
            if z_z == 0:
                continue  # Avoid division by zero
            z_w = np.dot(Z, W)
            rho_hat = z_w / z_z
            mu_hat = rho_hat * np.sqrt(z_z)

            group_stats[group_name] = {
                'Z': Z,
                'W': W,
                'Y': Y,
                'rho_hat': rho_hat,
                'mu_hat': mu_hat
            }
        return group_stats

    # Compute statistics for both splits
    stats_a = compute_group_stats(data_a)
    stats_b = compute_group_stats(data_b)

    # Calculate β^a(δ) using split-a data and split-b selection
    numerator_a, denominator_a = 0.0, 0.0
    for group in stats_a:
        if group in stats_b and stats_b[group]['mu_hat'] >= delta:
            Z_a = stats_a[group]['Z']
            W_a = stats_a[group]['W']
            Y_a = stats_a[group]['Y']
            rho_hat_b = stats_b[group]['rho_hat']

            zw = np.dot(Z_a, W_a)
            zy = np.dot(Z_a, Y_a)
            denominator_a += rho_hat_b * zw
            numerator_a += rho_hat_b * zy

    # Calculate β^b(δ) using split-b data and split-a selection
    numerator_b, denominator_b = 0.0, 0.0
    for group in stats_b:
        if group in stats_a and stats_a[group]['mu_hat'] >= delta:
            Z_b = stats_b[group]['Z']
            W_b = stats_b[group]['W']
            Y_b = stats_b[group]['Y']
            rho_hat_a = stats_a[group]['rho_hat']

            zw = np.dot(Z_b, W_b)
            zy = np.dot(Z_b, Y_b)
            denominator_b += rho_hat_a * zw
            numerator_b += rho_hat_a * zy

    # Compute split estimates
    beta_a = numerator_a / denominator_a if denominator_a != 0 else np.nan
    beta_b = numerator_b / denominator_b if denominator_b != 0 else np.nan

    # Average valid estimates
    valid_betas = [b for b in [beta_a, beta_b] if not np.isnan(b)]
    if not valid_betas:
        raise ValueError("No valid groups selected in either split. Adjust delta or check data.")

    beta_ssel_int = np.mean(valid_betas)
    return beta_ssel_int


# %%
betas = []
for _ in range(4000):
    df = gen_data(seed = None)
    beta_ssel_int = split_sample_select_and_interact(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'], delta=1)
    betas.append(beta_ssel_int)

plt.hist(betas, bins=20)
# Add vlines at mean and true beta
plt.axvline(float(np.mean(betas)), color='red', linestyle='--')
plt.axvline(0.8, color='green', linestyle='--')

# %%

def compute_group_mu_and_residuals(data, groups_col, y_col, w_col, z_col, x_cols):
    """Compute μ_hat and residuals for all groups in the full sample."""
    group_mu = {}
    u_residuals = []
    v_residuals = []

    for group_name, group_df in data.groupby(groups_col):
        X = group_df[x_cols].values
        Z_tilde = group_df[z_col].values
        W = group_df[w_col].values
        Y = group_df[y_col].values
        n = len(group_df)
        d = X.shape[1]

        if n <= d + 1:
            continue  # Skip small groups

        # Residualize Z_tilde on X (with constant)
        X_const = sm.add_constant(X, has_constant='add')
        model_Z = sm.OLS(Z_tilde, X_const)
        res_Z = model_Z.fit()
        Z = res_Z.resid

        # First-stage regression (W on Z and X)
        X_first = np.column_stack((Z, X_const))
        model_first = sm.OLS(W, X_first)
        res_first = model_first.fit()
        v = res_first.resid

        # Second-stage regression (Y on W_hat and X using 2SLS)
        W_hat = res_first.predict()
        X_second = np.column_stack((W_hat, X_const))
        model_second = sm.OLS(Y, X_second)
        res_second = model_second.fit()
        u = res_second.resid

        # Compute μ_hat for the group
        z_z = np.dot(Z, Z)
        if z_z == 0:
            continue
        rho_hat = np.dot(Z, W) / z_z
        mu_hat = rho_hat * np.sqrt(z_z)

        group_mu[group_name] = mu_hat
        u_residuals.extend(u)
        v_residuals.extend(v)

    return group_mu, u_residuals, v_residuals

def compute_variance_estimates(u_residuals, v_residuals):
    """Estimate σ_u², σ_v², and σ_uv using pooled residuals."""
    u = np.array(u_residuals)
    v = np.array(v_residuals)
    N = len(u)

    if N == 0:
        raise ValueError("No residuals available for variance estimation.")

    # Sample variances and covariance
    sigma_u_sq = np.var(u, ddof=1)
    sigma_v_sq = np.var(v, ddof=1)
    sigma_uv = np.cov(u, v, ddof=1)[0, 1]

    return sigma_u_sq, sigma_v_sq, sigma_uv

def compute_adaptive_delta(group_mu, sigma_u_sq, sigma_v_sq, sigma_uv, N):
    """Compute δ_hat using the adaptive selection procedure."""
    mu_values = list(group_mu.values())
    mu_sorted = sorted(mu_values, reverse=True)
    G = len(mu_sorted)

    if G == 0:
        raise ValueError("No groups available for delta computation.")

    kappa = (np.log(G)) ** 2  # Rule-of-thumb tuning parameter
    R_values = []

    for K in range(G + 1):
        sum_mu_sq = sum([mu**2 for mu in mu_sorted[K:]]) if K < G else 0
        term1 = (sigma_u_sq / N) * sum_mu_sq
        term2 = 2 * (sigma_u_sq * sigma_v_sq + sigma_uv**2) * (K / N)
        R = term1 + term2
        R_values.append(R)

    K_hat = np.argmin(R_values)
    delta_hat = mu_sorted[K_hat] / np.sqrt(kappa) if K_hat < G else 0.0
    return delta_hat

def adaptive_split_sample_estimator(data, groups_col, y_col, w_col, z_col, x_cols, random_state=42,
    return_delta=False):
    """
    Compute the adaptive split-sample select-and-interact estimator β_adpt.

    Parameters:
    - data: DataFrame containing all variables.
    - groups_col: Column name for group identifiers.
    - y_col, w_col, z_col: Outcome, endogenous, and instrument columns.
    - x_cols: List of exogenous covariates.
    - random_state: Seed for reproducibility.

    Returns:
    - β_adpt: Adaptive estimator using data-driven δ.
    """
    # Step 1: Compute group μ_hat and residuals
    group_mu, u_resids, v_resids = compute_group_mu_and_residuals(
        data, groups_col, y_col, w_col, z_col, x_cols
    )

    # Step 2: Estimate variance components
    sigma_u_sq, sigma_v_sq, sigma_uv = compute_variance_estimates(u_resids, v_resids)

    # Step 3: Compute adaptive δ
    N = len(data)
    delta_hat = compute_adaptive_delta(group_mu, sigma_u_sq, sigma_v_sq, sigma_uv, N)

    # Step 4: Split-sample estimator with δ_hat
    def split_sample_estimator(data, delta):
        # Reuse split_sample_select_and_interact from previous implementation
        # Note: Ensure this function is defined as in the previous answer
        return split_sample_select_and_interact(
            data, groups_col, y_col, w_col, z_col, x_cols, delta, random_state
        )

    if return_delta:
        return split_sample_estimator(data, delta_hat), delta_hat
    else:
        return split_sample_estimator(data, delta_hat)

# Example usage:
df = gen_data(seed=None)
beta_adapt, delta = adaptive_split_sample_estimator(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'], return_delta=True)

print(beta_adapt, delta)
# %%
betas = []
deltas = []
for _ in range(1000):
    df = gen_data(seed=None)
    beta_adapt, delta = adaptive_split_sample_estimator(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'], return_delta=True)
    betas.append(beta_adapt)
    deltas.append(delta)

# %%
plt.hist(betas, bins=20)
plt.xlabel("Estimator (β̂)")
plt.axvline(float(np.mean(betas)), color='red', linestyle='--')
plt.axvline(0.8, color='green', linestyle='--')
plt.hist(deltas, bins=20, alpha=0.2)
