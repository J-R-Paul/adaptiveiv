# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def split_sample_select_and_interact(data, groups_col, y_col, w_col, z_col, x_cols, delta, kappa, random_state=42):
    """
    Compute the split-sample select-and-interact estimator β_ssel,int(δ) with tilde_mu adjustment.
    """
    data_a, data_b = train_test_split(data, test_size=0.5, random_state=random_state)

    def compute_group_stats(split_data):
        group_stats = {}
        for group, df in split_data.groupby(groups_col):
            X = df[x_cols].values
            Z_tilde = df[z_col].values
            W = df[w_col].values
            Y = df[y_col].values
            n = len(df)
            d = X.shape[1]

            if n <= d + 1:
                continue

            # Residualize Z_tilde on X
            X_const = sm.add_constant(X, has_constant='add')
            model_Z = sm.OLS(Z_tilde, X_const)
            res_Z = model_Z.fit()
            Z = res_Z.resid

            z_z = np.dot(Z, Z)
            if z_z == 0:
                continue
            rho_hat = np.dot(Z, W) / z_z
            mu_hat = rho_hat * np.sqrt(z_z)
            tilde_mu = mu_hat / np.sqrt(kappa)  # Compute tilde_mu

            group_stats[group] = {
                'Z': Z, 'W': W, 'Y': Y,
                'rho_hat': rho_hat,
                'tilde_mu': tilde_mu  # Store tilde_mu instead of mu_hat
            }
        return group_stats

    stats_a = compute_group_stats(data_a)
    stats_b = compute_group_stats(data_b)

    # Calculate β^a(δ) using split-a data and split-b selection
    numerator_a, denominator_a = 0.0, 0.0
    for group in stats_a:
        if group in stats_b and stats_b[group]['tilde_mu'] >= delta:
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
        if group in stats_a and stats_a[group]['tilde_mu'] >= delta:
            Z_b = stats_b[group]['Z']
            W_b = stats_b[group]['W']
            Y_b = stats_b[group]['Y']
            rho_hat_a = stats_a[group]['rho_hat']

            zw = np.dot(Z_b, W_b)
            zy = np.dot(Z_b, Y_b)
            denominator_b += rho_hat_a * zw
            numerator_b += rho_hat_a * zy

    beta_a = numerator_a / denominator_a if denominator_a != 0 else np.nan
    beta_b = numerator_b / denominator_b if denominator_b != 0 else np.nan

    valid_betas = [b for b in [beta_a, beta_b] if not np.isnan(b)]
    if not valid_betas:
        raise ValueError("No valid groups selected. Adjust delta or check data.")

    return np.mean(valid_betas)

def compute_adaptive_delta(mu_hat_values, sigma_u_sq, sigma_v_sq, sigma_uv, N, G, kappa):
    """Compute δ_hat using the adaptive selection procedure with corrected R(K)."""
    tilde_mu_sorted = sorted([mu / np.sqrt(kappa) for mu in mu_hat_values], reverse=True)
    R_values = []

    for K in range(len(tilde_mu_sorted) + 1):
        sum_tilde_mu_sq = sum([mu**2 for mu in tilde_mu_sorted[K:]])
        term1 = (sigma_u_sq / N) * sum_tilde_mu_sq
        term2 = 2 * (sigma_u_sq * sigma_v_sq + sigma_uv**2) * (K / N)
        R = term1 + term2
        R_values.append(R)

    K_hat = np.argmin(R_values)
    delta_hat = tilde_mu_sorted[K_hat] if K_hat < len(tilde_mu_sorted) else 0.0
    return delta_hat

def adaptive_split_sample_estimator(data, groups_col, y_col, w_col, z_col, x_cols, random_state=42):
    # Split data into two halves
    data_a, data_b = train_test_split(data, test_size=0.5, random_state=random_state)

    # Compute kappa based on the number of groups in the full data
    G = data[groups_col].nunique()
    kappa = (np.log(G)) ** 2

    # Function to compute group mu_hat and residuals for variance estimation
    def process_split(split_data):
        group_mu = {}
        u_resids = []
        v_resids = []
        for group, df in split_data.groupby(groups_col):
            X = df[x_cols].values
            Z_tilde = df[z_col].values
            W = df[w_col].values
            Y = df[y_col].values
            n = len(df)
            d = X.shape[1]

            if n <= d + 1:
                continue

            X_const = sm.add_constant(X, has_constant='add')
            # Residualize Z_tilde
            model_Z = sm.OLS(Z_tilde, X_const)
            Z = model_Z.fit().resid
            # First-stage regression
            X_first = np.column_stack((Z, X_const))
            res_first = sm.OLS(W, X_first).fit()
            v = res_first.resid
            # Second-stage regression
            W_hat = res_first.predict()
            X_second = np.column_stack((W_hat, X_const))
            res_second = sm.OLS(Y, X_second).fit()
            u = res_second.resid

            z_z = np.dot(Z, Z)
            if z_z == 0:
                continue
            rho_hat = np.dot(Z, W) / z_z
            mu_hat = rho_hat * np.sqrt(z_z)

            group_mu[group] = mu_hat
            u_resids.extend(u)
            v_resids.extend(v)

        return group_mu, u_resids, v_resids

    # Process splits for variance estimation and delta calculation
    group_mu_a, u_a, v_a = process_split(data_a)
    group_mu_b, u_b, v_b = process_split(data_b)

    # Estimate variance components using split residuals
    sigma_u_sq_a, sigma_v_sq_a, sigma_uv_a = compute_variance_estimates(u_a, v_a)
    sigma_u_sq_b, sigma_v_sq_b, sigma_uv_b = compute_variance_estimates(u_b, v_b)

    # Compute adaptive delta for each split using the other split's mu_hat
    mu_hat_a = list(group_mu_a.values())
    mu_hat_b = list(group_mu_b.values())
    G_a = len(mu_hat_a)
    G_b = len(mu_hat_b)

    # Use split b's data to compute delta for split a and vice versa
    delta_hat_a = compute_adaptive_delta(
        mu_hat_b, sigma_u_sq_b, sigma_v_sq_b, sigma_uv_b, len(data_b), G, kappa
    )
    delta_hat_b = compute_adaptive_delta(
        mu_hat_a, sigma_u_sq_a, sigma_v_sq_a, sigma_uv_a, len(data_a), G, kappa
    )

    # Compute split estimates with respective deltas
    beta_a = split_sample_select_and_interact(
        data_a, groups_col, y_col, w_col, z_col, x_cols, delta_hat_a, kappa, random_state
    )
    beta_b = split_sample_select_and_interact(
        data_b, groups_col, y_col, w_col, z_col, x_cols, delta_hat_b, kappa, random_state
    )

    return (beta_a + beta_b) / 2

# Helper functions remain mostly the same but ensure they are correctly referenced
def compute_variance_estimates(u_residuals, v_residuals):
    u = np.array(u_residuals)
    v = np.array(v_residuals)
    sigma_u_sq = np.var(u, ddof=1)
    sigma_v_sq = np.var(v, ddof=1)
    sigma_uv = np.cov(u, v, ddof=1)[0, 1]
    return sigma_u_sq, sigma_v_sq, sigma_uv
# %%
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

df = gen_data()
beta_aiv = adaptive_split_sample_estimator(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'])
beta_aiv
# %%
betas = []
deltas = []
for _ in range(1000):
    df = gen_data(seed=None)
    beta_adapt = adaptive_split_sample_estimator(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'])
    betas.append(beta_adapt)

# %%
plt.hist(betas, bins=30)
plt.xlabel("Estimator (β̂)")
plt.axvline(float(np.mean(betas)), color='red', linestyle='--')
plt.axvline(0.8, color='green', linestyle='--')
