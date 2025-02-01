# %%
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

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
                continue  # Skip groups with insufficient observations

            # Residualize Z_tilde on X
            X_const = sm.add_constant(X, has_constant='add')
            model_Z = sm.OLS(Z_tilde, X_const)
            res_Z = model_Z.fit()
            Z = res_Z.resid

            z_z = np.dot(Z, Z)
            if z_z == 0:
                continue  # Skip groups with zero residualized instrument variance

            rho_hat = np.dot(Z, W) / z_z
            mu_hat = rho_hat * np.sqrt(z_z)
            tilde_mu = mu_hat / np.sqrt(kappa)  # Adjust mu_hat by kappa

            group_stats[group] = {
                'Z': Z, 'W': W, 'Y': Y,
                'rho_hat': rho_hat,
                'tilde_mu': tilde_mu
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

def compute_variance_estimates(u_residuals, v_residuals):
    """Helper function to compute variance and covariance estimates."""
    u = np.array(u_residuals)
    v = np.array(v_residuals)
    sigma_u_sq = np.var(u, ddof=1)
    sigma_v_sq = np.var(v, ddof=1)
    sigma_uv = np.cov(u, v, ddof=1)[0, 1]
    return sigma_u_sq, sigma_v_sq, sigma_uv

def adaptive_split_sample_estimator(data, groups_col, y_col, w_col, z_col, x_cols, random_state=42):
    # Split data into two halves
    data_a, data_b = train_test_split(data, test_size=0.5, random_state=random_state)

    # Compute kappa based on the number of groups in the full data
    G = data[groups_col].nunique()
    kappa = (np.log(G)) ** 2  # Rule-of-thumb from the problem statement

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
                continue  # Skip groups with insufficient observations

            X_const = sm.add_constant(X, has_constant='add')
            # Residualize Z_tilde on X
            model_Z = sm.OLS(Z_tilde, X_const)
            Z = model_Z.fit().resid

            # First-stage regression: W ~ Z + X
            X_first = np.column_stack((Z, X_const))
            res_first = sm.OLS(W, X_first).fit()
            v = res_first.resid

            # Second-stage regression: Y ~ W_hat + X
            W_hat = res_first.predict()
            X_second = np.column_stack((W_hat, X_const))
            res_second = sm.OLS(Y, X_second).fit()
            u = res_second.resid

            z_z = np.dot(Z, Z)
            if z_z == 0:
                continue  # Skip groups with zero residualized instrument variance

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
    N_a = len(data_a)
    N_b = len(data_b)

    delta_hat_a = compute_adaptive_delta(
        mu_hat_b, sigma_u_sq_a, sigma_v_sq_a, sigma_uv_a, N_b, G, kappa
    )
    delta_hat_b = compute_adaptive_delta(
        mu_hat_a, sigma_u_sq_b, sigma_v_sq_b, sigma_uv_b, N_a, G, kappa
    )

    # Compute split estimates with respective deltas
    beta_a = split_sample_select_and_interact(
        data_a, groups_col, y_col, w_col, z_col, x_cols, delta_hat_a, kappa, random_state
    )
    beta_b = split_sample_select_and_interact(
        data_b, groups_col, y_col, w_col, z_col, x_cols, delta_hat_b, kappa, random_state
    )

    return (beta_a + beta_b) / 2
# Bootstrap functions
# %%

def bootstrap_confidence_intervals(
    data,
    groups_col,
    y_col,
    w_col,
    z_col,
    x_cols,
    n_bootstrap=999,
    confidence_level=0.95,
    random_state=42,
    show_progress=True
):
    """
    Compute bootstrap confidence intervals for the adaptive split-sample estimator.

    Parameters:
    -----------
    data : pandas DataFrame
        The input dataset
    groups_col : str
        Name of the column containing group identifiers
    y_col : str
        Name of the outcome variable column
    w_col : str
        Name of the endogenous variable column
    z_col : str
        Name of the instrument column
    x_cols : list
        List of covariate column names
    n_bootstrap : int
        Number of bootstrap replications
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% confidence intervals)
    random_state : int
        Random seed for reproducibility
    show_progress : bool
        Whether to show progress bar during bootstrap

    Returns:
    --------
    dict
        Contains point estimate, standard error, and confidence intervals
    """
    np.random.seed(random_state)

    # Compute point estimate on full sample
    point_estimate = adaptive_split_sample_estimator(
        data, groups_col, y_col, w_col, z_col, x_cols, random_state
    )

    # Initialize storage for bootstrap estimates
    bootstrap_estimates = np.zeros(n_bootstrap)

    # Create progress bar if requested
    iterator = tqdm(range(n_bootstrap)) if show_progress else range(n_bootstrap)

    # Perform bootstrap replications
    for b in iterator:
        try:
            # Sample with replacement at the group level
            groups = data[groups_col].unique()
            bootstrap_groups = np.random.choice(groups, size=len(groups), replace=True)

            # Create bootstrap sample
            bootstrap_sample = pd.concat([
                data[data[groups_col] == group] for group in bootstrap_groups
            ]).reset_index(drop=True)

            # Compute estimate on bootstrap sample
            bootstrap_estimates[b] = adaptive_split_sample_estimator(
                bootstrap_sample, groups_col, y_col, w_col, z_col, x_cols,
                random_state=random_state + b  # Vary random seed across bootstraps
            )
        except Exception as e:
            print(f"Warning: Bootstrap iteration {b} failed with error: {str(e)}")
            bootstrap_estimates[b] = np.nan

    # Remove any failed bootstrap iterations
    bootstrap_estimates = bootstrap_estimates[~np.isnan(bootstrap_estimates)]

    if len(bootstrap_estimates) == 0:
        raise ValueError("All bootstrap iterations failed")

    # Compute standard error
    bootstrap_se = np.std(bootstrap_estimates, ddof=1)

    # Compute confidence intervals
    # Method 1: Percentile method
    alpha = 1 - confidence_level
    ci_lower_pct = np.percentile(bootstrap_estimates, 100 * (alpha/2))
    ci_upper_pct = np.percentile(bootstrap_estimates, 100 * (1 - alpha/2))

    # Method 2: Normal approximation
    z_score = stats.norm.ppf(1 - alpha/2)
    ci_lower_normal = point_estimate - z_score * bootstrap_se
    ci_upper_normal = point_estimate + z_score * bootstrap_se

    # Compute t-statistic and p-value
    t_stat = point_estimate / bootstrap_se
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))

    return {
        'point_estimate': point_estimate,
        'standard_error': bootstrap_se,
        'percentile_ci': (ci_lower_pct, ci_upper_pct),
        'normal_ci': (ci_lower_normal, ci_upper_normal),
        't_statistic': t_stat,
        'p_value': p_value,
        'n_successful_bootstraps': len(bootstrap_estimates),
        'bootstrap_estimates': bootstrap_estimates
    }

def print_bootstrap_results(results):
    """
    Print formatted results from bootstrap_confidence_intervals
    """
    print("\nAdaptive Split-Sample IV Results")
    print("=" * 50)
    print(f"Point Estimate: {results['point_estimate']:.4f}")
    print(f"Standard Error: {results['standard_error']:.4f}")
    print(f"t-statistic: {results['t_statistic']:.4f}")
    print(f"p-value: {results['p_value']:.4f}")
    print("\nConfidence Intervals:")
    print("-" * 50)
    print("Percentile Method:")
    print(f"  [{results['percentile_ci'][0]:.4f}, {results['percentile_ci'][1]:.4f}]")
    print("Normal Approximation:")
    print(f"  [{results['normal_ci'][0]:.4f}, {results['normal_ci'][1]:.4f}]")
    print(f"\nSuccessful Bootstrap Replications: {results['n_successful_bootstraps']}")

# %%
def gen_data(seed=42):
    np.random.seed(seed)  # For reproducibility

    # Sample size
    n = 10_000

    # Create n groups
    n_g = 40
    group = np.random.choice(range(n_g), n)

    # Generate instrument Z (exogenous)
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

# %%
#
bootstrap_results = bootstrap_confidence_intervals(
                                                    data=df,
                                                    groups_col='group',
                                                    y_col='Y',
                                                    w_col='D',
                                                    z_col='Z',
                                                    x_cols=['X1', 'X2'],
                                                    n_bootstrap=999,
                                                    confidence_level=0.95,
)
print_bootstrap_results(bootstrap_results)
# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_results['bootstrap_estimates'], bins=50, density=True)
plt.axvline(bootstrap_results['point_estimate'], color='r', linestyle='--', label='Point Estimate')
plt.axvline(bootstrap_results['percentile_ci'][0], color='g', linestyle=':', label='95% CI')
plt.axvline(bootstrap_results['percentile_ci'][1], color='g', linestyle=':')
plt.xlabel('Coefficient Estimate')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Adaptive Split-Sample IV Estimator')
plt.legend()
plt.show()

# %%
betas = []
deltas = []
for _ in range(1000):
    df = gen_data(seed=None)
    beta_adapt = adaptive_split_sample_estimator(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'])
    betas.append(beta_adapt)

# %%
import matplotlib.pyplot as plt
plt.hist(betas, bins=30)
plt.xlabel("Estimator (β̂)")
plt.axvline(float(np.mean(betas)), color='red', linestyle='--')
plt.axvline(0.8, color='green', linestyle='--')


# %%
from statsmodels.sandbox.regression.gmm import IV2SLS
bs = []
for i in range(1000):
    df = gen_data(seed=None)

    iv = IV2SLS(df['Y'], df[['D', 'X1', 'X2']], df[['Z', 'X1', 'X2']]).fit()
    bs.append(iv.params['D'])
# %%
# Plot the two histograms on the same plot
plt.hist(betas, bins=30, alpha=0.5, label='AdaptiveIV')
plt.hist(bs, alpha=0.5, bins=30)
plt.xlabel("Estimator (β̂)")
