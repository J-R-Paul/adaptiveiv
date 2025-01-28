# %%
import numpy as np
import pandas as pd
from adaptiveiv import AdaptiveIV

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
            D[i] = 0.5*Z[i] + 0.3*U[i] + np.random.normal(0, 0.5)
        elif group[i] == 1:
            D[i] = 0.2*Z[i] + 0.3*U[i] + np.random.normal(0, 0.5)
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

y = df['Y'].to_numpy()
d = df['D'].to_numpy()
z = df['Z'].to_numpy()
groups = df['group'].to_numpy()
x = df[["X1", "X2"]].to_numpy()


# %%
aiv = AdaptiveIV(y, d, z, groups, exog=x)

results = aiv.fit()


# %%

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
            zw = np.dot(Z_g, W_g)
            zy = np.dot(Z_g, Y_g)
            denominator += zw
            numerator += zy

    if denominator == 0:
        raise ValueError("No groups selected; adjust significance level or check instrument strength.")

    beta_selp = numerator / denominator
    return beta_selp

# Example usage:
beta_selp = selective_iv_estimator(df, 'group', 'Y', 'D', 'Z', ['X1', 'X2'])
print(f"Selective IV estimator: {beta_selp:.4f}")
