# %%
"""
adaptive_iv.py

A package implementing an adaptive split-sample select-and-interact IV estimator.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List
from sklearn.model_selection import train_test_split


def add_constant(X: np.ndarray) -> np.ndarray:
    """
    Adds a constant (a column of ones) to the design matrix X.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of predictors.

    Returns
    -------
    np.ndarray
        The design matrix with a constant as the first column.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack((np.ones(X.shape[0]), X))


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Residualizes y with respect to X using least squares.

    Parameters
    ----------
    y : np.ndarray
        Dependent variable.
    X : np.ndarray
        Independent variables.

    Returns
    -------
    np.ndarray
        Residuals from the regression as a 1D array.
    """
    X_const = add_constant(X)
    beta, _, _, _ = np.linalg.lstsq(X_const, y, rcond=None)
    return (y - X_const @ beta).ravel()


@dataclass
class GroupStats:
    """
    Data class holding group-level IV statistics.
    """
    Z: np.ndarray       # Residualized instrument (1D array)
    W: np.ndarray       # Endogenous regressor (1D array)
    Y: np.ndarray       # Outcome (1D array)
    rho_hat: float      # First-stage slope.
    tilde_mu: float     # Scaled instrument strength.
    mu_hat: float       # Raw instrument strength.


class GroupIVCalculator:
    """
    Computes group-level IV statistics.
    """
    def __init__(self, groups_col: str, y_col: str, w_col: str,
                 z_col: str, x_cols: List[str], kappa: float) -> None:
        self.groups_col = groups_col
        self.y_col = y_col
        self.w_col = w_col
        self.z_col = z_col
        self.x_cols = x_cols
        self.kappa = kappa

    def compute_stats(self, data: pd.DataFrame) -> Dict:
        """
        Computes group-level IV statistics for each group.

        Parameters
        ----------
        data : pd.DataFrame
            Data for one split.

        Returns
        -------
        Dict
            Mapping of group labels to GroupStats.
        """
        group_stats: Dict = {}
        for group, df in data.groupby(self.groups_col):
            X = df[self.x_cols].values
            W = df[self.w_col].values.ravel()
            Y = df[self.y_col].values.ravel()
            Z_tilde = df[self.z_col].values.ravel()

            n = len(df)
            d = X.shape[1] if X.ndim > 1 else 1
            if n <= d + 1:
                continue  # Not enough observations

            # Residualize the instrument.
            Z = residualize(Z_tilde, X)
            if np.allclose(np.dot(Z, Z), 0):
                continue

            # First-stage regression: compute slope.
            rho_hat = np.dot(Z, W) / np.dot(Z, Z)
            mu_hat = rho_hat * np.sqrt(np.dot(Z, Z))
            # Force mu_hat to be a float.
            mu_hat = float(mu_hat)
            tilde_mu = mu_hat / np.sqrt(self.kappa)

            group_stats[group] = GroupStats(Z=Z, W=W, Y=Y,
                                             rho_hat=rho_hat,
                                             tilde_mu=tilde_mu,
                                             mu_hat=mu_hat)
        return group_stats


class VarianceEstimator:
    """
    Computes variance and covariance estimates from regression residuals.
    """
    @staticmethod
    def compute(u_residuals: List[float],
                v_residuals: List[float]) -> Tuple[float, float, float]:
        """
        Computes sample variances and covariance.

        Parameters
        ----------
        u_residuals : List[float]
            Second-stage residuals.
        v_residuals : List[float]
            First-stage residuals.

        Returns
        -------
        Tuple[float, float, float]
            (sigma_u_sq, sigma_v_sq, sigma_uv)
        """
        u = np.array(u_residuals)
        v = np.array(v_residuals)
        sigma_u_sq = np.var(u, ddof=1)
        sigma_v_sq = np.var(v, ddof=1)
        sigma_uv = np.cov(u, v, ddof=1)[0, 1]
        return sigma_u_sq, sigma_v_sq, sigma_uv


class AdaptiveIVEstimator:
    """
    Implements the adaptive split-sample select-and-interact IV estimator.
    """
    def __init__(self, groups_col: str, y_col: str, w_col: str,
                 z_col: str, x_cols: List[str], random_state: int = 42) -> None:
        self.groups_col = groups_col
        self.y_col = y_col
        self.w_col = w_col
        self.z_col = z_col
        self.x_cols = x_cols
        self.random_state = random_state

    def compute_adaptive_delta(self, mu_hat_values: List[float],
                               sigma_u_sq: float, sigma_v_sq: float,
                               sigma_uv: float, N: int, kappa: float) -> float:
        """
        Computes the adaptive threshold delta_hat.

        Parameters
        ----------
        mu_hat_values : List[float]
            Raw instrument strength values (mu_hat) from the opposite split.
        sigma_u_sq : float
            Variance of second-stage residuals.
        sigma_v_sq : float
            Variance of first-stage residuals.
        sigma_uv : float
            Covariance between residuals.
        N : int
            Sample size of the split.
        kappa : float
            Tuning parameter.

        Returns
        -------
        float
            The adaptive threshold delta_hat.
        """
        # Scale and sort the mu_hat values.
        tilde_mu_sorted = sorted([mu / np.sqrt(kappa) for mu in mu_hat_values],
                                  reverse=True)
        R_values = []
        G_eff = len(tilde_mu_sorted)
        for K in range(G_eff + 1):
            sum_tilde_mu_sq = sum(mu_val ** 2 for mu_val in tilde_mu_sorted[K:])
            term1 = (sigma_u_sq / N) * sum_tilde_mu_sq
            term2 = 2 * (sigma_u_sq * sigma_v_sq + sigma_uv ** 2) * (K / N)
            R_values.append(term1 + term2)

        K_hat = int(np.argmin(R_values))
        delta_hat = tilde_mu_sorted[K_hat] if K_hat < G_eff else 0.0
        return delta_hat

    def select_and_interact(self, estimation_data: pd.DataFrame,
                            selection_data: pd.DataFrame,
                            delta: float, kappa: float) -> float:
        """
        Computes the split-sample select-and-interact estimator.

        Parameters
        ----------
        estimation_data : pd.DataFrame
            Data for the second-stage regression.
        selection_data : pd.DataFrame
            Data used for computing instrument strength.
        delta : float
            Adaptive threshold.
        kappa : float
            Tuning parameter.

        Returns
        -------
        float
            Estimated beta.
        """
        calc = GroupIVCalculator(self.groups_col, self.y_col,
                                 self.w_col, self.z_col, self.x_cols, kappa)
        stats_est: Dict = calc.compute_stats(estimation_data)
        stats_sel: Dict = calc.compute_stats(selection_data)

        numerator, denominator = 0.0, 0.0
        for group, stats in stats_est.items():
            sel_stats: GroupStats = stats_sel.get(group)
            if sel_stats and sel_stats.tilde_mu >= delta:
                numerator += sel_stats.rho_hat * np.dot(stats.Z, stats.Y)
                denominator += sel_stats.rho_hat * np.dot(stats.Z, stats.W)

        if np.isclose(denominator, 0):
            raise ValueError("No valid groups selected. Adjust delta or check data.")
        return numerator / denominator

    def process_split(self, split_data: pd.DataFrame, kappa: float
                      ) -> Tuple[Dict, List[float], List[float]]:
        """
        Processes a data split to compute group-level mu_hat values and collect residuals.

        Parameters
        ----------
        split_data : pd.DataFrame
            One half of the data.
        kappa : float
            Tuning parameter.

        Returns
        -------
        Tuple[Dict, List[float], List[float]]
            - Dictionary mapping groups to mu_hat.
            - List of second-stage residuals (u_resids).
            - List of first-stage residuals (v_resids).
        """
        group_mu: Dict = {}
        u_resids: List[float] = []
        v_resids: List[float] = []

        for group, df in split_data.groupby(self.groups_col):
            X = df[self.x_cols].values
            W = df[self.w_col].values.ravel()
            Y = df[self.y_col].values.ravel()
            Z_tilde = df[self.z_col].values.ravel()

            n = len(df)
            d = X.shape[1] if X.ndim > 1 else 1
            if n <= d + 1:
                continue

            # Residualize the instrument.
            Z = residualize(Z_tilde, X)
            if np.allclose(np.dot(Z, Z), 0):
                continue

            # Compute first-stage coefficient.
            rho_hat = np.dot(Z, W) / np.dot(Z, Z)
            mu_hat = rho_hat * np.sqrt(np.dot(Z, Z))
            group_mu[group] = float(mu_hat)  # Ensure it's a float

            # First-stage regression: W on Z.
            Z_const = add_constant(Z)
            beta_first, _, _, _ = np.linalg.lstsq(Z_const, W, rcond=None)
            W_hat = Z_const @ beta_first
            v = W - W_hat

            # Second-stage regression: Y on predicted W and X.
            X_const = add_constant(X)
            X_second = np.column_stack((W_hat, X_const))
            beta_second, _, _, _ = np.linalg.lstsq(X_second, Y, rcond=None)
            u = Y - X_second @ beta_second

            u_resids.extend(u.tolist())
            v_resids.extend(v.tolist())

        return group_mu, u_resids, v_resids

    def fit(self, data: pd.DataFrame) -> float:
        """
        Fits the adaptive IV estimator to the full dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The full dataset.

        Returns
        -------
        float
            Final estimated beta.
        """
        data_a, data_b = train_test_split(
            data, test_size=0.5, random_state=self.random_state)

        # Set kappa based on the number of groups.
        G = data[self.groups_col].nunique()
        kappa = (np.log(G)) ** 2

        # Process each split.
        group_mu_a, u_a, v_a = self.process_split(data_a, kappa)
        group_mu_b, u_b, v_b = self.process_split(data_b, kappa)

        sigma_u_sq_a, sigma_v_sq_a, sigma_uv_a = VarianceEstimator.compute(u_a, v_a)
        sigma_u_sq_b, sigma_v_sq_b, sigma_uv_b = VarianceEstimator.compute(u_b, v_b)

        mu_hat_a = list(group_mu_a.values())
        mu_hat_b = list(group_mu_b.values())

        N_a = len(data_a)
        N_b = len(data_b)

        delta_hat_a = self.compute_adaptive_delta(mu_hat_values=mu_hat_b,
                                                    sigma_u_sq=sigma_u_sq_a,
                                                    sigma_v_sq=sigma_v_sq_a,
                                                    sigma_uv=sigma_uv_a,
                                                    N=N_a,
                                                    kappa=kappa)
        delta_hat_b = self.compute_adaptive_delta(mu_hat_values=mu_hat_a,
                                                    sigma_u_sq=sigma_u_sq_b,
                                                    sigma_v_sq=sigma_v_sq_b,
                                                    sigma_uv=sigma_uv_b,
                                                    N=N_b,
                                                    kappa=kappa)

        beta_a = self.select_and_interact(estimation_data=data_a,
                                          selection_data=data_b,
                                          delta=delta_hat_a, kappa=kappa)
        beta_b = self.select_and_interact(estimation_data=data_b,
                                          selection_data=data_a,
                                          delta=delta_hat_b, kappa=kappa)
        return (beta_a + beta_b) / 2





# %%
# # %%
def gen_data(seed=None):
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
group_col = ['group']
y_col = 'Y'
iv_col = 'D'
z_cols = 'Z'
x_cols = ['X1', 'X2']

beta_aiv = AdaptiveIVEstimator(group_col, y_col, iv_col, z_cols, x_cols, random_state=42).fit(df)
beta_aiv
