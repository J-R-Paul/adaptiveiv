import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, Any

from .data_manager import DataManager


class _SplitSampleSelectInteractEstimator:
    """
    Implements the split-sample select-and-interact estimator with μ adjustment.

    This class provides functionality to compute the β_ssel,int(δ) estimator
    using a split-sample approach with cross-validation between data splits.

    Parameters
    ----------
    data : DataManager

    Methods
    -------
    estimate(data, delta, kappa, random_state=42)
        Compute the estimator using split-sample selection and interaction
    """

    def __init__(self, data: DataManager):
        if data._data_type_flag != "pandas" or not hasattr(self, "_data"):
            raise ValueError("No data loaded, or data is not a pandas DataFrame.")

        self.data = data


    def estimate(self, delta: float, kappa: float,
                random_state: int = 42) -> float:
        """
        Compute the split-sample select-and-interact estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Input data containing all required columns
        delta : float
            Selection threshold for group inclusion
        kappa : float
            Scaling factor for μ adjustment
        random_state : int, optional
            Seed for reproducible data splitting

        Returns
        -------
        float
            Estimated coefficient β

        Raises
        ------
        ValueError
            If input validation fails or no valid groups are selected
        """
        # self._validate_estimate_params(delta, kappa, random_state)

        # Split data into halves
        data_a, data_b = self.data.split_data_in_two(prop=0.5, random_state=random_state)

        # Compute group statistics for both splits
        stats_a = self._compute_group_stats(data_a, kappa)
        stats_b = self._compute_group_stats(data_b, kappa)

        # Calculate cross-split estimates
        beta_a = self._calculate_split_estimate(stats_a, stats_b, delta)
        beta_b = self._calculate_split_estimate(stats_b, stats_a, delta)

        return self._aggregate_results(beta_a, beta_b)

    # def _validate_estimate_params(delta: float,
    #                              kappa: float, random_state: int) -> None:
    #     """Validate estimation parameters"""

    #     if not isinstance(delta, (int, float)) or delta < 0:
    #         raise ValueError("delta must be a non-negative number")
    #     if not isinstance(kappa, (int, float)) or kappa <= 0:
    #         raise ValueError("kappa must be a positive number")
    #     if not isinstance(random_state, int) or None:
    #         raise TypeError("random_state must be an integer")

    def _compute_group_stats(self, data: pd.DataFrame, kappa: float) -> Dict[Any, Dict]:
        """Compute group-level statistics including μ adjustment"""
        group_stats = {}

        for group, df in data.groupby(self.data._group_col):
            if data._x_cols is None:
                # If X is not provided, use a constant
                X = np.ones((df.shape[0], 1))
            else:
                X = df[self.data._x_cols].values

            n, d = X.shape

            if n <= d + 1:  # Check sufficient observations
                continue

            # Residualize instrument
            Z_tilde = df[self.data._z_cols].values
            X_const = sm.add_constant(X, has_constant='skip')
            Z_resid = sm.OLS(Z_tilde, X_const).fit().resid

            if np.dot(Z_resid, Z_resid) == 0:  # Check instrument strength
                continue

            # Calculate group statistics
            W = df[self.data._w_col].values
            Y = df[self.data._y_col].values
            rho_hat = np.dot(Z_resid, W) / np.dot(Z_resid, Z_resid)
            mu_hat = rho_hat * np.sqrt(np.dot(Z_resid, Z_resid))

            group_stats[group] = {
                'Z': Z_resid,
                'W': W,
                'Y': Y,
                'rho_hat': rho_hat,
                'tilde_mu': mu_hat / np.sqrt(kappa)
            }

        return group_stats

    def _calculate_split_estimate(self, current_stats: Dict, other_stats: Dict,
                                 delta: float) -> float:
        """Calculate β estimate for one split using the other split's selection"""
        numerator, denominator = 0.0, 0.0

        for group, stats in current_stats.items():
            other_group_stats = other_stats.get(group)
            if other_group_stats and other_group_stats['tilde_mu'] >= delta:
                rho_hat_other = other_group_stats['rho_hat']
                Z, W, Y = stats['Z'], stats['W'], stats['Y']

                zw = Z @ W
                zy = Z @ Y

                numerator += rho_hat_other * zy
                denominator += rho_hat_other * zw

        return numerator / denominator if denominator != 0 else np.nan

    def _aggregate_results(self, beta_a: float, beta_b: float) -> float:
        """Aggregate results from both splits"""
        valid_betas = [b for b in [beta_a, beta_b] if not np.isnan(b)]

        if not valid_betas:
            raise ValueError("No valid groups selected. Adjust delta or check data.")
            # Might have to pass if using the bootstrap method as it may not always
            # have valid groups during the bootstrapping process.

        return np.mean(valid_betas)  # type: ignore
