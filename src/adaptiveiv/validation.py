from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .estimators import (
    compute_group_statistics,
    fit_liml_interacted,
    split_select_and_interact,
)
from .model import AdaptiveIV


def estimate_methods_once(
    data: pd.DataFrame,
    *,
    dependent: str = "Y",
    endogenous: str = "W",
    instrument: str = "Z",
    exog: list[str] | None = None,
    groups: str = "group",
    true_beta: float | None = None,
    random_state: int | None = None,
    n_splits: int = 1,
    oracle_strength_threshold: float = 0.5,
    methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Fit the implemented validation estimators on one simulated dataset."""
    exog = ["X"] if exog is None and "X" in data.columns else (exog or [])
    true_beta = _infer_true_beta(data, true_beta)
    method_list = list(
        methods
        or [
            "pooled",
            "fully_interacted",
            "split_interacted",
            "liml_interacted",
            "adaptive",
            "oracle",
        ]
    )
    rows = []
    model = AdaptiveIV(
        data=data,
        dependent=dependent,
        endogenous=endogenous,
        instruments=instrument,
        exog=exog,
        groups=groups,
    )

    for method in method_list:
        if method == "liml_interacted":
            rows.append(
                _benchmark_row(
                    fit_liml_interacted(
                        data,
                        dependent=dependent,
                        endogenous=endogenous,
                        instrument=instrument,
                        exog=exog,
                        groups=groups,
                    ),
                    true_beta=true_beta,
                    nobs=len(data),
                    ngroups=int(data[groups].nunique()),
                    n_splits=n_splits,
                )
            )
            continue
        if method == "oracle":
            rows.append(
                _oracle_row(
                    model,
                    data,
                    dependent=dependent,
                    endogenous=endogenous,
                    instrument=instrument,
                    exog=exog,
                    groups=groups,
                    true_beta=true_beta,
                    random_state=random_state,
                    oracle_strength_threshold=oracle_strength_threshold,
                )
            )
            continue
        try:
            result = model.fit(
                method=method,
                random_state=random_state,
                n_splits=n_splits,
                cov_type="none",
            )
            rows.append(
                {
                    "method": method,
                    "beta_hat": float(result.params[endogenous]),
                    "true_beta": true_beta,
                    "nobs": len(data),
                    "ngroups": int(data[groups].nunique()),
                    "selected_total": result.selection_summary.get("selected_total", 0),
                    "threshold_selected_total": result.selection_summary.get(
                        "threshold_selected_total",
                        result.selection_summary.get("selected_total", 0),
                    ),
                    "n_splits": result.selection_summary.get("n_splits", 1),
                    "finite_split_estimates": result.selection_summary.get(
                        "finite_split_estimates",
                        1,
                    ),
                    "finite": bool(np.isfinite(result.params[endogenous])),
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - exercised by failure artifacts.
            rows.append(
                _failed_row(
                    method,
                    true_beta,
                    len(data),
                    data[groups].nunique(),
                    n_splits,
                    exc,
                )
            )

    return pd.DataFrame(rows)


def summarize_simulation_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate simulation rows into method-level validation metrics."""
    if results.empty:
        return pd.DataFrame()

    group_cols = [
        column
        for column in [
            "config_index",
            "scenario",
            "source_table",
            "dgp",
            "error_distribution",
            "n_groups",
            "n_per_group",
            "strong_fraction",
            "weak_fraction",
            "method",
        ]
        if column in results.columns
    ]
    if "method" not in group_cols:
        group_cols.append("method")

    rows: list[dict[str, Any]] = []
    for keys, group in results.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        beta_hat = pd.to_numeric(group["beta_hat"], errors="coerce")
        true_beta = pd.to_numeric(group["true_beta"], errors="coerce")
        finite = np.isfinite(beta_hat.to_numpy(dtype=float))
        errors = beta_hat[finite].to_numpy(dtype=float) - true_beta[finite].to_numpy(
            dtype=float
        )
        abs_errors = np.abs(errors)
        squared_errors = errors**2
        nobs = float(pd.to_numeric(group["nobs"], errors="coerce").median())
        total_squared_error = float(np.sum(squared_errors)) if errors.size else np.nan
        max_squared_error = float(np.max(squared_errors)) if errors.size else np.nan
        top_mse_share = (
            max_squared_error / total_squared_error
            if errors.size and total_squared_error > 0
            else np.nan
        )
        row.update(
            {
                "repetitions": int(len(group)),
                "finite_share": float(np.mean(finite)) if len(finite) else 0.0,
                "mean_beta": float(np.mean(beta_hat[finite])) if errors.size else np.nan,
                "bias": float(np.mean(errors)) if errors.size else np.nan,
                "mse": float(np.mean(squared_errors)) if errors.size else np.nan,
                "scaled_mse": float(nobs * np.mean(squared_errors))
                if errors.size and np.isfinite(nobs)
                else np.nan,
                "mad": float(np.median(abs_errors)) if errors.size else np.nan,
                "max_abs_error": float(np.max(abs_errors)) if errors.size else np.nan,
                "q95_abs_error": float(np.quantile(abs_errors, 0.95))
                if errors.size
                else np.nan,
                "q99_abs_error": float(np.quantile(abs_errors, 0.99))
                if errors.size
                else np.nan,
                "max_scaled_sq_error": float(nobs * max_squared_error)
                if errors.size and np.isfinite(nobs)
                else np.nan,
                "top_abs_error_mse_share": float(top_mse_share)
                if np.isfinite(top_mse_share)
                else np.nan,
                "mean_selected_total": float(group["selected_total"].mean()),
                "mean_threshold_selected_total": float(
                    group["threshold_selected_total"].mean()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def validate_simulation_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Return qualitative validation checks for a simulation summary table."""
    rows = []
    for scenario, scenario_summary in _iter_scenarios(summary):
        by_method = scenario_summary.set_index("method")
        rows.extend(
            [
                _comparison_check(
                    scenario,
                    by_method,
                    "adaptive_improves_over_pooled",
                    "adaptive",
                    "pooled",
                    "scaled_mse",
                    operator="<=",
                    tolerance=1.0,
                ),
                _comparison_check(
                    scenario,
                    by_method,
                    "adaptive_close_to_oracle",
                    "adaptive",
                    "oracle",
                    "scaled_mse",
                    operator="<=",
                    tolerance=1.50,
                ),
                _comparison_check(
                    scenario,
                    by_method,
                    "adaptive_selects_no_more_than_split_interacted",
                    "adaptive",
                    "split_interacted",
                    "mean_selected_total",
                    operator="<=",
                    tolerance=1.0,
                ),
            ]
        )
    return pd.DataFrame(rows)


def _oracle_row(
    model: AdaptiveIV,
    data: pd.DataFrame,
    *,
    dependent: str,
    endogenous: str,
    instrument: str,
    exog: list[str],
    groups: str,
    true_beta: float,
    random_state: int | None,
    oracle_strength_threshold: float,
) -> dict[str, Any]:
    if "rho_g" not in data.columns:
        raise ValueError("oracle validation requires a rho_g column")
    oracle_groups = (
        data[[groups, "rho_g"]]
        .drop_duplicates()
        .loc[lambda frame: frame["rho_g"] >= oracle_strength_threshold, groups]
        .tolist()
    )
    split_a, split_b = model._split_data(random_state)
    stats_a, _ = compute_group_statistics(
        split_a, dependent, endogenous, instrument, exog, groups, "a"
    )
    stats_b, _ = compute_group_statistics(
        split_b, dependent, endogenous, instrument, exog, groups, "b"
    )
    estimate_a = split_select_and_interact(stats_a, stats_b, oracle_groups, "a")
    estimate_b = split_select_and_interact(stats_b, stats_a, oracle_groups, "b")
    betas = [
        estimate.beta
        for estimate in [estimate_a, estimate_b]
        if np.isfinite(estimate.beta)
    ]
    beta_hat = float(np.mean(betas)) if betas else np.nan
    selected = set(estimate_a.selected_groups) | set(estimate_b.selected_groups)
    return {
        "method": "oracle",
        "beta_hat": beta_hat,
        "true_beta": true_beta,
        "nobs": len(data),
        "ngroups": int(data[groups].nunique()),
        "selected_total": len(selected),
        "threshold_selected_total": len(set(oracle_groups)),
        "finite": bool(np.isfinite(beta_hat)),
        "error": "",
    }


def _benchmark_row(
    estimate: Any,
    *,
    true_beta: float,
    nobs: int,
    ngroups: int,
    n_splits: int,
) -> dict[str, Any]:
    beta_hat = float(estimate.beta)
    return {
        "method": estimate.method,
        "beta_hat": beta_hat,
        "true_beta": true_beta,
        "nobs": nobs,
        "ngroups": int(ngroups),
        "selected_total": 0,
        "threshold_selected_total": 0,
        "n_splits": n_splits,
        "finite_split_estimates": int(np.isfinite(beta_hat)),
        "finite": bool(np.isfinite(beta_hat)),
        "error": "",
    }


def _failed_row(
    method: str,
    true_beta: float,
    nobs: int,
    ngroups: int,
    n_splits: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "method": method,
        "beta_hat": np.nan,
        "true_beta": true_beta,
        "nobs": nobs,
        "ngroups": int(ngroups),
        "selected_total": 0,
        "threshold_selected_total": 0,
        "n_splits": n_splits,
        "finite_split_estimates": 0,
        "finite": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _infer_true_beta(data: pd.DataFrame, true_beta: float | None) -> float:
    if true_beta is not None:
        return float(true_beta)
    if "beta" in data.columns:
        return float(data["beta"].iloc[0])
    return 0.0


def _iter_scenarios(summary: pd.DataFrame):
    if "scenario" not in summary.columns:
        yield "all", summary
        return
    for scenario, scenario_summary in summary.groupby("scenario", sort=False):
        yield str(scenario), scenario_summary


def _comparison_check(
    scenario: str,
    by_method: pd.DataFrame,
    check: str,
    left_method: str,
    right_method: str,
    metric: str,
    *,
    operator: str,
    tolerance: float,
) -> dict[str, Any]:
    if left_method not in by_method.index or right_method not in by_method.index:
        return {
            "scenario": scenario,
            "check": check,
            "passed": False,
            "detail": f"missing {left_method} or {right_method}",
        }
    left = float(by_method.loc[left_method, metric])
    right = float(by_method.loc[right_method, metric])
    if operator == "<=":
        passed = bool(left <= tolerance * right)
        detail = f"{left_method} {metric}={left:.6g}; {right_method} {metric}={right:.6g}"
    else:  # pragma: no cover - no other operators are used yet.
        raise ValueError(f"Unsupported operator: {operator}")
    return {"scenario": scenario, "check": check, "passed": passed, "detail": detail}
