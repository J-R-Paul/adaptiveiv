from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass(frozen=True)
class GroupStatistics:
    group: Any
    split: str
    nobs: int
    z_resid: np.ndarray
    w: np.ndarray
    y: np.ndarray
    x: np.ndarray
    rho_hat: float
    mu_hat: float
    z_variance: float
    usable: bool = True
    skip_reason: str | None = None
    u_resid: np.ndarray | None = None
    v_resid: np.ndarray | None = None


def unusable_group_statistics(
    *,
    group: Any,
    split: str,
    nobs: int,
    reason: str,
) -> GroupStatistics:
    return GroupStatistics(
        group=group,
        split=split,
        nobs=nobs,
        z_resid=np.array([], dtype=float),
        w=np.array([], dtype=float),
        y=np.array([], dtype=float),
        x=np.empty((0, 0)),
        rho_hat=np.nan,
        mu_hat=np.nan,
        z_variance=np.nan,
        usable=False,
        skip_reason=reason,
    )


@dataclass(frozen=True)
class ThresholdResult:
    delta: float
    k_hat: int
    selected_groups: list[Any]
    risk_by_k: dict[int, float]
    scaled_mu_by_group: dict[Any, float]


@dataclass(frozen=True)
class SplitEstimate:
    split: str
    beta: float
    numerator: float
    denominator: float
    selected_groups: list[Any]
    selected_count: int


@dataclass(frozen=True)
class BenchmarkEstimate:
    method: str
    beta: float


@dataclass(frozen=True)
class InferenceEstimate:
    bse: float
    variance: float
    df_resid: int
    notes: list[str]
    component_variances: dict[str, float]
    component_df_resid: dict[str, int]


def normal_two_sided_pvalue(t_value: float) -> float:
    return erfc(abs(t_value) / sqrt(2.0))


def group_design(data: pd.DataFrame, exog: list[str]) -> np.ndarray:
    nobs = len(data)
    if exog:
        x = data[exog].to_numpy(dtype=float)
        return np.column_stack([np.ones(nobs), x])
    return np.ones((nobs, 1))


def residualize_on_design(values: np.ndarray, design: np.ndarray) -> np.ndarray:
    beta = np.linalg.pinv(design.T @ design) @ design.T @ values
    return values - design @ beta


def compute_group_statistics(
    data: pd.DataFrame,
    dependent: str,
    endogenous: str,
    instrument: str,
    exog: list[str],
    groups: str,
    split_label: str,
) -> tuple[dict[Any, GroupStatistics], list[str]]:
    stats: dict[Any, GroupStatistics] = {}
    warnings: list[str] = []

    for group, group_data in data.groupby(groups, sort=True):
        y = group_data[dependent].to_numpy(dtype=float)
        w = group_data[endogenous].to_numpy(dtype=float)
        z_raw = group_data[instrument].to_numpy(dtype=float)
        x = group_data[exog].to_numpy(dtype=float) if exog else np.empty((len(y), 0))
        design = group_design(group_data, exog)

        if len(y) <= design.shape[1]:
            reason = "too few observations for controls"
            warnings.append(f"Split {split_label}, group {group}: {reason}")
            stats[group] = unusable_group_statistics(
                group=group,
                split=split_label,
                nobs=len(y),
                reason=reason,
            )
            continue

        z_resid = residualize_on_design(z_raw, design)
        z_inner = float(z_resid @ z_resid)
        if not np.isfinite(z_inner) or np.isclose(z_inner, 0.0):
            reason = "zero residualized instrument variance"
            warnings.append(f"Split {split_label}, group {group}: {reason}")
            stats[group] = unusable_group_statistics(
                group=group,
                split=split_label,
                nobs=len(y),
                reason=reason,
            )
            continue

        rho_hat = float((z_resid @ w) / z_inner)
        mu_hat = float((z_resid @ w) / np.sqrt(z_inner))

        first_stage_design = np.column_stack([design, z_resid])
        fs_params = np.linalg.pinv(first_stage_design.T @ first_stage_design) @ (
            first_stage_design.T @ w
        )
        v_resid = w - first_stage_design @ fs_params

        outcome_design = np.column_stack([design, w])
        outcome_params = np.linalg.pinv(outcome_design.T @ outcome_design) @ (
            outcome_design.T @ y
        )
        u_resid = y - outcome_design @ outcome_params

        stats[group] = GroupStatistics(
            group=group,
            split=split_label,
            nobs=len(y),
            z_resid=z_resid,
            w=w,
            y=y,
            x=x,
            rho_hat=rho_hat,
            mu_hat=mu_hat,
            z_variance=float(z_inner / max(len(y), 1)),
            u_resid=u_resid,
            v_resid=v_resid,
        )

    return stats, warnings


def variance_estimates(
    stats: dict[Any, GroupStatistics],
) -> tuple[float, float, float]:
    u_parts = [stat.u_resid for stat in stats.values() if stat.u_resid is not None]
    v_parts = [stat.v_resid for stat in stats.values() if stat.v_resid is not None]
    if not u_parts or not v_parts:
        return np.nan, np.nan, np.nan
    u = np.concatenate(u_parts)
    v = np.concatenate(v_parts)
    if len(u) < 2 or len(v) < 2 or len(u) != len(v):
        return np.nan, np.nan, np.nan
    sigma_u_sq = float(np.var(u, ddof=1))
    sigma_v_sq = float(np.var(v, ddof=1))
    sigma_uv = float(np.cov(u, v, ddof=1)[0, 1])
    return sigma_u_sq, sigma_v_sq, sigma_uv


def adaptive_threshold(
    mu_hat_by_group: dict[Any, float],
    sigma_u_sq: float,
    sigma_v_sq: float,
    sigma_uv: float,
    nobs: int,
    kappa: float,
    selection_rule: str = "positive",
    selection_mu_hat_by_group: dict[Any, float] | None = None,
) -> ThresholdResult:
    if nobs <= 0:
        raise ValueError("nobs must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if selection_rule not in {"positive", "absolute"}:
        raise ValueError("selection_rule must be 'positive' or 'absolute'")
    selection_mu = selection_mu_hat_by_group or mu_hat_by_group
    if not all(np.isfinite([sigma_u_sq, sigma_v_sq, sigma_uv])):
        scaled_all = _scaled_mu_by_group(selection_mu, kappa, selection_rule)
        return ThresholdResult(np.inf, 0, [], {0: np.nan}, scaled_all)

    risk_scaled_all = _scaled_mu_by_group(mu_hat_by_group, kappa, selection_rule)
    selection_scaled_all = _scaled_mu_by_group(selection_mu, kappa, selection_rule)
    scaled = {
        group: scaled_mu
        for group, scaled_mu in risk_scaled_all.items()
        if scaled_mu > 0.0
    }
    ordered = sorted(scaled.items(), key=lambda item: item[1], reverse=True)
    g_eff = len(ordered)
    if g_eff == 0:
        return ThresholdResult(np.inf, 0, [], {0: 0.0}, selection_scaled_all)

    strengths = np.array([value for _, value in ordered], dtype=float)
    var_term = sigma_u_sq * sigma_v_sq + sigma_uv**2
    risk_by_k: dict[int, float] = {}
    for k in range(g_eff + 1):
        unselected_sum = float(np.sum(strengths[k:] ** 2))
        risk_by_k[k] = (
            (sigma_u_sq / nobs) * unselected_sum
            + 2.0 * var_term * (k / nobs)
        )

    k_hat = min(risk_by_k, key=lambda key: risk_by_k[key])
    selection_ordered = sorted(
        (
            (group, scaled_mu)
            for group, scaled_mu in selection_scaled_all.items()
            if scaled_mu > 0.0
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    selected_groups = [group for group, _ in selection_ordered[:k_hat]]
    if k_hat == 0:
        delta = np.inf
    else:
        delta = (
            float(selection_ordered[k_hat - 1][1])
            if len(selection_ordered) >= k_hat
            else np.inf
        )

    return ThresholdResult(
        delta=delta,
        k_hat=k_hat,
        selected_groups=selected_groups,
        risk_by_k=risk_by_k,
        scaled_mu_by_group=selection_scaled_all,
    )


def _scaled_mu_by_group(
    mu_hat_by_group: dict[Any, float],
    kappa: float,
    selection_rule: str,
) -> dict[Any, float]:
    return {
        group: float(
            abs(mu_hat) / np.sqrt(kappa)
            if selection_rule == "absolute"
            else mu_hat / np.sqrt(kappa)
        )
        for group, mu_hat in mu_hat_by_group.items()
        if np.isfinite(mu_hat)
    }


def split_select_and_interact(
    stats_est: dict[Any, GroupStatistics],
    stats_sel: dict[Any, GroupStatistics],
    selected_groups: Iterable[Any],
    split_label: str,
) -> SplitEstimate:
    selected = [
        group
        for group in selected_groups
        if group in stats_est
        and group in stats_sel
        and stats_est[group].usable
        and stats_sel[group].usable
    ]
    numerator = 0.0
    denominator = 0.0

    for group in selected:
        est = stats_est[group]
        sel = stats_sel[group]
        numerator += sel.rho_hat * float(est.z_resid @ est.y)
        denominator += sel.rho_hat * float(est.z_resid @ est.w)

    if not selected or np.isclose(denominator, 0.0):
        beta = np.nan
    else:
        beta = float(numerator / denominator)

    return SplitEstimate(
        split=split_label,
        beta=beta,
        numerator=float(numerator),
        denominator=float(denominator),
        selected_groups=selected,
        selected_count=len(selected),
    )


def homoskedastic_split_inference(
    stats_a: dict[Any, GroupStatistics],
    stats_b: dict[Any, GroupStatistics],
    estimate_a: SplitEstimate,
    estimate_b: SplitEstimate,
    beta: float,
) -> InferenceEstimate | None:
    """Paper-baseline homoskedastic SE for one split-sample estimate."""
    component_a = _component_homoskedastic_variance(
        stats_est=stats_a,
        stats_sel=stats_b,
        selected_groups=estimate_a.selected_groups,
        beta=estimate_a.beta,
    )
    component_b = _component_homoskedastic_variance(
        stats_est=stats_b,
        stats_sel=stats_a,
        selected_groups=estimate_b.selected_groups,
        beta=estimate_b.beta,
    )
    if component_a is None or component_b is None:
        return None

    variance_a, df_a = component_a
    variance_b, df_b = component_b
    variance = float((variance_a + variance_b) / 4.0)
    if not np.isfinite(variance) or variance <= 0.0:
        return None

    return InferenceEstimate(
        bse=float(np.sqrt(variance)),
        variance=variance,
        df_resid=int(min(df_a, df_b)),
        notes=[
            "paper_homoskedastic covariance for n_splits=1 and positive selection",
            "cross-component covariance treated as first-order negligible",
            "residual variance estimated from selected estimation groups only",
        ],
        component_variances={"a": float(variance_a), "b": float(variance_b)},
        component_df_resid={"a": int(df_a), "b": int(df_b)},
    )


def _component_homoskedastic_variance(
    *,
    stats_est: dict[Any, GroupStatistics],
    stats_sel: dict[Any, GroupStatistics],
    selected_groups: Iterable[Any],
    beta: float,
) -> tuple[float, int] | None:
    q_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    nuisance_terms = 0
    nobs = 0

    for group in selected_groups:
        if group not in stats_est or group not in stats_sel:
            continue
        est = stats_est[group]
        sel = stats_sel[group]
        if not est.usable or not sel.usable:
            continue

        q = sel.rho_hat * est.z_resid
        design = np.column_stack([np.ones(est.nobs), est.x])
        structural_resid = residualize_on_design(est.y - beta * est.w, design)
        q_parts.append(q)
        w_parts.append(est.w)
        residual_parts.append(structural_resid)
        nuisance_terms += design.shape[1]
        nobs += est.nobs

    if not q_parts:
        return None

    q_all = np.concatenate(q_parts)
    w_all = np.concatenate(w_parts)
    residual_all = np.concatenate(residual_parts)
    denominator = float(q_all @ w_all)
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        return None

    df_resid = int(nobs - nuisance_terms - 1)
    if df_resid <= 0:
        return None

    sigma_u_sq = float((residual_all @ residual_all) / df_resid)
    q_sq_sum = float(q_all @ q_all)
    variance = sigma_u_sq * q_sq_sum / (denominator**2)
    if not np.isfinite(variance) or variance <= 0.0:
        return None
    return float(variance), df_resid


def fit_pooled_2sls(
    data: pd.DataFrame,
    dependent: str,
    endogenous: str,
    instrument: str,
    exog: list[str] | None = None,
) -> BenchmarkEstimate:
    exog = exog or []
    first_stage_x = sm.add_constant(data[[instrument] + exog], has_constant="add")
    first_stage = sm.OLS(data[endogenous], first_stage_x).fit()
    second_stage_x = sm.add_constant(data[exog], has_constant="add")
    second_stage_x = second_stage_x.copy()
    second_stage_x[endogenous] = first_stage.fittedvalues
    second_stage = sm.OLS(data[dependent], second_stage_x).fit()
    return BenchmarkEstimate(method="pooled", beta=float(second_stage.params[endogenous]))


def fit_fully_interacted_2sls(
    data: pd.DataFrame,
    dependent: str,
    endogenous: str,
    instrument: str,
    exog: list[str] | None,
    groups: str,
) -> BenchmarkEstimate:
    group_stats, _ = compute_group_statistics(
        data=data,
        dependent=dependent,
        endogenous=endogenous,
        instrument=instrument,
        exog=exog or [],
        groups=groups,
        split_label="full",
    )
    selected_groups = list(group_stats)
    estimate = split_select_and_interact(
        stats_est=group_stats,
        stats_sel=group_stats,
        selected_groups=selected_groups,
        split_label="full",
    )
    return BenchmarkEstimate(method="fully_interacted", beta=estimate.beta)


def fit_liml_interacted(
    data: pd.DataFrame,
    dependent: str,
    endogenous: str,
    instrument: str,
    exog: list[str] | None,
    groups: str,
) -> BenchmarkEstimate:
    y, w, z = _interacted_residual_matrices(
        data=data,
        dependent=dependent,
        endogenous=endogenous,
        instrument=instrument,
        exog=exog or [],
        groups=groups,
    )
    if z.shape[1] == 0:
        return BenchmarkEstimate(method="liml_interacted", beta=np.nan)

    yy_ww = np.column_stack([y, w])
    yy_ww_cross = yy_ww.T @ yy_ww
    z_cross_inv = np.linalg.pinv(z.T @ z)
    projected_cross = (yy_ww.T @ z) @ z_cross_inv @ (z.T @ yy_ww)
    residual_cross = yy_ww_cross - projected_cross
    eigenvalues = np.linalg.eigvals(np.linalg.pinv(yy_ww_cross) @ residual_cross)
    real_eigenvalues = np.real(eigenvalues[np.isclose(np.imag(eigenvalues), 0.0)])
    finite_eigenvalues = real_eigenvalues[np.isfinite(real_eigenvalues)]
    if finite_eigenvalues.size == 0:
        return BenchmarkEstimate(method="liml_interacted", beta=np.nan)

    kappa = float(np.max(finite_eigenvalues))
    xpx = float(projected_cross[1, 1])
    xpy = float(projected_cross[1, 0])
    xx = float(w @ w)
    xy = float(w @ y)
    denominator = (1.0 - kappa) * xx + kappa * xpx
    numerator = (1.0 - kappa) * xy + kappa * xpy
    if not np.isfinite(denominator) or np.isclose(denominator, 0.0):
        beta = np.nan
    else:
        beta = float(numerator / denominator)
    return BenchmarkEstimate(method="liml_interacted", beta=beta)


def _interacted_residual_matrices(
    *,
    data: pd.DataFrame,
    dependent: str,
    endogenous: str,
    instrument: str,
    exog: list[str],
    groups: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []
    z_columns: list[np.ndarray] = []
    start = 0

    for _, group_data in data.groupby(groups, sort=True):
        design = group_design(group_data, exog)
        nobs = len(group_data)
        y_raw = group_data[dependent].to_numpy(dtype=float)
        w_raw = group_data[endogenous].to_numpy(dtype=float)
        z_raw = group_data[instrument].to_numpy(dtype=float)
        y_resid = residualize_on_design(y_raw, design)
        w_resid = residualize_on_design(w_raw, design)
        z_resid = residualize_on_design(z_raw, design)

        y_parts.append(y_resid)
        w_parts.append(w_resid)
        if nobs > design.shape[1] and not np.isclose(float(z_resid @ z_resid), 0.0):
            column = np.zeros(len(data), dtype=float)
            column[start : start + nobs] = z_resid
            z_columns.append(column)
        start += nobs

    y = np.concatenate(y_parts) if y_parts else np.array([], dtype=float)
    w = np.concatenate(w_parts) if w_parts else np.array([], dtype=float)
    z = (
        np.column_stack(z_columns)
        if z_columns
        else np.empty((len(y), 0), dtype=float)
    )
    return y, w, z
