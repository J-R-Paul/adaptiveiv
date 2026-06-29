from __future__ import annotations

from statistics import NormalDist
from typing import Any

import pandas as pd

from .estimators import ThresholdResult, normal_two_sided_pvalue


class AdaptiveIVResults:
    """Statsmodels-like results for adaptive IV estimators."""

    def __init__(
        self,
        *,
        beta: float,
        param_name: str,
        model: Any,
        method: str,
        nobs: int,
        ngroups: int,
        cov_type: str,
        bse: float | None,
        cov_estimator: str | None,
        inference_notes: list[str],
        inference_diagnostics: pd.DataFrame | None,
        df_resid: int | None,
        split_estimates: dict[str, float],
        thresholds: dict[str, ThresholdResult],
        selected_groups: dict[str, list[Any]],
        group_diagnostics: pd.DataFrame,
        component_diagnostics: pd.DataFrame,
        selection_summary: dict[str, Any],
        warnings: list[str],
        random_state: int | None,
        kappa: float,
    ):
        self.beta_adaptive = beta
        self.model = model
        self.method = method
        self.nobs = nobs
        self.ngroups = ngroups
        self.cov_type = cov_type
        self.cov_estimator = cov_estimator
        self.inference_notes = inference_notes
        self._inference_diagnostics = (
            inference_diagnostics.copy() if inference_diagnostics is not None else None
        )
        self.random_state = random_state
        self.kappa = kappa
        self.split_estimates = split_estimates
        self.thresholds = thresholds
        self.selected_groups = selected_groups
        self.group_diagnostics = group_diagnostics
        self.first_stage = group_diagnostics
        self.component_diagnostics = component_diagnostics
        self.selection_summary = selection_summary
        self.warnings = warnings

        self.params = pd.Series([beta], index=[param_name], dtype=float)
        self.inference_available = bse is not None
        self.reference_distribution = "normal" if self.inference_available else None
        self._bse = (
            pd.Series([bse], index=[param_name], dtype=float)
            if bse is not None
            else None
        )
        self._cov = (
            pd.DataFrame(
                [[bse**2]],
                index=[param_name],
                columns=[param_name],
                dtype=float,
            )
            if bse is not None
            else None
        )
        self.df_model = 1
        self.df_resid = int(df_resid) if df_resid is not None else max(nobs - 1, 0)

    @property
    def bse(self) -> pd.Series:
        if self._bse is None:
            raise NotImplementedError(
                "Standard errors are not yet available for this estimator."
            )
        return self._bse

    @property
    def std_errors(self) -> pd.Series:
        return self.bse

    @property
    def cov(self) -> pd.DataFrame:
        if self._cov is None:
            raise NotImplementedError(
                "Covariance estimates are not yet available for this estimator."
            )
        return self._cov

    def cov_params(self) -> pd.DataFrame:
        return self.cov

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        return self.tvalues.apply(normal_two_sided_pvalue)

    @property
    def inference_diagnostics(self) -> pd.DataFrame:
        if self._inference_diagnostics is None:
            raise NotImplementedError(
                "Inference diagnostics are not available for this estimator."
            )
        return self._inference_diagnostics.copy()

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        if not self.inference_available:
            raise NotImplementedError(
                "Confidence intervals are not yet available for this estimator."
            )
        alpha_value = float(alpha)
        if not 0.0 < alpha_value < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        critical_value = NormalDist().inv_cdf(1.0 - alpha_value / 2.0)
        lower = self.params - critical_value * self.bse
        upper = self.params + critical_value * self.bse
        return pd.DataFrame({"lower": lower, "upper": upper})

    def summary(self, title: str | None = None) -> str:
        title = title or "Adaptive Split-Sample Select-and-Interact IV Results"
        coef_data = {"coef": self.params}
        if self.inference_available:
            intervals = self.conf_int()
            coef_data.update(
                {
                    "std err": self.bse,
                    "t": self.tvalues,
                    "P>|t|": self.pvalues,
                    "[0.025": intervals["lower"],
                    "0.975]": intervals["upper"],
                }
            )
        coef_table = pd.DataFrame(coef_data)
        selection = self.selection_summary
        selected = selection.get("selected_total", 0)
        lines = [
            title,
            "=" * len(title),
            f"Estimator: {self.method}",
            f"Dep. Variable: {self.model.dependent}",
            f"Endogenous: {self.model.endogenous}",
            f"Nobs: {self.nobs}",
            f"Groups: {self.ngroups}",
            f"Split repetitions: {selection.get('n_splits', 1)}",
            "Selected groups: "
            f"{selected} (a={selection.get('selected_a', 0)}, "
            f"b={selection.get('selected_b', 0)})",
            f"Kappa: {self.kappa:.6g}",
            f"Covariance request: {self.cov_type}",
        ]
        if self.cov_estimator is not None:
            lines.append(f"Covariance estimator: {self.cov_estimator}")
        if self.reference_distribution is not None:
            lines.append(f"Reference distribution: {self.reference_distribution}")
        if "k_hat_a" in selection and "k_hat_b" in selection:
            delta_a = _format_threshold(selection.get("delta_a"))
            delta_b = _format_threshold(selection.get("delta_b"))
            lines.extend(
                [
                    "Threshold-selected groups: "
                    f"{selection.get('threshold_selected_total', selected)} "
                    f"(a={selection.get('threshold_selected_a', 0)}, "
                    f"b={selection.get('threshold_selected_b', 0)})",
                    f"Threshold k_hat: a={selection['k_hat_a']}, b={selection['k_hat_b']}",
                    f"Threshold delta: a={delta_a}, b={delta_b}",
                ]
            )
        if selection.get("n_splits", 1) > 1:
            split_sd = _format_threshold(selection.get("split_beta_sd"))
            lines.append(
                "Split estimate stability: "
                f"finite={selection.get('finite_split_estimates', 0)}, sd={split_sd}"
            )
        lines.extend(
            [
                "",
                coef_table.to_string(float_format=lambda value: f"{value: .4f}"),
                "",
                "Notes:",
            ]
        )
        if self.inference_available:
            lines.extend(f"- {note}." for note in self.inference_notes)
        else:
            lines.extend(
                [
                    "- Standard errors, p-values, and confidence intervals are not available.",
                    "- Inspect group_diagnostics and selection_summary before interpreting estimates.",
                ]
            )
        if self.warnings:
            lines.append("- Warnings were produced during fitting.")
        return "\n".join(lines)


def _format_threshold(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric == float("inf"):
        return "inf"
    if numeric == float("-inf"):
        return "-inf"
    return f"{numeric:.6g}"
