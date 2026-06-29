from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd

from .estimators import (
    InferenceEstimate,
    adaptive_threshold,
    compute_group_statistics,
    fit_fully_interacted_2sls,
    fit_pooled_2sls,
    homoskedastic_split_inference,
    split_select_and_interact,
    variance_estimates,
)
from .formula import parse_iv_formula
from .results import AdaptiveIVResults


@dataclass(frozen=True)
class InferenceSupport:
    """Machine-readable support status for an inference request."""

    supported: bool
    method: str
    cov_type: str
    cov_estimator: str | None
    reference_distribution: str | None
    reason: str
    required_cov_type: str | None = None


class AdaptiveIV:
    """Adaptive split-sample select-and-interact IV model."""

    def __init__(
        self,
        data: pd.DataFrame,
        dependent: str,
        endogenous: str | None = None,
        instruments: str | list[str] | None = None,
        exog: list[str] | None = None,
        groups: str | None = None,
        *,
        exog_endog: str | None = None,
        exog_exog: list[str] | None = None,
        instrument: str | None = None,
        formula: str | None = None,
        add_intercept: bool = True,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

        if endogenous is None:
            endogenous = exog_endog
        if instruments is None:
            instruments = instrument
        if exog is None:
            exog = exog_exog or []
        if groups is None:
            raise ValueError("groups must be provided")
        if endogenous is None:
            raise ValueError("endogenous must be provided")
        if instruments is None:
            raise ValueError("instruments must be provided")

        instrument_list = [instruments] if isinstance(instruments, str) else list(instruments)
        if len(instrument_list) != 1:
            raise ValueError("adaptiveiv currently supports one excluded instrument")

        self.data = data.copy()
        self.dependent = dependent
        self.endogenous = endogenous
        self.instruments = instrument_list
        self.instrument = instrument_list[0]
        self.exog = list(exog)
        self.groups = groups
        self.formula = formula
        self.add_intercept = add_intercept

        self.exog_endog = endogenous
        self.exog_exog = self.exog

        required = [dependent, endogenous, self.instrument, groups] + self.exog
        missing = [column for column in required if column not in self.data.columns]
        if missing:
            raise ValueError(f"Missing columns in DataFrame: {', '.join(missing)}")

        for column in [dependent, endogenous, self.instrument] + self.exog:
            self.data[column] = pd.to_numeric(self.data[column])

        original_nobs = len(self.data)
        self.data = self.data.dropna(subset=required).copy()
        dropped = original_nobs - len(self.data)
        if dropped:
            warnings.warn(
                f"Dropped {dropped} rows with missing model values.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.data.empty:
            raise ValueError("No observations remain after dropping missing values")

        self.nobs = len(self.data)
        self.ngroups = int(self.data[groups].nunique())
        if self.ngroups <= 1:
            raise ValueError("AdaptiveIV requires at least two groups")

    @classmethod
    def from_formula(
        cls,
        formula: str,
        data: pd.DataFrame,
        groups: str,
    ) -> "AdaptiveIV":
        spec = parse_iv_formula(formula)
        if not spec.add_intercept:
            raise ValueError(
                "No-intercept formulas are not supported; a group intercept is required."
            )
        return cls(
            data=data,
            dependent=spec.dependent,
            endogenous=spec.endogenous,
            instruments=spec.instrument,
            exog=spec.exog,
            groups=groups,
            formula=formula,
            add_intercept=spec.add_intercept,
        )

    def fit(
        self,
        *,
        method: str = "adaptive",
        random_state: int | None = None,
        n_splits: int = 1,
        kappa: str | float = "log2",
        kappa_power: float | None = None,
        cov_type: str | None = "homoskedastic",
        delta: float | None = None,
        selection_rule: str = "positive",
    ) -> AdaptiveIVResults:
        if selection_rule not in {"positive", "absolute"}:
            raise ValueError("selection_rule must be 'positive' or 'absolute'")
        if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("n_splits must be a positive integer")

        method = method.replace("-", "_")
        cov_request = self._resolve_cov_type(cov_type)
        inference_requested = cov_request in {"homoskedastic", "unadjusted"}
        if inference_requested and n_splits > 1:
            raise ValueError(
                "Inference for n_splits > 1 is not yet available. Use "
                "cov_type='none' for point estimates and stability diagnostics."
            )
        if inference_requested and selection_rule == "absolute":
            raise ValueError(
                "Inference for selection_rule='absolute' is not yet available. "
                "Use cov_type='none' for point estimates and diagnostics."
            )
        kappa_value = self._resolve_kappa(kappa, kappa_power)

        if method == "pooled":
            if inference_requested:
                raise ValueError(
                    "Inference for method='pooled' is not yet available. Use "
                    "cov_type='none' for benchmark point estimates."
                )
            estimate = fit_pooled_2sls(
                self.data,
                dependent=self.dependent,
                endogenous=self.endogenous,
                instrument=self.instrument,
                exog=self.exog,
            )
            return self._benchmark_results(estimate.beta, method, cov_request, kappa_value)

        if method in {"interacted", "fully_interacted"}:
            if inference_requested:
                raise ValueError(
                    "Inference for method='fully_interacted' is not yet available. "
                    "Use cov_type='none' for benchmark point estimates."
                )
            estimate = fit_fully_interacted_2sls(
                self.data,
                dependent=self.dependent,
                endogenous=self.endogenous,
                instrument=self.instrument,
                exog=self.exog,
                groups=self.groups,
            )
            return self._benchmark_results(
                estimate.beta, "fully_interacted", cov_request, kappa_value
            )

        split_seeds = self._split_seeds(random_state, n_splits)
        split_estimates: dict[str, float] = {}
        thresholds: dict[str, Any] = {}
        selected_groups: dict[str, list[Any]] = {}
        group_diagnostics_parts: list[pd.DataFrame] = []
        component_diagnostics_parts: list[pd.DataFrame] = []
        summaries: list[dict[str, Any]] = []
        all_warnings: list[str] = []
        inference_estimates: list[Any] = []

        for repetition, split_seed in enumerate(split_seeds):
            repetition_result = self._fit_split_repetition(
                method=method,
                split_seed=split_seed,
                repetition=repetition,
                n_splits=n_splits,
                kappa_value=kappa_value,
                delta=delta,
                selection_rule=selection_rule,
                inference_requested=inference_requested,
            )
            split_estimates.update(repetition_result["split_estimates"])
            thresholds.update(repetition_result["thresholds"])
            selected_groups.update(repetition_result["selected_groups"])
            group_diagnostics_parts.append(repetition_result["group_diagnostics"])
            component_diagnostics_parts.append(repetition_result["component_diagnostics"])
            summaries.append(repetition_result["selection_summary"])
            all_warnings.extend(repetition_result["warnings"])
            if inference_requested:
                inference_estimates.append(repetition_result["inference"])

        component_diagnostics = pd.concat(
            component_diagnostics_parts,
            ignore_index=True,
        )
        betas = component_diagnostics.loc[
            np.isfinite(component_diagnostics["beta"]),
            "beta",
        ].to_numpy(dtype=float)
        if len(betas) == 0:
            raise RuntimeError("No finite split-sample estimates were produced")
        beta = float(np.mean(betas))
        inference = inference_estimates[0] if inference_estimates else None
        if inference_requested and inference is None:
            raise RuntimeError(
                "Homoskedastic inference could not be computed for this fit. "
                "Use cov_type='none' for point estimates and diagnostics."
            )

        result = AdaptiveIVResults(
            beta=beta,
            param_name=self.endogenous,
            model=self,
            method="adaptive" if method == "adaptive" else method,
            nobs=self.nobs,
            ngroups=self.ngroups,
            cov_type=cov_request,
            bse=inference.bse if inference is not None else None,
            cov_estimator="paper_homoskedastic" if inference is not None else None,
            inference_notes=inference.notes if inference is not None else [],
            inference_diagnostics=_inference_diagnostics(inference),
            df_resid=inference.df_resid if inference is not None else None,
            split_estimates=split_estimates,
            thresholds=thresholds,
            selected_groups=selected_groups,
            group_diagnostics=pd.concat(group_diagnostics_parts, ignore_index=True),
            component_diagnostics=component_diagnostics,
            selection_summary=self._combine_selection_summaries(
                summaries,
                component_diagnostics,
                selected_groups,
                thresholds,
                n_splits,
            ),
            warnings=all_warnings,
            random_state=random_state,
            kappa=kappa_value,
            )
        return result

    def inference_support(
        self,
        *,
        method: str = "adaptive",
        cov_type: str | None = "homoskedastic",
        n_splits: int = 1,
        selection_rule: str = "positive",
    ) -> InferenceSupport:
        """Describe whether a covariance request has validated inference support."""
        method_value = method.replace("-", "_")
        cov_label = "none" if cov_type is None else str(cov_type).lower()
        try:
            cov_request = self._resolve_cov_type(cov_type)
        except ValueError as error:
            return InferenceSupport(
                supported=False,
                method=method_value,
                cov_type=cov_label,
                cov_estimator=None,
                reference_distribution=None,
                reason=str(error),
                required_cov_type="homoskedastic",
            )

        if cov_request == "none":
            return InferenceSupport(
                supported=False,
                method=method_value,
                cov_type=cov_request,
                cov_estimator=None,
                reference_distribution=None,
                reason=(
                    "point-estimate-only cov_type requested; use "
                    "cov_type='homoskedastic' for supported analytic inference"
                ),
                required_cov_type="homoskedastic",
            )
        if not isinstance(n_splits, int) or isinstance(n_splits, bool) or n_splits < 1:
            return InferenceSupport(
                supported=False,
                method=method_value,
                cov_type=cov_request,
                cov_estimator=None,
                reference_distribution=None,
                reason="n_splits must be a positive integer",
                required_cov_type="homoskedastic",
            )
        if n_splits > 1:
            return InferenceSupport(
                supported=False,
                method=method_value,
                cov_type=cov_request,
                cov_estimator=None,
                reference_distribution=None,
                reason="Inference for n_splits > 1 is not yet available",
                required_cov_type=None,
            )
        if selection_rule not in {"positive", "absolute"}:
            return InferenceSupport(
                supported=False,
                method=method_value,
                cov_type=cov_request,
                cov_estimator=None,
                reference_distribution=None,
                reason="selection_rule must be 'positive' or 'absolute'",
                required_cov_type="homoskedastic",
            )
        if selection_rule == "absolute":
            return InferenceSupport(
                supported=False,
                method=method_value,
                cov_type=cov_request,
                cov_estimator=None,
                reference_distribution=None,
                reason="Inference for selection_rule='absolute' is not yet available",
                required_cov_type=None,
            )
        if method_value == "pooled":
            return InferenceSupport(
                supported=False,
                method=method_value,
                cov_type=cov_request,
                cov_estimator=None,
                reference_distribution=None,
                reason="Inference for method='pooled' is not yet available",
                required_cov_type=None,
            )
        if method_value in {"interacted", "fully_interacted"}:
            return InferenceSupport(
                supported=False,
                method="fully_interacted",
                cov_type=cov_request,
                cov_estimator=None,
                reference_distribution=None,
                reason="Inference for method='fully_interacted' is not yet available",
                required_cov_type=None,
            )
        if method_value not in {"adaptive", "select", "split_interacted"}:
            return InferenceSupport(
                supported=False,
                method=method_value,
                cov_type=cov_request,
                cov_estimator=None,
                reference_distribution=None,
                reason=f"Unknown estimation method: {method}",
                required_cov_type=None,
            )
        return InferenceSupport(
            supported=True,
            method=method_value,
            cov_type=cov_request,
            cov_estimator="paper_homoskedastic",
            reference_distribution="normal",
            reason="supported",
            required_cov_type=None,
        )

    def supports_inference(
        self,
        *,
        method: str = "adaptive",
        cov_type: str | None = "homoskedastic",
        n_splits: int = 1,
        selection_rule: str = "positive",
    ) -> bool:
        """Return whether a covariance request has validated inference support."""
        return self.inference_support(
            method=method,
            cov_type=cov_type,
            n_splits=n_splits,
            selection_rule=selection_rule,
        ).supported

    def _fit_split_repetition(
        self,
        *,
        method: str,
        split_seed: int | None,
        repetition: int,
        n_splits: int,
        kappa_value: float,
        delta: float | None,
        selection_rule: str,
        inference_requested: bool,
    ) -> dict[str, Any]:
        split_a, split_b = self._split_data(split_seed)
        stats_a, warnings_a = compute_group_statistics(
            split_a,
            self.dependent,
            self.endogenous,
            self.instrument,
            self.exog,
            self.groups,
            "a",
        )
        stats_b, warnings_b = compute_group_statistics(
            split_b,
            self.dependent,
            self.endogenous,
            self.instrument,
            self.exog,
            self.groups,
            "b",
        )

        if not stats_a or not stats_b:
            raise RuntimeError("No usable groups in one or both sample splits")

        threshold_a, threshold_b = self._thresholds(
            method,
            stats_a,
            stats_b,
            split_a,
            split_b,
            kappa_value,
            delta,
            selection_rule,
        )
        estimate_a = split_select_and_interact(
            stats_est=stats_a,
            stats_sel=stats_b,
            selected_groups=threshold_a.selected_groups,
            split_label="a",
        )
        estimate_b = split_select_and_interact(
            stats_est=stats_b,
            stats_sel=stats_a,
            selected_groups=threshold_b.selected_groups,
            split_label="b",
        )
        inference = None
        if (
            inference_requested
            and selection_rule == "positive"
            and np.isfinite(estimate_a.beta)
            and np.isfinite(estimate_b.beta)
        ):
            inference = homoskedastic_split_inference(
                stats_a,
                stats_b,
                estimate_a,
                estimate_b,
                beta=float((estimate_a.beta + estimate_b.beta) / 2.0),
            )

        key_a = "a" if n_splits == 1 else f"r{repetition}:a"
        key_b = "b" if n_splits == 1 else f"r{repetition}:b"
        group_diagnostics = self._group_diagnostics(
            stats_a,
            stats_b,
            estimate_a.selected_groups,
            estimate_b.selected_groups,
            repetition=repetition,
        )
        component_diagnostics = self._component_diagnostics(
            estimate_a,
            estimate_b,
            repetition=repetition,
        )
        selection_summary = self._selection_summary(
                stats_a,
                stats_b,
                threshold_a,
                threshold_b,
                estimate_a.selected_groups,
                estimate_b.selected_groups,
        )
        return {
            "split_estimates": {key_a: estimate_a.beta, key_b: estimate_b.beta},
            "thresholds": {key_a: threshold_a, key_b: threshold_b},
            "selected_groups": {
                key_a: estimate_a.selected_groups,
                key_b: estimate_b.selected_groups,
            },
            "group_diagnostics": group_diagnostics,
            "component_diagnostics": component_diagnostics,
            "selection_summary": selection_summary,
            "inference": inference,
            "warnings": warnings_a + warnings_b,
        }

    def _resolve_cov_type(self, cov_type: str | None) -> str:
        if cov_type is None:
            return "none"
        cov_value = str(cov_type).lower()
        if cov_value not in {"none", "homoskedastic", "unadjusted"}:
            raise ValueError(
                "Unsupported cov_type. Supported values are None, 'none', "
                "'homoskedastic', and 'unadjusted'."
            )
        return cov_value

    def _resolve_kappa(self, kappa: str | float, kappa_power: float | None) -> float:
        if kappa_power is not None:
            return float(np.log(self.ngroups) ** kappa_power)
        if isinstance(kappa, str):
            if kappa == "log2":
                return float(np.log(self.ngroups) ** 2)
            raise ValueError("kappa must be 'log2' or a positive float")
        kappa_value = float(kappa)
        if kappa_value <= 0:
            raise ValueError("kappa must be positive")
        return kappa_value

    def _split_data(self, random_state: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(random_state)
        a_indices: list[Any] = []
        b_indices: list[Any] = []
        for _, group_data in self.data.groupby(self.groups, sort=False):
            indices = np.array(group_data.index)
            rng.shuffle(indices)
            cut = len(indices) // 2
            if cut == 0:
                b_indices.extend(indices.tolist())
            else:
                a_indices.extend(indices[:cut].tolist())
                b_indices.extend(indices[cut:].tolist())
        return self.data.loc[a_indices].copy(), self.data.loc[b_indices].copy()

    def _split_seeds(self, random_state: int | None, n_splits: int) -> list[int | None]:
        if n_splits == 1:
            return [random_state]
        rng = np.random.default_rng(random_state)
        return [
            int(seed)
            for seed in rng.integers(0, np.iinfo(np.uint32).max, size=n_splits)
        ]

    def _thresholds(
        self,
        method: str,
        stats_a: dict[Any, Any],
        stats_b: dict[Any, Any],
        split_a: pd.DataFrame,
        split_b: pd.DataFrame,
        kappa: float,
        delta: float | None,
        selection_rule: str,
    ):
        if method == "split_interacted":
            return (
                _fixed_threshold(-np.inf, stats_b, list(stats_b)),
                _fixed_threshold(-np.inf, stats_a, list(stats_a)),
            )
        if method == "select":
            if delta is None:
                raise ValueError("delta must be provided for method='select'")
            return (
                _fixed_threshold(delta, stats_b, selection_rule=selection_rule),
                _fixed_threshold(delta, stats_a, selection_rule=selection_rule),
            )
        if method != "adaptive":
            raise ValueError(f"Unknown estimation method: {method}")

        stats_full, _ = compute_group_statistics(
            self.data,
            self.dependent,
            self.endogenous,
            self.instrument,
            self.exog,
            self.groups,
            "full",
        )
        sigma_u_full, sigma_v_full, sigma_uv_full = variance_estimates(stats_full)
        return (
            adaptive_threshold(
                {group: stat.mu_hat for group, stat in stats_full.items()},
                sigma_u_full,
                sigma_v_full,
                sigma_uv_full,
                len(self.data),
                kappa,
                selection_rule=selection_rule,
                selection_mu_hat_by_group={
                    group: stat.mu_hat for group, stat in stats_b.items()
                },
            ),
            adaptive_threshold(
                {group: stat.mu_hat for group, stat in stats_full.items()},
                sigma_u_full,
                sigma_v_full,
                sigma_uv_full,
                len(self.data),
                kappa,
                selection_rule=selection_rule,
                selection_mu_hat_by_group={
                    group: stat.mu_hat for group, stat in stats_a.items()
                },
            ),
        )

    def _benchmark_results(
        self,
        beta: float,
        method: str,
        cov_type: str,
        kappa: float,
    ) -> AdaptiveIVResults:
        return AdaptiveIVResults(
            beta=beta,
            param_name=self.endogenous,
            model=self,
            method=method,
            nobs=self.nobs,
            ngroups=self.ngroups,
            cov_type=cov_type,
            bse=None,
            cov_estimator=None,
            inference_notes=[],
            inference_diagnostics=None,
            df_resid=None,
            split_estimates={},
            thresholds={},
            selected_groups={},
            group_diagnostics=pd.DataFrame(),
            component_diagnostics=pd.DataFrame(),
            selection_summary={"selected_total": 0},
            warnings=[],
            random_state=None,
            kappa=kappa,
        )

    def _group_diagnostics(
        self,
        stats_a: dict[Any, Any],
        stats_b: dict[Any, Any],
        selected_a: list[Any],
        selected_b: list[Any],
        *,
        repetition: int = 0,
    ) -> pd.DataFrame:
        rows = []
        selected_by_split = {"a": set(selected_a), "b": set(selected_b)}
        for split, stats in [("a", stats_a), ("b", stats_b)]:
            for group, stat in stats.items():
                rows.append(
                    {
                        "group": group,
                        "repetition": repetition,
                        "split": split,
                        "nobs": stat.nobs,
                        "rho_hat": stat.rho_hat,
                        "mu_hat": stat.mu_hat,
                        "selected": group in selected_by_split[split],
                        "usable": stat.usable,
                        "z_variance": stat.z_variance,
                        "skip_reason": stat.skip_reason,
                    }
                )
        return pd.DataFrame(rows)

    def _component_diagnostics(
        self,
        estimate_a: Any,
        estimate_b: Any,
        *,
        repetition: int = 0,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "repetition": repetition,
                    "split": estimate.split,
                    "beta": estimate.beta,
                    "numerator": estimate.numerator,
                    "denominator": estimate.denominator,
                    "selected_count": estimate.selected_count,
                }
                for estimate in [estimate_a, estimate_b]
            ]
        )

    def _combine_selection_summaries(
        self,
        summaries: list[dict[str, Any]],
        component_diagnostics: pd.DataFrame,
        selected_groups: dict[str, list[Any]],
        thresholds: dict[str, Any],
        n_splits: int,
    ) -> dict[str, Any]:
        finite_beta = component_diagnostics.loc[
            np.isfinite(component_diagnostics["beta"]),
            "beta",
        ].to_numpy(dtype=float)
        selected_union = set().union(*(set(groups) for groups in selected_groups.values()))
        threshold_union = set().union(
            *(set(threshold.selected_groups) for threshold in thresholds.values())
        )
        combined = {
            "n_splits": n_splits,
            "finite_split_estimates": int(len(finite_beta)),
            "split_beta_mean": float(np.mean(finite_beta)) if len(finite_beta) else np.nan,
            "split_beta_sd": float(np.std(finite_beta, ddof=1))
            if len(finite_beta) > 1
            else np.nan,
            "selected_total": len(selected_union),
            "threshold_selected_total": len(threshold_union),
            "mean_selected_total": float(
                np.mean([summary["selected_total"] for summary in summaries])
            ),
            "mean_threshold_selected_total": float(
                np.mean([summary["threshold_selected_total"] for summary in summaries])
            ),
            "selected_a": float(np.mean([summary["selected_a"] for summary in summaries])),
            "selected_b": float(np.mean([summary["selected_b"] for summary in summaries])),
            "threshold_selected_a": float(
                np.mean([summary["threshold_selected_a"] for summary in summaries])
            ),
            "threshold_selected_b": float(
                np.mean([summary["threshold_selected_b"] for summary in summaries])
            ),
            "unusable_total": float(
                np.mean([summary["unusable_total"] for summary in summaries])
            ),
            "skipped_total": float(
                np.mean([summary["skipped_total"] for summary in summaries])
            ),
            "weak_total": float(np.mean([summary["weak_total"] for summary in summaries])),
            "k_hat_a": float(np.mean([summary["k_hat_a"] for summary in summaries])),
            "k_hat_b": float(np.mean([summary["k_hat_b"] for summary in summaries])),
            "delta_a": float(np.mean([summary["delta_a"] for summary in summaries])),
            "delta_b": float(np.mean([summary["delta_b"] for summary in summaries])),
        }
        if n_splits == 1:
            combined.update(summaries[0])
            combined["n_splits"] = 1
            combined["finite_split_estimates"] = int(len(finite_beta))
            combined["split_beta_mean"] = (
                float(np.mean(finite_beta)) if len(finite_beta) else np.nan
            )
            combined["split_beta_sd"] = (
                float(np.std(finite_beta, ddof=1)) if len(finite_beta) > 1 else np.nan
            )
            combined["mean_selected_total"] = float(summaries[0]["selected_total"])
            combined["mean_threshold_selected_total"] = float(
                summaries[0]["threshold_selected_total"]
            )
        return combined

    def _selection_summary(
        self,
        stats_a: dict[Any, Any],
        stats_b: dict[Any, Any],
        threshold_a: Any,
        threshold_b: Any,
        selected_a: list[Any],
        selected_b: list[Any],
    ) -> dict[str, Any]:
        selected_a_set = set(selected_a)
        selected_b_set = set(selected_b)
        threshold_a_set = set(threshold_a.selected_groups)
        threshold_b_set = set(threshold_b.selected_groups)
        return {
            "selected_total": len(selected_a_set | selected_b_set),
            "selected_a": len(selected_a),
            "selected_b": len(selected_b),
            "threshold_selected_total": len(threshold_a_set | threshold_b_set),
            "threshold_selected_a": len(threshold_a.selected_groups),
            "threshold_selected_b": len(threshold_b.selected_groups),
            "dropped_after_threshold_a": len(threshold_a_set - selected_a_set),
            "dropped_after_threshold_b": len(threshold_b_set - selected_b_set),
            "dropped_after_threshold_total": len(threshold_a_set - selected_a_set)
            + len(threshold_b_set - selected_b_set),
            "usable_a": sum(stat.usable for stat in stats_a.values()),
            "usable_b": sum(stat.usable for stat in stats_b.values()),
            "unusable_a": sum(not stat.usable for stat in stats_a.values()),
            "unusable_b": sum(not stat.usable for stat in stats_b.values()),
            "unusable_total": sum(not stat.usable for stat in stats_a.values())
            + sum(not stat.usable for stat in stats_b.values()),
            "skipped_total": (len(stats_a) - len(selected_a))
            + (len(stats_b) - len(selected_b)),
            "weak_a": sum(
                stat.usable
                and group in stats_b
                and stats_b[group].usable
                and group not in threshold_a_set
                for group, stat in stats_a.items()
            ),
            "weak_b": sum(
                stat.usable
                and group in stats_a
                and stats_a[group].usable
                and group not in threshold_b_set
                for group, stat in stats_b.items()
            ),
            "weak_total": sum(
                stat.usable
                and group in stats_b
                and stats_b[group].usable
                and group not in threshold_a_set
                for group, stat in stats_a.items()
            )
            + sum(
                stat.usable
                and group in stats_a
                and stats_a[group].usable
                and group not in threshold_b_set
                for group, stat in stats_b.items()
            ),
            "k_hat_a": threshold_a.k_hat,
            "k_hat_b": threshold_b.k_hat,
            "delta_a": threshold_a.delta,
            "delta_b": threshold_b.delta,
        }


def _inference_diagnostics(inference: InferenceEstimate | None) -> pd.DataFrame | None:
    if inference is None:
        return None
    rows: list[dict[str, Any]] = []
    for component in ["a", "b"]:
        variance = inference.component_variances[component]
        rows.append(
            {
                "component": component,
                "variance": variance,
                "bse": float(np.sqrt(variance)),
                "df_resid": inference.component_df_resid[component],
                "cov_estimator": "paper_homoskedastic",
            }
        )
    rows.append(
        {
            "component": "average",
            "variance": inference.variance,
            "bse": inference.bse,
            "df_resid": inference.df_resid,
            "cov_estimator": "paper_homoskedastic",
        }
    )
    return pd.DataFrame(rows)


def _fixed_threshold(
    delta: float,
    stats: dict[Any, Any],
    groups: list[Any] | None = None,
    *,
    selection_rule: str = "positive",
):
    from .estimators import ThresholdResult

    if selection_rule not in {"positive", "absolute"}:
        raise ValueError("selection_rule must be 'positive' or 'absolute'")

    if groups is None:
        selected = [
            group
            for group, stat in stats.items()
            if stat.usable
            and (
                abs(stat.mu_hat)
                if selection_rule == "absolute"
                else stat.mu_hat
            )
            >= delta
        ]
    else:
        selected = [group for group in groups if stats[group].usable]
    scaled_mu_by_group = {
        group: float(
            abs(stat.mu_hat)
            if selection_rule == "absolute" and np.isfinite(stat.mu_hat)
            else stat.mu_hat
        )
        for group, stat in stats.items()
    }
    return ThresholdResult(
        delta=delta,
        k_hat=len(selected),
        selected_groups=list(selected),
        risk_by_k={},
        scaled_mu_by_group=scaled_mu_by_group,
    )
