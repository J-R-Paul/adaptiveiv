import numpy as np
import pandas as pd
import pytest

import adaptiveiv.model as model_module
from adaptiveiv import AdaptiveIV, InferenceSupport


def test_direct_api_uses_statistics_package_naming(valid_iv_data):
    model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    results = model.fit(random_state=123, kappa="log2", cov_type="homoskedastic")

    assert results.method == "adaptive"
    assert results.cov_type == "homoskedastic"
    assert isinstance(results.params, pd.Series)
    assert results.params.index.tolist() == ["W"]
    assert np.isfinite(results.params["W"])
    assert results.nobs == len(valid_iv_data)
    assert results.ngroups == valid_iv_data["group"].nunique()


def test_backward_compatible_constructor_aliases_still_work(valid_iv_data):
    model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        exog_endog="W",
        instrument="Z",
        exog_exog=["X1", "X2"],
        groups="group",
    )

    results = model.fit(random_state=123)

    assert np.isfinite(results.params["W"])


def test_results_expose_diagnostics_and_selection_summary(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123)

    assert {"a", "b"} == set(results.split_estimates)
    assert {"a", "b"} == set(results.thresholds)
    assert isinstance(results.selected_groups, dict)
    assert "a" in results.selected_groups
    assert "b" in results.selected_groups
    assert not results.group_diagnostics.empty
    assert {
        "group",
        "split",
        "rho_hat",
        "mu_hat",
        "selected",
        "usable",
        "z_variance",
    }.issubset(results.group_diagnostics.columns)
    assert results.selection_summary["selected_total"] >= 1
    assert "skipped_total" in results.selection_summary
    assert "unusable_total" in results.selection_summary
    assert "weak_total" in results.selection_summary
    assert "numerator" in results.component_diagnostics.columns
    assert "denominator" in results.component_diagnostics.columns


def test_repeated_splits_average_finite_component_estimates(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, n_splits=4, cov_type="none")

    finite_components = results.component_diagnostics.loc[
        np.isfinite(results.component_diagnostics["beta"])
    ]

    assert results.selection_summary["n_splits"] == 4
    assert results.selection_summary["finite_split_estimates"] == len(finite_components)
    assert len(results.component_diagnostics) == 8
    assert set(results.component_diagnostics["repetition"]) == {0, 1, 2, 3}
    assert np.isclose(results.params["W"], finite_components["beta"].mean())
    assert "split_beta_sd" in results.selection_summary
    assert "mean_selected_total" in results.selection_summary
    assert "r0:a" in results.split_estimates
    assert "r3:b" in results.split_estimates


def test_repeated_splits_are_reproducible_for_fixed_seed(valid_iv_data):
    model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    first = model.fit(random_state=456, n_splits=3, cov_type="none")
    second = model.fit(random_state=456, n_splits=3, cov_type="none")

    assert first.params.equals(second.params)
    pd.testing.assert_frame_equal(first.component_diagnostics, second.component_diagnostics)
    assert first.selection_summary == second.selection_summary


def test_n_splits_must_be_positive_integer(valid_iv_data):
    model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    with pytest.raises(ValueError, match="n_splits"):
        model.fit(n_splits=0)


def test_fit_exposes_absolute_strength_selection_as_explicit_opt_in(valid_iv_data):
    data = valid_iv_data.copy()
    data.loc[data["group"] == 0, "Z"] *= -1

    default_results = AdaptiveIV(
        data=data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="none")
    absolute_results = AdaptiveIV(
        data=data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, selection_rule="absolute", cov_type="none")

    default_selected = set(default_results.selected_groups["a"]) | set(
        default_results.selected_groups["b"]
    )
    absolute_selected = set(absolute_results.selected_groups["a"]) | set(
        absolute_results.selected_groups["b"]
    )
    assert 0 not in default_selected
    assert 0 in absolute_selected


def test_fixed_threshold_selection_rule_can_use_absolute_strength(valid_iv_data):
    data = valid_iv_data.copy()
    data.loc[data["group"] == 0, "Z"] *= -1

    positive_results = AdaptiveIV(
        data=data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(method="select", delta=0.1, random_state=123, cov_type="none")
    absolute_results = AdaptiveIV(
        data=data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(
        method="select",
        delta=0.1,
        random_state=123,
        selection_rule="absolute",
        cov_type="none",
    )

    positive_selected = set(positive_results.selected_groups["a"]) | set(
        positive_results.selected_groups["b"]
    )
    absolute_selected = set(absolute_results.selected_groups["a"]) | set(
        absolute_results.selected_groups["b"]
    )
    assert 0 not in positive_selected
    assert 0 in absolute_selected


def test_selected_group_diagnostics_report_groups_used_in_estimation_split():
    base = pd.DataFrame(
        {
            "Y": [0.0] * 12,
            "W": [0.0] * 12,
            "Z": np.tile(np.array([-1.0, 0.0, 1.0, 2.0, -2.0, 3.0]), 2),
            "group": [0] * 6 + [1] * 6,
        }
    )
    base["W"] = 0.8 * base["Z"] + np.linspace(-0.2, 0.2, len(base))
    base["Y"] = 0.5 * base["W"] + np.linspace(0.1, 0.3, len(base))

    setup_model = AdaptiveIV(base, "Y", endogenous="W", instruments="Z", groups="group")
    split_a, split_b = setup_model._split_data(random_state=9)
    group0_a_index = split_a.loc[split_a["group"] == 0].index
    group0_b_index = split_b.loc[split_b["group"] == 0].index

    data = base.copy()
    data.loc[group0_a_index, "Z"] = 1.0
    data.loc[group0_b_index, "Z"] = [-1.0, 0.0, 2.0]
    data["W"] = 0.8 * data["Z"] + np.linspace(-0.2, 0.2, len(data))
    data["Y"] = 0.5 * data["W"] + np.linspace(0.1, 0.3, len(data))

    results = AdaptiveIV(
        data=data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        groups="group",
    ).fit(method="split-interacted", random_state=9, cov_type="none")

    assert 0 in results.thresholds["a"].selected_groups
    assert 0 not in results.selected_groups["a"]
    assert results.selection_summary["selected_a"] == len(results.selected_groups["a"])
    assert results.selection_summary["threshold_selected_a"] == len(
        results.thresholds["a"].selected_groups
    )
    assert results.selection_summary["dropped_after_threshold_a"] >= 1
    assert results.selection_summary["weak_b"] == 0

    diagnostic = results.group_diagnostics.loc[
        (results.group_diagnostics["split"] == "a")
        & (results.group_diagnostics["group"] == 0)
    ].iloc[0]
    assert not bool(diagnostic["usable"])
    assert not bool(diagnostic["selected"])


def test_results_do_not_report_fake_inference_for_unsupported_covariance(valid_iv_data):
    with pytest.raises(ValueError, match="Unsupported cov_type"):
        AdaptiveIV(
            data=valid_iv_data,
            dependent="Y",
            endogenous="W",
            instruments="Z",
            exog=["X1", "X2"],
            groups="group",
        ).fit(random_state=123, cov_type="clustered")


def test_inference_support_reports_supported_homoskedastic_contract(valid_iv_data):
    model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    support = model.inference_support(cov_type="unadjusted")

    assert isinstance(support, InferenceSupport)
    assert support.supported is True
    assert support.cov_type == "unadjusted"
    assert support.cov_estimator == "paper_homoskedastic"
    assert support.reference_distribution == "normal"
    assert support.reason == "supported"
    assert support.required_cov_type is None
    assert model.supports_inference(cov_type="homoskedastic") is True


def test_inference_support_reports_point_estimate_only_contract(valid_iv_data):
    model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    support = model.inference_support(cov_type="none")

    assert support.supported is False
    assert support.cov_type == "none"
    assert support.required_cov_type == "homoskedastic"
    assert "point-estimate-only" in support.reason
    assert model.supports_inference(cov_type="none") is False


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"n_splits": 3}, "n_splits > 1"),
        ({"selection_rule": "absolute"}, "selection_rule='absolute'"),
        ({"method": "pooled"}, "method='pooled'"),
        ({"method": "fully_interacted"}, "method='fully_interacted'"),
        ({"cov_type": "robust"}, "Unsupported cov_type"),
    ],
)
def test_inference_support_reports_unsupported_contracts(
    valid_iv_data,
    kwargs,
    reason,
):
    model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    support = model.inference_support(**kwargs)

    assert support.supported is False
    assert reason in support.reason
    assert model.supports_inference(**kwargs) is False


def test_inference_diagnostics_expose_component_and_final_variances(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="homoskedastic")

    diagnostics = results.inference_diagnostics

    assert diagnostics["component"].tolist() == ["a", "b", "average"]
    assert {
        "component",
        "variance",
        "bse",
        "df_resid",
        "cov_estimator",
    }.issubset(diagnostics.columns)
    assert np.isclose(
        diagnostics.loc[diagnostics["component"] == "average", "variance"].iloc[0],
        float(results.cov.loc["W", "W"]),
    )
    assert np.isclose(
        diagnostics.loc[diagnostics["component"] == "average", "bse"].iloc[0],
        float(results.bse["W"]),
    )


def test_inference_diagnostics_raise_when_inference_unavailable(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="none")

    with pytest.raises(NotImplementedError, match="Inference diagnostics"):
        _ = results.inference_diagnostics


def test_homoskedastic_inference_is_available_for_supported_adaptive_fit(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="homoskedastic")

    assert results.inference_available is True
    assert results.cov_type == "homoskedastic"
    assert results.cov_estimator == "paper_homoskedastic"
    assert results.df_resid > 0
    assert results.inference_notes
    assert results.bse.index.tolist() == ["W"]
    assert results.cov.index.tolist() == ["W"]
    assert results.cov.columns.tolist() == ["W"]
    assert np.isfinite(results.bse["W"])
    assert results.bse["W"] > 0
    assert np.isclose(results.cov.loc["W", "W"], results.bse["W"] ** 2)
    assert np.isfinite(results.tvalues["W"])
    assert 0.0 <= results.pvalues["W"] <= 1.0
    conf_int = results.conf_int()
    assert conf_int.index.tolist() == ["W"]
    assert {"lower", "upper"} == set(conf_int.columns)
    assert conf_int.loc["W", "lower"] < results.params["W"] < conf_int.loc["W", "upper"]
    summary = results.summary()
    assert "std err" in summary
    assert "paper_homoskedastic" in summary


def test_results_expose_stats_ecosystem_inference_aliases(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="homoskedastic")

    pd.testing.assert_series_equal(results.std_errors, results.bse)
    pd.testing.assert_frame_equal(results.cov_params(), results.cov)
    assert results.reference_distribution == "normal"
    assert "Reference distribution: normal" in results.summary()


def test_inference_aliases_raise_when_unavailable(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="none")

    with pytest.raises(NotImplementedError, match="Standard errors"):
        _ = results.std_errors
    with pytest.raises(NotImplementedError, match="Covariance"):
        results.cov_params()
    assert results.reference_distribution is None


def test_conf_int_supports_arbitrary_normal_approximation_alpha(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="homoskedastic")

    ci_95 = results.conf_int(alpha=0.05)
    ci_90 = results.conf_int(alpha=0.10)

    assert ci_90.loc["W", "lower"] > ci_95.loc["W", "lower"]
    assert ci_90.loc["W", "upper"] < ci_95.loc["W", "upper"]
    assert ci_90.index.tolist() == ["W"]
    assert ci_90.columns.tolist() == ["lower", "upper"]


def test_unadjusted_is_homoskedastic_inference_alias(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="unadjusted")

    assert results.inference_available is True
    assert results.cov_type == "unadjusted"
    assert results.cov_estimator == "paper_homoskedastic"
    assert np.isfinite(results.bse["W"])


def test_results_mark_inference_unavailable_for_explicit_none_covariance(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="none")

    assert results.inference_available is False
    assert results.cov_type == "none"
    assert results.cov_estimator is None
    with pytest.raises(NotImplementedError, match="Standard errors"):
        _ = results.bse
    with pytest.raises(NotImplementedError, match="Covariance"):
        _ = results.cov
    with pytest.raises(NotImplementedError, match="Confidence intervals"):
        results.conf_int()
    assert "not available" in results.summary()


def test_none_covariance_does_not_compute_hidden_homoskedastic_inference(
    valid_iv_data,
    monkeypatch,
):
    def fail_if_called(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("homoskedastic inference should not be computed")

    monkeypatch.setattr(model_module, "homoskedastic_split_inference", fail_if_called)

    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="none")

    assert results.inference_available is False
    assert results.cov_estimator is None


def test_repeated_splits_require_explicit_no_inference_request(valid_iv_data):
    model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    with pytest.raises(ValueError, match="n_splits > 1"):
        model.fit(random_state=123, n_splits=3, cov_type="homoskedastic")

    results = model.fit(random_state=123, n_splits=3, cov_type="none")
    assert results.inference_available is False
    assert "split_beta_sd" in results.selection_summary


def test_absolute_selection_requires_explicit_no_inference_request(valid_iv_data):
    model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    with pytest.raises(ValueError, match="selection_rule='absolute'"):
        model.fit(random_state=123, selection_rule="absolute", cov_type="homoskedastic")

    results = model.fit(
        random_state=123,
        selection_rule="absolute",
        cov_type="none",
    )
    assert results.inference_available is False


def test_summary_mentions_inference_caveat(valid_iv_data):
    results = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    ).fit(random_state=123, cov_type="none")

    text = results.summary()

    assert "Adaptive Split-Sample Select-and-Interact IV" in text
    assert "Selected groups" in text
    assert "Threshold k_hat" in text
    assert "Covariance request: none" in text
    assert "placeholder" not in text.lower()


def test_missing_values_are_dropped_with_warning_and_unusable_groups_are_reported():
    data = pd.DataFrame(
        {
            "Y": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0,
                  2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "W": [1.2, 1.8, 2.4, 4.2, 4.8, 5.9, 7.1, 8.2,
                  1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "Z": [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0,
                  7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
            "X": [0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.5, 4.0,
                  0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "group": [0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 1, 1, 1, 1],
        }
    )

    with pytest.warns(RuntimeWarning, match="Dropped 1 rows"):
        model = AdaptiveIV(data, "Y", endogenous="W", instruments="Z", exog=["X"], groups="group")

    results = model.fit(method="split-interacted", random_state=1, cov_type="none")

    assert results.nobs == 13
    assert results.selection_summary["unusable_total"] >= 1
    assert (results.group_diagnostics["usable"] == False).any()  # noqa: E712
