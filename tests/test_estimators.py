import numpy as np
import pandas as pd

from adaptiveiv.estimators import (
    _component_homoskedastic_variance,
    adaptive_threshold,
    compute_group_statistics,
    fit_liml_interacted,
    homoskedastic_split_inference,
    split_select_and_interact,
)


def test_group_statistics_residualize_against_group_intercept_without_exog():
    data = pd.DataFrame(
        {
            "Y": [0.0, 1.0, 2.0, 0.0, 2.0, 4.0],
            "W": [2.0, 4.0, 6.0, -1.0, 1.0, 3.0],
            "Z": [1.0, 2.0, 3.0, -1.0, 0.0, 1.0],
            "group": [0, 0, 0, 1, 1, 1],
        }
    )

    stats, warnings = compute_group_statistics(
        data,
        dependent="Y",
        endogenous="W",
        instrument="Z",
        exog=[],
        groups="group",
        split_label="a",
    )

    assert warnings == []
    for group in [0, 1]:
        row = stats[group]
        assert np.isclose(row.z_resid.sum(), 0.0)
        assert np.isclose(row.rho_hat, 2.0)
        assert np.isclose(row.mu_hat, 2.0 * np.sqrt(np.dot(row.z_resid, row.z_resid)))


def test_group_statistics_residualized_instrument_orthogonal_to_controls(valid_iv_data):
    stats, warnings = compute_group_statistics(
        valid_iv_data,
        dependent="Y",
        endogenous="W",
        instrument="Z",
        exog=["X1", "X2"],
        groups="group",
        split_label="full",
    )

    assert warnings == []
    first = stats[0]
    x = valid_iv_data.loc[valid_iv_data["group"] == 0, ["X1", "X2"]].to_numpy()
    design = np.column_stack([np.ones(len(x)), x])
    assert np.allclose(design.T @ first.z_resid, 0.0, atol=1e-10)


def test_adaptive_threshold_selects_risk_minimizing_top_k_groups():
    result = adaptive_threshold(
        mu_hat_by_group={0: 10.0, 1: 8.0, 2: 0.1},
        sigma_u_sq=1.0,
        sigma_v_sq=1.0,
        sigma_uv=0.0,
        nobs=1000,
        kappa=1.0,
    )

    assert result.k_hat == 2
    assert result.selected_groups == [0, 1]
    assert result.delta == 8.0
    assert result.risk_by_k[2] < result.risk_by_k[1]
    assert result.risk_by_k[2] < result.risk_by_k[3]


def test_adaptive_threshold_can_use_risk_k_to_select_top_split_groups():
    result = adaptive_threshold(
        mu_hat_by_group={0: 10.0, 1: 8.0, 2: 0.1},
        sigma_u_sq=1.0,
        sigma_v_sq=1.0,
        sigma_uv=0.0,
        nobs=1000,
        kappa=1.0,
        selection_mu_hat_by_group={0: 0.1, 1: 9.0, 2: 8.0},
    )

    assert result.k_hat == 2
    assert result.selected_groups == [1, 2]
    assert result.delta == 8.0
    assert result.risk_by_k[2] < result.risk_by_k[1]
    assert result.risk_by_k[2] < result.risk_by_k[3]


def test_adaptive_threshold_one_sided_rule_ignores_negative_first_stages():
    result = adaptive_threshold(
        mu_hat_by_group={0: -10.0, 1: -8.0, 2: -0.1},
        sigma_u_sq=1.0,
        sigma_v_sq=1.0,
        sigma_uv=0.0,
        nobs=1000,
        kappa=1.0,
    )

    assert result.k_hat == 0
    assert result.selected_groups == []
    assert result.delta == np.inf


def test_adaptive_threshold_can_select_by_absolute_strength_when_requested():
    result = adaptive_threshold(
        mu_hat_by_group={0: -10.0, 1: 2.0, 2: 0.1},
        sigma_u_sq=1.0,
        sigma_v_sq=1.0,
        sigma_uv=0.0,
        nobs=1000,
        kappa=1.0,
        selection_rule="absolute",
    )

    assert result.selected_groups[0] == 0
    assert result.scaled_mu_by_group[0] == 10.0


def test_group_statistics_keep_unusable_groups_for_diagnostics():
    data = pd.DataFrame(
        {
            "Y": [1.0, 2.0, 3.0, 4.0],
            "W": [1.0, 2.0, 3.0, 4.0],
            "Z": [5.0, 5.0, 5.0, 6.0],
            "X": [0.0, 1.0, 2.0, 3.0],
            "group": [0, 0, 0, 1],
        }
    )

    stats, warnings = compute_group_statistics(
        data,
        dependent="Y",
        endogenous="W",
        instrument="Z",
        exog=[],
        groups="group",
        split_label="a",
    )

    assert set(stats) == {0, 1}
    assert stats[0].usable is False
    assert stats[0].skip_reason == "zero residualized instrument variance"
    assert stats[1].usable is False
    assert stats[1].skip_reason == "too few observations for controls"
    assert len(warnings) == 2


def test_split_select_and_interact_returns_components(valid_iv_data):
    split_a = valid_iv_data.groupby("group", group_keys=False).sample(
        frac=0.5, random_state=11
    )
    split_b = valid_iv_data.drop(split_a.index)
    stats_a, _ = compute_group_statistics(
        split_a, "Y", "W", "Z", ["X1", "X2"], "group", "a"
    )
    stats_b, _ = compute_group_statistics(
        split_b, "Y", "W", "Z", ["X1", "X2"], "group", "b"
    )

    estimate = split_select_and_interact(
        stats_est=stats_a,
        stats_sel=stats_b,
        selected_groups=[0, 1, 2, 3],
        split_label="a",
    )

    assert np.isfinite(estimate.beta)
    assert estimate.selected_groups == [0, 1, 2, 3]
    assert estimate.selected_count == 4
    assert np.isfinite(estimate.numerator)
    assert np.isfinite(estimate.denominator)


def test_homoskedastic_split_inference_matches_component_formula(valid_iv_data):
    split_a = valid_iv_data.groupby("group", group_keys=False).sample(
        frac=0.5, random_state=17
    )
    split_b = valid_iv_data.drop(split_a.index)
    stats_a, _ = compute_group_statistics(
        split_a, "Y", "W", "Z", ["X1", "X2"], "group", "a"
    )
    stats_b, _ = compute_group_statistics(
        split_b, "Y", "W", "Z", ["X1", "X2"], "group", "b"
    )
    estimate_a = split_select_and_interact(stats_a, stats_b, [0, 1, 2], "a")
    estimate_b = split_select_and_interact(stats_b, stats_a, [0, 1, 2], "b")
    beta = float((estimate_a.beta + estimate_b.beta) / 2.0)

    inference = homoskedastic_split_inference(
        stats_a,
        stats_b,
        estimate_a,
        estimate_b,
        beta=beta,
    )

    assert inference is not None
    assert inference.df_resid > 0
    assert inference.bse > 0
    assert set(inference.component_variances) == {"a", "b"}
    assert np.isclose(
        inference.variance,
        (
            inference.component_variances["a"]
            + inference.component_variances["b"]
        )
        / 4.0,
    )
    assert any("cross-component covariance" in note for note in inference.notes)


def test_homoskedastic_split_inference_uses_component_specific_residuals(
    valid_iv_data,
):
    split_a = valid_iv_data.groupby("group", group_keys=False).sample(
        frac=0.5, random_state=23
    )
    split_b = valid_iv_data.drop(split_a.index)
    stats_a, _ = compute_group_statistics(
        split_a, "Y", "W", "Z", ["X1", "X2"], "group", "a"
    )
    stats_b, _ = compute_group_statistics(
        split_b, "Y", "W", "Z", ["X1", "X2"], "group", "b"
    )
    estimate_a = split_select_and_interact(stats_a, stats_b, [0, 1, 2], "a")
    estimate_b = split_select_and_interact(stats_b, stats_a, [0, 1, 2], "b")

    inference = homoskedastic_split_inference(
        stats_a,
        stats_b,
        estimate_a,
        estimate_b,
        beta=999.0,
    )
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

    assert inference is not None
    assert component_a is not None
    assert component_b is not None
    assert np.isclose(inference.component_variances["a"], component_a[0])
    assert np.isclose(inference.component_variances["b"], component_b[0])


def test_homoskedastic_split_inference_is_unavailable_without_selected_groups(
    valid_iv_data,
):
    split_a = valid_iv_data.groupby("group", group_keys=False).sample(
        frac=0.5, random_state=19
    )
    split_b = valid_iv_data.drop(split_a.index)
    stats_a, _ = compute_group_statistics(
        split_a, "Y", "W", "Z", ["X1", "X2"], "group", "a"
    )
    stats_b, _ = compute_group_statistics(
        split_b, "Y", "W", "Z", ["X1", "X2"], "group", "b"
    )
    estimate_a = split_select_and_interact(stats_a, stats_b, [], "a")
    estimate_b = split_select_and_interact(stats_b, stats_a, [], "b")

    inference = homoskedastic_split_inference(
        stats_a,
        stats_b,
        estimate_a,
        estimate_b,
        beta=0.0,
    )

    assert inference is None


def test_liml_interacted_matches_just_identified_iv_without_exog():
    data = pd.DataFrame(
        {
            "Y": [1.0, 2.0, 2.5, 4.0, 4.5, 6.0],
            "W": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "Z": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "group": [0, 0, 0, 0, 0, 0],
        }
    )

    estimate = fit_liml_interacted(
        data,
        dependent="Y",
        endogenous="W",
        instrument="Z",
        exog=[],
        groups="group",
    )
    y = data["Y"].to_numpy(dtype=float)
    w = data["W"].to_numpy(dtype=float)
    z = data["Z"].to_numpy(dtype=float)
    design = np.ones((len(data), 1))
    y_resid = y - design @ (np.linalg.pinv(design) @ y)
    w_resid = w - design @ (np.linalg.pinv(design) @ w)
    z_resid = z - design @ (np.linalg.pinv(design) @ z)
    expected = float((z_resid @ y_resid) / (z_resid @ w_resid))

    assert estimate.method == "liml_interacted"
    assert np.isclose(estimate.beta, expected)

