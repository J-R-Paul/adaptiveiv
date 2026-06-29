import numpy as np
import pandas as pd
import statsmodels.api as sm

from adaptiveiv import AdaptiveIV
from adaptiveiv.estimators import fit_fully_interacted_2sls, fit_pooled_2sls
from adaptiveiv.simulation import simulate_paper_dgp


def test_paper_style_dgp_uses_valid_instrument_and_adaptive_recovers_beta():
    data = simulate_paper_dgp(
        n_groups=20,
        n_per_group=150,
        beta=0.5,
        strong_fraction=0.3,
        weak_fraction=0.2,
        rho_uv=0.4,
        seed=321,
    )

    assert abs(data["Z"].corr(data["u"])) < 0.05

    results = AdaptiveIV.from_formula(
        "Y ~ 1 + X + [W ~ Z]",
        data=data,
        groups="group",
    ).fit(random_state=99)

    assert abs(results.params["W"] - 0.5) < 0.25
    assert results.selection_summary["selected_total"] >= 1


def test_benchmark_estimators_return_finite_comparison_estimates(valid_iv_data):
    pooled = fit_pooled_2sls(
        valid_iv_data,
        dependent="Y",
        endogenous="W",
        instrument="Z",
        exog=["X1", "X2"],
    )
    interacted = fit_fully_interacted_2sls(
        valid_iv_data,
        dependent="Y",
        endogenous="W",
        instrument="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    assert np.isfinite(pooled.beta)
    assert np.isfinite(interacted.beta)
    assert pooled.method == "pooled"
    assert interacted.method == "fully_interacted"


def test_fully_interacted_estimator_matches_explicit_statsmodels_2sls(valid_iv_data):
    estimate = fit_fully_interacted_2sls(
        valid_iv_data,
        dependent="Y",
        endogenous="W",
        instrument="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    groups = pd.get_dummies(valid_iv_data["group"], prefix="g", dtype=float)
    exog_parts = [groups]
    for column in ["X1", "X2"]:
        exog_parts.append(groups.mul(valid_iv_data[column].to_numpy(), axis=0))
    exog = pd.concat(exog_parts, axis=1)
    instruments = groups.mul(valid_iv_data["Z"].to_numpy(), axis=0)
    first_stage = sm.OLS(valid_iv_data["W"], pd.concat([exog, instruments], axis=1)).fit()
    second_stage_x = exog.copy()
    second_stage_x["W_hat"] = first_stage.fittedvalues
    second_stage = sm.OLS(valid_iv_data["Y"], second_stage_x).fit()

    assert np.isclose(estimate.beta, second_stage.params["W_hat"])
