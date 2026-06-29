import numpy as np
import pytest

from adaptiveiv import AdaptiveIV
from adaptiveiv.formula import parse_iv_formula


def test_formula_parser_extracts_linearmodels_style_iv_block():
    spec = parse_iv_formula("Y ~ 1 + X1 + X2 + [W ~ Z]")

    assert spec.dependent == "Y"
    assert spec.exog == ["X1", "X2"]
    assert spec.endogenous == "W"
    assert spec.instrument == "Z"
    assert spec.add_intercept is True


def test_formula_api_matches_direct_api(valid_iv_data):
    formula_model = AdaptiveIV.from_formula(
        "Y ~ 1 + X1 + X2 + [W ~ Z]",
        data=valid_iv_data,
        groups="group",
    )
    direct_model = AdaptiveIV(
        data=valid_iv_data,
        dependent="Y",
        endogenous="W",
        instruments="Z",
        exog=["X1", "X2"],
        groups="group",
    )

    formula_results = formula_model.fit(random_state=123)
    direct_results = direct_model.fit(random_state=123)

    assert formula_model.dependent == "Y"
    assert formula_model.endogenous == "W"
    assert formula_model.instruments == ["Z"]
    assert formula_model.exog == ["X1", "X2"]
    assert np.isclose(formula_results.params["W"], direct_results.params["W"])


def test_formula_api_rejects_missing_iv_block(valid_iv_data):
    with pytest.raises(ValueError, match="exactly one"):
        AdaptiveIV.from_formula("Y ~ 1 + X1 + W", data=valid_iv_data, groups="group")


def test_formula_api_rejects_multiple_instruments(valid_iv_data):
    with pytest.raises(ValueError, match="one excluded instrument"):
        AdaptiveIV.from_formula(
            "Y ~ 1 + X1 + [W ~ Z + X2]",
            data=valid_iv_data,
            groups="group",
        )


def test_formula_api_supports_quoted_names(valid_iv_data):
    data = valid_iv_data.rename(
        columns={
            "Y": "outcome var",
            "W": "treatment-var",
            "Z": "instrument var",
            "X1": "control one",
        }
    )

    model = AdaptiveIV.from_formula(
        'Q("outcome var") ~ 1 + Q("control one") + X2 + '
        '[Q("treatment-var") ~ Q("instrument var")]',
        data=data,
        groups="group",
    )
    results = model.fit(random_state=123)

    assert model.dependent == "outcome var"
    assert model.endogenous == "treatment-var"
    assert model.instrument == "instrument var"
    assert model.exog == ["control one", "X2"]
    assert np.isfinite(results.params["treatment-var"])


def test_formula_api_rejects_no_intercept_formulas(valid_iv_data):
    with pytest.raises(ValueError, match="intercept"):
        AdaptiveIV.from_formula("Y ~ 0 + X1 + [W ~ Z]", data=valid_iv_data, groups="group")


def test_formula_api_rejects_minus_one_no_intercept_formulas(valid_iv_data):
    with pytest.raises(ValueError, match="intercept"):
        AdaptiveIV.from_formula("Y ~ X1 - 1 + [W ~ Z]", data=valid_iv_data, groups="group")
