import numpy as np
import pandas as pd
import pytest

from adaptiveiv.simulation import paper_group_strengths, simulate_paper_section4_dgp
from adaptiveiv.validation import (
    estimate_methods_once,
    summarize_simulation_results,
    validate_simulation_summary,
)
from adaptiveiv.paper_benchmarks import (
    compare_summary_to_paper_targets,
    paper_method_coverage,
    paper_table_targets,
    section4_paper_configurations,
)
from simulations.validate_paper_tables import selected_configurations
from simulations.validate_paper_tables import _repetitions as paper_table_repetitions
import simulations.validate_paper_tables as validate_paper_tables
import simulations.diagnose_pooled_tail_seeds as diagnose_pooled_tail_seeds
import simulations.diagnose_pooled_mse_targets as diagnose_pooled_mse_targets
import simulations.audit_pooled_rng_conventions as audit_pooled_rng_conventions
import simulations.audit_pooled_error_conventions as audit_pooled_error_conventions
import simulations.diagnose_pooled_tail_splice as diagnose_pooled_tail_splice
import simulations.reconstruct_paper_table_seeds as reconstruct_paper_table_seeds
from simulations.aggregate_paper_table_chunks import aggregate_paper_table_chunks
from adaptiveiv.estimators import fit_pooled_2sls


def test_paper_group_strengths_match_dgp1_counts():
    strengths = paper_group_strengths(
        "dgp1",
        n_groups=20,
        strong_fraction=0.25,
        seed=123,
    )

    assert len(strengths) == 20
    assert np.sum(strengths == 1.0) == 5
    assert np.sum(strengths == 0.0) == 15


def test_paper_group_strengths_match_dgp2_counts():
    strengths = paper_group_strengths(
        "dgp2",
        n_groups=40,
        strong_fraction=0.125,
        weak_fraction=0.125,
        weak_strength=0.2,
        seed=123,
    )

    assert np.sum(strengths == 1.0) == 5
    assert np.sum(strengths == 0.2) == 5
    assert np.sum(strengths == 0.0) == 30


def test_paper_group_strengths_match_dgp3_mixture_shape():
    strengths = paper_group_strengths("dgp3", n_groups=100, seed=123)

    assert np.sum(strengths == 0.0) == 90
    assert np.sum(strengths != 0.0) == 10
    assert 0.35 < strengths[strengths != 0.0].mean() < 0.85


def test_paper_group_strengths_dgp3_use_untruncated_normal_mixture():
    strengths = paper_group_strengths("dgp3", n_groups=40, seed=3)

    assert strengths.min() < 0.0


def test_paper_section4_dgp_is_reproducible_and_records_metadata():
    first = simulate_paper_section4_dgp(
        dgp="dgp2",
        n_groups=12,
        n_per_group=20,
        strong_fraction=0.25,
        weak_fraction=0.25,
        seed=2024,
    )
    second = simulate_paper_section4_dgp(
        dgp="dgp2",
        n_groups=12,
        n_per_group=20,
        strong_fraction=0.25,
        weak_fraction=0.25,
        seed=2024,
    )

    pd.testing.assert_frame_equal(first, second)
    assert {
        "Y",
        "W",
        "Z",
        "X",
        "u",
        "v",
        "rho_g",
        "group",
        "dgp",
        "error_distribution",
    }.issubset(first.columns)
    assert first["group"].nunique() == 12
    assert first.groupby("group").size().eq(20).all()
    assert first["dgp"].unique().tolist() == ["dgp2"]


def test_paper_section4_dgp_can_use_fixed_group_strengths():
    strengths = np.array([0.0, 0.2, 1.0])

    data = simulate_paper_section4_dgp(
        dgp="dgp3",
        n_groups=3,
        n_per_group=5,
        group_strengths=strengths,
        seed=123,
    )

    observed = (
        data[["group", "rho_g"]]
        .drop_duplicates()
        .sort_values("group")["rho_g"]
        .to_numpy()
    )
    np.testing.assert_allclose(observed, strengths)


def test_paper_section4_chisq_errors_are_centered_and_valid_iv():
    data = simulate_paper_section4_dgp(
        dgp="dgp1",
        n_groups=30,
        n_per_group=80,
        strong_fraction=0.2,
        error_distribution="chisq3",
        seed=99,
    )

    assert abs(float(data["u"].mean())) < 0.15
    assert abs(float(data["v"].mean())) < 0.15
    assert 5.0 < float(data["v"].var()) < 7.0
    assert abs(float(data["Z"].corr(data["u"]))) < 0.08


def test_estimate_methods_once_returns_all_validation_estimators():
    data = simulate_paper_section4_dgp(
        dgp="dgp1",
        n_groups=10,
        n_per_group=80,
        strong_fraction=0.4,
        seed=1234,
    )

    estimates = estimate_methods_once(data, random_state=44)

    assert set(estimates["method"]) == {
        "pooled",
        "fully_interacted",
        "split_interacted",
        "liml_interacted",
        "adaptive",
        "oracle",
    }
    assert estimates["beta_hat"].notna().all()
    assert estimates["nobs"].eq(len(data)).all()
    assert estimates.loc[estimates["method"] == "oracle", "selected_total"].iloc[0] == 4
    assert (
        estimates.loc[estimates["method"] == "adaptive", "threshold_selected_total"].iloc[0]
        <= 10
    )


def test_estimate_methods_once_can_request_liml_interacted_benchmark():
    data = simulate_paper_section4_dgp(
        dgp="dgp1",
        n_groups=8,
        n_per_group=80,
        strong_fraction=0.5,
        seed=1234,
    )

    estimates = estimate_methods_once(
        data,
        random_state=44,
        methods=["liml_interacted"],
    )

    assert estimates["method"].tolist() == ["liml_interacted"]
    assert estimates["finite"].tolist() == [True]
    assert np.isfinite(estimates["beta_hat"].iloc[0])


def test_estimate_methods_once_can_use_repeated_splits():
    data = simulate_paper_section4_dgp(
        dgp="dgp1",
        n_groups=10,
        n_per_group=80,
        strong_fraction=0.4,
        seed=1234,
    )

    estimates = estimate_methods_once(data, random_state=44, n_splits=3)
    adaptive = estimates.loc[estimates["method"] == "adaptive"].iloc[0]

    assert adaptive["n_splits"] == 3
    assert adaptive["finite_split_estimates"] >= 1


def test_summarize_simulation_results_computes_metrics_by_method():
    raw = pd.DataFrame(
        {
            "scenario": ["demo"] * 6,
            "dgp": ["dgp1"] * 6,
            "error_distribution": ["normal"] * 6,
            "n_groups": [40] * 6,
            "n_per_group": [500] * 6,
            "strong_fraction": [0.05] * 6,
            "weak_fraction": [0.0] * 6,
            "method": ["pooled", "pooled", "adaptive", "adaptive", "oracle", "oracle"],
            "beta_hat": [1.0, 3.0, 1.8, 2.2, 1.9, 2.1],
            "true_beta": [2.0] * 6,
            "nobs": [100] * 6,
            "selected_total": [10, 10, 4, 4, 4, 4],
            "threshold_selected_total": [10, 10, 4, 4, 4, 4],
        }
    )

    summary = summarize_simulation_results(raw)
    pooled = summary.loc[summary["method"] == "pooled"].iloc[0]
    adaptive = summary.loc[summary["method"] == "adaptive"].iloc[0]

    assert pooled["repetitions"] == 2
    assert pooled["finite_share"] == 1.0
    assert pooled["mse"] == 1.0
    assert pooled["scaled_mse"] == 100.0
    assert np.isclose(adaptive["mad"], 0.2)
    assert adaptive["mean_selected_total"] == 4.0
    assert adaptive["strong_fraction"] == 0.05
    assert adaptive["weak_fraction"] == 0.0


def test_summarize_simulation_results_reports_tail_error_diagnostics():
    raw = pd.DataFrame(
        {
            "scenario": ["demo"] * 3,
            "dgp": ["dgp1"] * 3,
            "error_distribution": ["chisq3"] * 3,
            "n_groups": [40] * 3,
            "n_per_group": [500] * 3,
            "strong_fraction": [0.05] * 3,
            "weak_fraction": [0.0] * 3,
            "method": ["pooled"] * 3,
            "beta_hat": [2.0, 3.0, 5.0],
            "true_beta": [2.0] * 3,
            "nobs": [20_000] * 3,
            "selected_total": [40] * 3,
            "threshold_selected_total": [40] * 3,
        }
    )

    summary = summarize_simulation_results(raw)
    row = summary.iloc[0]

    assert row["max_abs_error"] == 3.0
    assert np.isclose(row["q95_abs_error"], np.quantile([0.0, 1.0, 3.0], 0.95))
    assert np.isclose(row["q99_abs_error"], np.quantile([0.0, 1.0, 3.0], 0.99))
    assert row["max_scaled_sq_error"] == 180_000.0
    assert np.isclose(row["top_abs_error_mse_share"], 0.9)


def test_paper_table_targets_cover_implemented_section4_configs():
    targets = paper_table_targets()

    assert len(targets) == 330
    assert set(targets["method"]) == {
        "pooled",
        "fully_interacted",
        "split_interacted",
        "liml_interacted",
        "oracle",
        "adaptive",
    }
    assert set(targets["dgp"]) == {"dgp1", "dgp2", "dgp3"}
    assert set(targets["metric"]) == {"scaled_mse", "scaled_mad"}

    dgp1_target = targets.loc[
        (targets["dgp"] == "dgp1")
        & (targets["error_distribution"] == "normal")
        & (targets["n_groups"] == 40)
        & (targets["strong_fraction"] == 0.05)
        & (targets["method"] == "adaptive")
        & (targets["metric"] == "scaled_mse")
    ].iloc[0]
    assert dgp1_target["paper_value"] == 20.390

    dgp3_target = targets.loc[
        (targets["dgp"] == "dgp3")
        & (targets["error_distribution"] == "chisq3")
        & (targets["n_groups"] == 200)
        & (targets["method"] == "oracle")
        & (targets["metric"] == "scaled_mad")
    ].iloc[0]
    assert dgp3_target["paper_value"] == 2480.836

    liml_targets = targets.loc[targets["method"] == "liml_interacted"]
    assert len(liml_targets) == 30
    assert set(liml_targets["metric"]) == {"scaled_mad"}

    dgp1_liml = liml_targets.loc[
        (liml_targets["dgp"] == "dgp1")
        & (liml_targets["error_distribution"] == "normal")
        & (liml_targets["n_groups"] == 40)
        & (liml_targets["strong_fraction"] == 0.05)
    ].iloc[0]
    assert dgp1_liml["paper_value"] == 432.387

    dgp2_liml = liml_targets.loc[
        (liml_targets["dgp"] == "dgp2")
        & (liml_targets["error_distribution"] == "chisq3")
        & (liml_targets["n_groups"] == 100)
        & (liml_targets["strong_fraction"] == 0.025)
        & (liml_targets["weak_fraction"] == 0.025)
    ].iloc[0]
    assert dgp2_liml["paper_value"] == 2883.843

    dgp3_liml = liml_targets.loc[
        (liml_targets["dgp"] == "dgp3")
        & (liml_targets["error_distribution"] == "chisq3")
        & (liml_targets["n_groups"] == 200)
    ].iloc[0]
    assert dgp3_liml["paper_value"] == 2604.448

def test_paper_method_coverage_marks_unimplemented_original_paper_methods():
    coverage = paper_method_coverage()

    assert coverage["paper_method"].tolist() == [
        "2SLS-P",
        "2SLS-INT",
        "2SLS-SSINT",
        "2SLS-INF",
        "2SLS-ADPT",
        "LIML-INT",
        "2SLS-SSL",
        "UJIVE",
        "IJIVE",
    ]
    implemented = coverage.loc[coverage["implemented"], "paper_method"].tolist()
    missing = coverage.loc[~coverage["implemented"], "paper_method"].tolist()
    missing_targets = coverage.loc[
        coverage["implemented"] & ~coverage["paper_table_targets_transcribed"],
        "paper_method",
    ].tolist()

    assert implemented == [
        "2SLS-P",
        "2SLS-INT",
        "2SLS-SSINT",
        "2SLS-INF",
        "2SLS-ADPT",
        "LIML-INT",
    ]
    assert missing == ["2SLS-SSL", "UJIVE", "IJIVE"]
    assert missing_targets == []


def test_compare_summary_to_paper_targets_reports_relative_errors():
    summary = pd.DataFrame(
        {
            "scenario": ["demo"],
            "config_index": [0],
            "dgp": ["dgp1"],
            "error_distribution": ["normal"],
            "n_groups": [40],
            "n_per_group": [500],
            "strong_fraction": [0.05],
            "weak_fraction": [0.0],
            "method": ["adaptive"],
            "scaled_mse": [22.429],
            "mad": [431.0 / (40 * 500)],
        }
    )

    comparison = compare_summary_to_paper_targets(summary)

    mse = comparison.loc[comparison["metric"] == "scaled_mse"].iloc[0]
    mad = comparison.loc[comparison["metric"] == "scaled_mad"].iloc[0]
    assert mse["config_index"] == 0
    assert mse["paper_value"] == 20.390
    assert np.isclose(mse["relative_error"], 0.1)
    assert mad["paper_value"] == 430.740
    assert np.isclose(mad["observed_value"], 431.0)


def test_compare_summary_to_paper_targets_preserves_tail_diagnostics():
    summary = pd.DataFrame(
        {
            "scenario": ["demo"],
            "config_index": [0],
            "dgp": ["dgp1"],
            "error_distribution": ["normal"],
            "n_groups": [40],
            "n_per_group": [500],
            "strong_fraction": [0.05],
            "weak_fraction": [0.0],
            "method": ["adaptive"],
            "scaled_mse": [22.429],
            "mad": [431.0 / (40 * 500)],
            "max_abs_error": [2.5],
            "q95_abs_error": [1.5],
            "q99_abs_error": [2.1],
            "max_scaled_sq_error": [125_000.0],
            "top_abs_error_mse_share": [0.75],
        }
    )

    comparison = compare_summary_to_paper_targets(summary)
    row = comparison.loc[comparison["metric"] == "scaled_mse"].iloc[0]

    assert row["max_abs_error"] == 2.5
    assert row["q95_abs_error"] == 1.5
    assert row["q99_abs_error"] == 2.1
    assert row["max_scaled_sq_error"] == 125_000.0
    assert row["top_abs_error_mse_share"] == 0.75


def test_compare_summary_to_paper_targets_reports_implied_paper_tail_event():
    summary = pd.DataFrame(
        {
            "scenario": ["demo"],
            "config_index": [18],
            "dgp": ["dgp2"],
            "error_distribution": ["chisq3"],
            "n_groups": [40],
            "n_per_group": [500],
            "strong_fraction": [0.025],
            "weak_fraction": [0.025],
            "method": ["pooled"],
            "repetitions": [500],
            "scaled_mse": [486_588.466424],
            "mad": [8_722.373 / (40 * 500)],
            "max_abs_error": [98.665825],
            "max_scaled_sq_error": [194_698_901.936],
        }
    )

    comparison = compare_summary_to_paper_targets(summary)
    row = comparison.loc[comparison["metric"] == "scaled_mse"].iloc[0]
    observed_total = row["observed_value"] * summary["repetitions"].iloc[0]
    paper_total = row["paper_value"] * summary["repetitions"].iloc[0]
    expected_scaled_sq = (
        paper_total - (observed_total - summary["max_scaled_sq_error"].iloc[0])
    )

    assert np.isclose(row["paper_implied_single_tail_abs_error"], 124.799585)
    assert np.isclose(
        row["paper_implied_single_tail_scaled_sq_error"],
        expected_scaled_sq,
    )
    assert row["paper_implied_tail_to_observed_max_ratio"] > 1.0


def test_paper_table_largest_deviations_show_tail_diagnostics_when_available():
    comparison = pd.DataFrame(
        {
            "source_table": ["Table 3"],
            "dgp": ["dgp2"],
            "error_distribution": ["chisq3"],
            "n_groups": [100],
            "strong_fraction": [0.025],
            "weak_fraction": [0.025],
            "method": ["pooled"],
            "metric": ["scaled_mse"],
            "observed_value": [1.0],
            "paper_value": [100.0],
            "relative_error": [-0.99],
            "max_abs_error": [70.0],
            "top_abs_error_mse_share": [0.98],
            "paper_implied_single_tail_abs_error": [900.0],
        }
    )

    display = validate_paper_tables._largest_deviations(comparison)

    assert "max_abs_error" in display.columns
    assert "top_abs_error_mse_share" in display.columns
    assert "paper_implied_single_tail_abs_error" in display.columns


def test_pooled_tail_seed_diagnostic_matches_pooled_estimator():
    config = {**section4_paper_configurations()[19], "config_index": 19}
    seed = 22160992
    data = simulate_paper_section4_dgp(
        dgp=config["dgp"],
        n_groups=config["n_groups"],
        n_per_group=config["n_per_group"],
        strong_fraction=config["strong_fraction"],
        weak_fraction=config["weak_fraction"],
        error_distribution=config["error_distribution"],
        seed=seed,
    )

    diagnostic = diagnose_pooled_tail_seeds.pooled_tail_seed_diagnostic(
        config,
        seed,
    )
    pooled = fit_pooled_2sls(data, "Y", "W", "Z", ["X"])

    assert np.isclose(diagnostic["beta_hat"], pooled.beta)
    assert np.isclose(diagnostic["abs_beta_hat"], abs(pooled.beta))
    assert diagnostic["seed"] == seed
    assert diagnostic["config_index"] == 19


def test_pooled_tail_seed_scan_reports_top_seeds_and_threshold_counts():
    config = {**section4_paper_configurations()[19], "config_index": 19}
    seeds = [22160992, 3317033614, 459030378]

    scan, counts = diagnose_pooled_tail_seeds.scan_pooled_tail_seeds(
        config,
        seeds,
        top_k=2,
        thresholds=[100.0, 1000.0],
    )

    assert scan["seed"].tolist() == [459030378, 3317033614]
    assert scan["abs_beta_hat"].iloc[0] > scan["abs_beta_hat"].iloc[1]
    assert counts["abs_beta_gt_100"] == 2
    assert counts["abs_beta_gt_1000"] == 1
    assert counts["seed_count"] == 3


def test_pooled_tail_seed_report_renders_without_optional_tabulate():
    config = {**section4_paper_configurations()[19], "config_index": 19}
    scan = pd.DataFrame(
        {
            "seed": [459030378],
            "beta_hat": [-2009.7],
            "abs_beta_hat": [2009.7],
            "pooled_denominator": [0.59],
        }
    )
    counts = {"seed_count": 1, "max_abs_beta_hat": 2009.7}

    report = diagnose_pooled_tail_seeds.render_report(config, scan, counts)

    assert "adaptiveiv Pooled Tail Seed Diagnostic" in report
    assert "| seed | beta_hat | abs_beta_hat | pooled_denominator |" in report
    assert "459030378" in report
    assert "4.5903e+08" not in report


def test_pooled_mse_target_diagnostic_classifies_failed_rows():
    comparison = pd.DataFrame(
        {
            "source_table": ["Table 2", "Table 3", "Table 3"],
            "config_index": [7, 18, 19],
            "dgp": ["dgp1", "dgp2", "dgp2"],
            "error_distribution": ["chisq3", "chisq3", "chisq3"],
            "n_groups": [100, 40, 100],
            "n_per_group": [500, 500, 500],
            "strong_fraction": [0.05, 0.025, 0.025],
            "weak_fraction": [0.0, 0.025, 0.025],
            "method": ["pooled", "pooled", "pooled"],
            "metric": ["scaled_mse", "scaled_mse", "scaled_mse"],
            "observed_value": [3682.348493, 486588.466424, 59188.887147],
            "paper_value": [2791.348, 720188.116, 98979469.114],
            "relative_error": [0.319201, -0.324359, -0.999402],
            "paper_implied_single_tail_abs_error": [np.nan, 124.799585, 994.648995],
        }
    )

    targets = diagnose_pooled_mse_targets.failed_pooled_mse_targets(
        comparison,
        relative_tolerance=0.25,
    )

    assert targets["config_index"].tolist() == [7, 18, 19]
    assert targets["target_status"].tolist() == [
        "observed_exceeds_paper",
        "missing_larger_tail",
        "missing_larger_tail",
    ]
    assert np.isnan(targets.loc[targets["config_index"] == 7, "target_abs_beta"].iloc[0])
    assert np.isclose(
        targets.loc[targets["config_index"] == 19, "target_abs_beta"].iloc[0],
        994.648995,
    )


def test_pooled_mse_target_seed_scan_ranks_candidates_by_target_distance():
    targets = pd.DataFrame(
        {
            "config_index": [19],
            "source_table": ["Table 3"],
            "dgp": ["dgp2"],
            "error_distribution": ["chisq3"],
            "n_groups": [100],
            "n_per_group": [500],
            "strong_fraction": [0.025],
            "weak_fraction": [0.025],
            "target_status": ["missing_larger_tail"],
            "target_abs_beta": [994.648995],
        }
    )
    seeds = {19: [459030378, 2612616469, 22160992]}

    candidates = diagnose_pooled_mse_targets.scan_target_seed_candidates(
        targets,
        seeds_by_config=seeds,
        top_k=2,
    )

    assert candidates["seed"].iloc[0] == 2612616469
    assert candidates["target_abs_beta"].eq(994.648995).all()
    assert candidates["target_distance"].iloc[0] < candidates["target_distance"].iloc[1]
    assert candidates["target_distance"].is_monotonic_increasing
    assert "beta_hat" in candidates.columns


def test_pooled_rng_convention_audit_reproduces_current_convention():
    config = {**section4_paper_configurations()[19], "config_index": 19}
    seed = 22160992

    audit_row = audit_pooled_rng_conventions.pooled_convention_diagnostic(
        config,
        seed,
        convention="shuffled_separate",
    )
    tail_row = diagnose_pooled_tail_seeds.pooled_tail_seed_diagnostic(config, seed)

    assert np.isclose(audit_row["beta_hat"], tail_row["beta_hat"])
    assert np.isclose(audit_row["pooled_denominator"], tail_row["pooled_denominator"])
    assert audit_row["convention"] == "shuffled_separate"


def test_pooled_rng_convention_report_renders():
    results = pd.DataFrame(
        {
            "config_index": [19],
            "convention": ["shuffled_separate"],
            "scaled_mse": [59188.9],
            "paper_value": [98_979_469.114],
            "relative_error": [-0.999402],
            "max_abs_beta_hat": [11.127],
            "max_abs_beta_seed": [22160992],
        }
    )

    report = audit_pooled_rng_conventions.render_report(results)

    assert "adaptiveiv Pooled RNG Convention Audit" in report
    assert "| config_index | convention | scaled_mse |" in report
    assert "shuffled_separate" in report


def test_pooled_error_convention_audit_ranks_conventions():
    config = {**section4_paper_configurations()[7], "config_index": 7}

    results = audit_pooled_error_conventions.audit_pooled_error_conventions(
        config,
        seed_base=20260623,
        repetitions=3,
        conventions=("centered", "standardized"),
    )

    assert set(results["error_convention"]) == {"centered", "standardized"}
    assert {"scaled_mse", "paper_value", "relative_error", "rank"}.issubset(
        results.columns
    )
    assert results["rank"].tolist() == [1, 2]
    assert (
        results.loc[
            results["error_convention"].eq("centered"),
            "scaled_mse",
        ].iloc[0]
        > results.loc[
            results["error_convention"].eq("standardized"),
            "scaled_mse",
        ].iloc[0]
    )


def test_pooled_error_convention_report_renders():
    results = pd.DataFrame(
        {
            "config_index": [7],
            "error_convention": ["centered"],
            "scaled_mse": [3682.35],
            "paper_value": [2791.348],
            "relative_error": [0.319201],
            "rank": [1],
        }
    )

    report = audit_pooled_error_conventions.render_report(results)

    assert "adaptiveiv Pooled Error Convention Audit" in report
    assert "| config_index | error_convention | scaled_mse |" in report
    assert "chi-square" in report


def test_pooled_tail_splice_ranks_candidates_by_spliced_paper_error():
    comparison = pd.DataFrame(
        {
            "config_index": [19],
            "source_table": ["Table 3"],
            "dgp": ["dgp2"],
            "error_distribution": ["chisq3"],
            "n_groups": [100],
            "n_per_group": [500],
            "method": ["pooled"],
            "metric": ["scaled_mse"],
            "observed_value": [59_188.887147],
            "paper_value": [98_979_469.114],
            "relative_error": [-0.999402],
            "max_scaled_sq_error": [6_191_060.0],
            "nobs": [50_000.0],
            "repetitions": [500],
        }
    )
    candidates = pd.DataFrame(
        {
            "config_index": [19, 19],
            "seed": [459030378, 2612616469],
            "abs_beta_hat": [2009.722287, 879.397117],
            "beta_hat": [-2009.722287, -879.397117],
        }
    )

    splice = diagnose_pooled_tail_splice.tail_splice_candidates(
        comparison,
        candidates,
        relative_tolerance=0.25,
    )

    assert splice["seed"].iloc[0] == 2612616469
    assert abs(splice["spliced_relative_error"].iloc[0]) <= 0.25
    assert splice["spliced_abs_relative_error"].is_monotonic_increasing
    assert splice.loc[splice["seed"] == 459030378, "spliced_relative_error"].iloc[0] > 0


def test_pooled_tail_splice_report_renders():
    splice = pd.DataFrame(
        {
            "config_index": [19],
            "seed": [2612616469],
            "spliced_scaled_mse": [77_380_740.0],
            "paper_value": [98_979_469.114],
            "spliced_relative_error": [-0.218214],
            "spliced_within_tolerance": [True],
        }
    )

    report = diagnose_pooled_tail_splice.render_report(splice)

    assert "adaptiveiv Pooled Tail Splice Diagnostic" in report
    assert "counterfactual" in report
    assert "| config_index | seed | spliced_scaled_mse |" in report


def test_seed_reconstruction_parses_replacement_map():
    replacements = reconstruct_paper_table_seeds.parse_replacement_map(
        ["7:229:12345", "19:369:2612616469"]
    )

    assert replacements[0].config_index == 7
    assert replacements[0].replication == 229
    assert replacements[0].seed == 12345
    assert replacements[1].config_index == 19
    assert replacements[1].replication == 369
    assert replacements[1].seed == 2612616469


def test_seed_reconstruction_replaces_full_replication_blocks():
    existing = pd.DataFrame(
        {
            "config_index": [7, 7, 7, 8],
            "replication": [229, 229, 230, 229],
            "method": ["pooled", "adaptive", "pooled", "pooled"],
            "beta_hat": [1.0, 2.0, 3.0, 4.0],
        }
    )
    replacement_rows = pd.DataFrame(
        {
            "config_index": [7, 7],
            "replication": [229, 229],
            "method": ["pooled", "adaptive"],
            "beta_hat": [0.1, 0.2],
        }
    )

    replaced = reconstruct_paper_table_seeds.replace_simulation_rows(
        existing,
        replacement_rows,
    )

    block = replaced.loc[
        replaced["config_index"].eq(7) & replaced["replication"].eq(229)
    ].sort_values("method")
    assert block["beta_hat"].tolist() == [0.2, 0.1]
    assert set(replaced.loc[replaced["config_index"].eq(8), "beta_hat"]) == {4.0}
    assert set(replaced.loc[replaced["replication"].eq(230), "beta_hat"]) == {3.0}


def test_paper_comparison_checks_flag_incomplete_target_matches():
    comparison = pd.DataFrame(
        {
            "metric": ["scaled_mse"],
            "relative_error": [0.0],
        }
    )

    checks = validate_paper_tables.paper_comparison_checks(
        comparison,
        relative_tolerance=0.25,
        expected_matches=2,
    )

    target_check = checks.loc[checks["check"] == "paper_targets_matched"].iloc[0]
    assert not target_check["passed"]
    assert "matched rows=1; expected=2" in target_check["detail"]


def test_section4_paper_configurations_are_unique_and_complete():
    configs = section4_paper_configurations()

    assert len(configs) == 30
    assert all(config["n_per_group"] == 500 for config in configs)
    assert len({tuple(sorted(config.items())) for config in configs}) == 30
    assert sum(config["dgp"] == "dgp3" for config in configs) == 6


def test_paper_table_configuration_chunks_keep_original_indices():
    args = _paper_table_args(config_start=3, config_stop=6)

    configs = selected_configurations(args)

    assert len(configs) == 3
    assert [config["config_index"] for config in configs] == [3, 4, 5]
    assert [config["n_groups"] for config in configs] == [40, 100, 200]


def test_paper_table_configuration_chunks_reject_invalid_ranges():
    args = _paper_table_args(config_start=6, config_stop=3)

    with pytest.raises(ValueError, match="config-stop"):
        selected_configurations(args)


def test_paper_table_chunk_seeds_use_original_config_index(monkeypatch):
    observed_seeds = []

    def fake_simulate_paper_section4_dgp(**kwargs):
        observed_seeds.append(kwargs["seed"])
        return pd.DataFrame({"group": [0], "beta": [0.0]})

    def fake_estimate_methods_once(data, **kwargs):
        return pd.DataFrame(
            [
                {
                    "method": "adaptive",
                    "beta_hat": 0.0,
                    "true_beta": 0.0,
                    "nobs": 1,
                    "ngroups": 1,
                    "selected_total": 1,
                    "threshold_selected_total": 1,
                    "n_splits": kwargs["n_splits"],
                    "finite_split_estimates": 1,
                    "finite": True,
                    "error": "",
                }
            ]
        )

    monkeypatch.setattr(
        validate_paper_tables,
        "simulate_paper_section4_dgp",
        fake_simulate_paper_section4_dgp,
    )
    monkeypatch.setattr(
        validate_paper_tables,
        "estimate_methods_once",
        fake_estimate_methods_once,
    )
    args = _paper_table_args(
        seed=11,
        config_start=5,
        config_stop=6,
        repetitions=2,
        n_splits=1,
    )
    configs = selected_configurations(args)

    results = validate_paper_tables.run_paper_table_validation(args, configs)

    assert observed_seeds == [500011, 500012]
    assert results["seed"].tolist() == [500011, 500012]


def test_paper_table_validation_reuses_fixed_dgp3_strengths(monkeypatch):
    observed_strengths = []

    def fake_simulate_paper_section4_dgp(**kwargs):
        observed_strengths.append(tuple(kwargs["group_strengths"]))
        return pd.DataFrame({"group": [0], "beta": [0.0]})

    def fake_estimate_methods_once(data, **kwargs):
        return pd.DataFrame(
            [
                {
                    "method": "adaptive",
                    "beta_hat": 0.0,
                    "true_beta": 0.0,
                    "nobs": 1,
                    "ngroups": 1,
                    "selected_total": 1,
                    "threshold_selected_total": 1,
                    "n_splits": kwargs["n_splits"],
                    "finite_split_estimates": 1,
                    "finite": True,
                    "error": "",
                }
            ]
        )

    monkeypatch.setattr(
        validate_paper_tables,
        "simulate_paper_section4_dgp",
        fake_simulate_paper_section4_dgp,
    )
    monkeypatch.setattr(
        validate_paper_tables,
        "estimate_methods_once",
        fake_estimate_methods_once,
    )
    args = _paper_table_args(
        seed=11,
        config_start=24,
        config_stop=25,
        repetitions=2,
        n_splits=1,
    )
    configs = selected_configurations(args)

    results = validate_paper_tables.run_paper_table_validation(args, configs)

    assert len(observed_strengths) == 2
    assert observed_strengths[0] == observed_strengths[1]
    assert results["dgp3_strength_mode"].tolist() == ["fixed", "fixed"]
    assert results["dgp3_strength_seed"].tolist() == [2_400_011, 2_400_011]
    assert "dgp3_strength_sum_squares" in results.columns
    assert "dgp3_strength_vector" in results.columns
    assert np.isclose(
        results["dgp3_strength_sum_squares"].iloc[0],
        np.sum(np.asarray(observed_strengths[0]) ** 2),
    )
    parsed = validate_paper_tables.parse_strength_vector(
        results["dgp3_strength_vector"].iloc[0]
    )
    np.testing.assert_allclose(parsed, np.asarray(observed_strengths[0]))


def test_paper_table_config_manifest_records_dgp3_strength_state():
    args = _paper_table_args(
        seed=11,
        config_start=24,
        config_stop=25,
        n_splits=1,
    )
    configs = selected_configurations(args)

    manifest = validate_paper_tables.paper_table_config_manifest(args, configs)

    row = manifest.iloc[0]
    assert row["dgp"] == "dgp3"
    assert row["dgp3_strength_mode"] == "fixed"
    assert row["dgp3_strength_seed"] == 2_400_011
    assert row["dgp3_strength_nonzero_count"] == 4
    parsed = validate_paper_tables.parse_strength_vector(row["dgp3_strength_vector"])
    assert len(parsed) == 40
    assert np.isclose(row["dgp3_strength_sum_squares"], float(parsed @ parsed))


def test_paper_table_dgp3_strength_seed_base_is_independent_of_data_seed():
    args = _paper_table_args(
        seed=11,
        dgp3_strength_seed_base=777,
        config_start=24,
        config_stop=25,
        n_splits=1,
    )
    configs = selected_configurations(args)

    manifest = validate_paper_tables.paper_table_config_manifest(args, configs)

    assert manifest["dgp3_strength_seed"].iloc[0] == 2_400_777


def test_paper_table_dgp3_exact_strength_seed_overrides_seed_base():
    args = _paper_table_args(
        seed=11,
        dgp3_strength_seed_base=777,
        dgp3_strength_seed=11890,
        config_start=24,
        config_stop=25,
        n_splits=1,
    )
    configs = selected_configurations(args)

    manifest = validate_paper_tables.paper_table_config_manifest(args, configs)

    assert manifest["dgp3_strength_seed"].iloc[0] == 11_890


def test_paper_table_dgp3_strength_seed_map_overrides_single_seed():
    args = _paper_table_args(
        seed=11,
        dgp3_strength_seed=11890,
        dgp3_strength_seed_map="24=11890,26=45610",
        only_dgp="dgp3",
        config_start=0,
        config_stop=3,
        n_splits=1,
    )
    configs = selected_configurations(args)

    manifest = validate_paper_tables.paper_table_config_manifest(args, configs)

    assert manifest["config_index"].tolist() == [24, 25, 26]
    assert manifest["dgp3_strength_seed"].tolist() == [11_890, 11_890, 45_610]


def test_paper_table_report_records_dgp3_strength_controls():
    args = _paper_table_args(
        dgp3_strength_seed_base=777,
        dgp3_strength_seed=11890,
        dgp3_strength_seed_map="24=11890,26=45610",
        redraw_dgp3_strengths=True,
    )

    report = validate_paper_tables.render_report(
        args,
        configs=[],
        comparison=pd.DataFrame(),
        checks=pd.DataFrame({"check": ["paper_targets_matched"], "passed": [True]}),
    )

    assert "- DGP3 strength mode: redraw per replication" in report
    assert "- DGP3 strength seed base: 777" in report
    assert "- DGP3 exact strength seed: 11890" in report
    assert "- DGP3 strength seed map: 24=11890,26=45610" in report


def test_paper_table_release_preset_uses_stable_repetition_count():
    args = _paper_table_args(preset="release")

    assert paper_table_repetitions(args) == 100


def test_aggregate_paper_table_chunks_writes_combined_outputs(tmp_path):
    chunk_a = tmp_path / "chunk_a"
    chunk_b = tmp_path / "chunk_b"
    output_dir = tmp_path / "combined"
    _write_fake_paper_chunk(chunk_a, [0])
    _write_fake_paper_chunk(chunk_b, [1])

    aggregate_paper_table_chunks(
        [chunk_a, chunk_b],
        output_dir,
        expected_config_count=2,
        relative_tolerance=10.0,
    )

    manifest = pd.read_csv(output_dir / "config_manifest.csv")
    results = pd.read_csv(output_dir / "simulation_results.csv")
    comparison = pd.read_csv(output_dir / "paper_comparison.csv")
    checks = pd.read_csv(output_dir / "checks.csv")
    report = (output_dir / "report.md").read_text()

    assert sorted(manifest["config_index"].tolist()) == [0, 1]
    assert sorted(results["config_index"].unique().tolist()) == [0, 1]
    assert sorted(comparison["config_index"].unique().tolist()) == [0, 1]
    assert checks.loc[checks["check"] == "config_coverage", "passed"].iloc[0]
    assert "# adaptiveiv Combined Paper Table Validation Report" in report


def test_aggregate_paper_table_chunks_flags_missing_indices(tmp_path):
    chunk = tmp_path / "chunk"
    output_dir = tmp_path / "combined"
    _write_fake_paper_chunk(chunk, [0])

    aggregate_paper_table_chunks(
        [chunk],
        output_dir,
        expected_config_count=2,
        relative_tolerance=10.0,
    )

    checks = pd.read_csv(output_dir / "checks.csv")
    coverage = checks.loc[checks["check"] == "config_coverage"].iloc[0]
    assert not coverage["passed"]
    assert "missing=[1]" in coverage["detail"]


def _paper_table_args(**overrides):
    values = {
        "preset": "release",
        "max_configs": None,
        "only_table": None,
        "only_dgp": None,
        "only_error": None,
        "config_start": 0,
        "config_stop": None,
        "repetitions": None,
        "n_splits": 1,
        "seed": 20260623,
        "relative_tolerance": None,
        "redraw_dgp3_strengths": False,
        "dgp3_strength_seed_base": None,
        "dgp3_strength_seed": None,
        "dgp3_strength_seed_map": None,
    }
    values.update(overrides)
    return type("Args", (), values)()


def _write_fake_paper_chunk(path, config_indices):
    path.mkdir()
    configs = section4_paper_configurations()
    manifest_rows = []
    result_rows = []
    for config_index in config_indices:
        config = configs[config_index]
        manifest_rows.append({**config, "config_index": config_index})
        nobs = int(config["n_groups"]) * int(config["n_per_group"])
        for replication in range(2):
            for method in [
                "pooled",
                "fully_interacted",
                "split_interacted",
                "oracle",
                "adaptive",
            ]:
                result_rows.append(
                    {
                        "scenario": f"config_{config_index}",
                        "source_table": config["source_table"],
                        "config_index": config_index,
                        "replication": replication,
                        "method": method,
                        "beta_hat": 0.0,
                        "true_beta": 0.0,
                        "nobs": nobs,
                        "ngroups": config["n_groups"],
                        "selected_total": 1,
                        "threshold_selected_total": 1,
                        "finite": True,
                        "error": "",
                        "dgp": config["dgp"],
                        "error_distribution": config["error_distribution"],
                        "n_groups": config["n_groups"],
                        "n_per_group": config["n_per_group"],
                        "strong_fraction": config["strong_fraction"],
                        "weak_fraction": config["weak_fraction"],
                        "seed": 1000 + replication,
                    }
                )
    pd.DataFrame(manifest_rows).to_csv(path / "config_manifest.csv", index=False)
    pd.DataFrame(result_rows).to_csv(path / "simulation_results.csv", index=False)


def test_validate_simulation_summary_reports_qualitative_checks():
    summary = pd.DataFrame(
        {
            "scenario": ["demo"] * 4,
            "method": ["pooled", "split_interacted", "adaptive", "oracle"],
            "scaled_mse": [100.0, 40.0, 22.0, 20.0],
            "mean_selected_total": [10.0, 10.0, 4.0, 4.0],
            "finite_share": [1.0, 1.0, 1.0, 1.0],
        }
    )

    checks = validate_simulation_summary(summary)

    assert checks["passed"].all()
    assert {
        "adaptive_improves_over_pooled",
        "adaptive_close_to_oracle",
        "adaptive_selects_no_more_than_split_interacted",
    }.issubset(set(checks["check"]))
