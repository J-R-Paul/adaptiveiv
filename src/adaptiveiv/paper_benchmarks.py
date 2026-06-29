from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

IMPLEMENTED_PAPER_METHODS: tuple[str, ...] = (
    "pooled",
    "fully_interacted",
    "split_interacted",
    "oracle",
    "adaptive",
)

TRANSCRIBED_PAPER_METHODS: tuple[str, ...] = (
    *IMPLEMENTED_PAPER_METHODS,
    "liml_interacted",
)

FULL_SECTION4_PAPER_METHOD_LABELS: tuple[str, ...] = (
    "2SLS-P",
    "2SLS-INT",
    "2SLS-SSINT",
    "2SLS-INF",
    "2SLS-ADPT",
    "LIML-INT",
    "2SLS-SSL",
    "UJIVE",
    "IJIVE",
)

PAPER_METHOD_LABELS: dict[str, str] = {
    "pooled": "2SLS-P",
    "fully_interacted": "2SLS-INT",
    "split_interacted": "2SLS-SSINT",
    "oracle": "2SLS-INF",
    "adaptive": "2SLS-ADPT",
    "liml_interacted": "LIML-INT",
}

_PACKAGE_METHOD_BY_PAPER_LABEL: dict[str, str] = {
    label: method for method, label in PAPER_METHOD_LABELS.items()
}

_TAIL_DIAGNOSTIC_COLUMNS: tuple[str, ...] = (
    "max_abs_error",
    "q95_abs_error",
    "q99_abs_error",
    "max_scaled_sq_error",
    "top_abs_error_mse_share",
)

_PAPER_IMPLIED_TAIL_COLUMNS: tuple[str, ...] = (
    "paper_implied_single_tail_abs_error",
    "paper_implied_single_tail_scaled_sq_error",
    "paper_implied_tail_to_observed_max_ratio",
    "paper_implied_single_tail_mse_share",
)


def paper_table_targets() -> pd.DataFrame:
    """Return transcribed Section 4 targets for implemented package methods.

    The table contains the subset of Abadie, Gu, and Shen Tables 2-4 that is
    directly comparable to estimators implemented in this package. Values are
    transcribed from the published paper's reported ``N x MSE`` and ``N x MAD``
    entries; rejection rates are intentionally excluded because only the first
    adaptive homoskedastic inference slice is currently validated.
    """
    rows: list[dict[str, Any]] = []
    _extend_table2(rows)
    _extend_table3(rows)
    _extend_table4(rows)
    return pd.DataFrame(rows)


def paper_method_coverage() -> pd.DataFrame:
    """Return Section 4 paper method labels covered by package validation."""
    rows = []
    transcribed_labels = {
        PAPER_METHOD_LABELS[method] for method in TRANSCRIBED_PAPER_METHODS
    }
    for label in FULL_SECTION4_PAPER_METHOD_LABELS:
        package_method = _PACKAGE_METHOD_BY_PAPER_LABEL.get(label)
        rows.append(
            {
                "paper_method": label,
                "package_method": package_method or "",
                "implemented": package_method is not None,
                "paper_table_targets_transcribed": label in transcribed_labels,
            }
        )
    return pd.DataFrame(rows)


def section4_paper_configurations() -> list[dict[str, Any]]:
    """Return unique paper Section 4 simulation configurations."""
    columns = [
        "source_table",
        "dgp",
        "error_distribution",
        "n_groups",
        "n_per_group",
        "strong_fraction",
        "weak_fraction",
    ]
    configs = (
        paper_table_targets()[columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return configs.to_dict(orient="records")


def compare_summary_to_paper_targets(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare validation summaries with the paper's Section 4 target table."""
    observed = _summary_metric_rows(summary)
    if observed.empty:
        return pd.DataFrame()

    targets = paper_table_targets()
    key = [
        "dgp",
        "error_distribution",
        "n_groups",
        "n_per_group",
        "strong_fraction",
        "weak_fraction",
        "method",
        "metric",
    ]
    comparison = observed.merge(
        targets,
        on=key,
        how="inner",
        validate="many_to_one",
    )
    comparison["absolute_error"] = (
        comparison["observed_value"] - comparison["paper_value"]
    )
    comparison["relative_error"] = comparison["absolute_error"] / comparison[
        "paper_value"
    ].abs()
    comparison = _add_paper_implied_tail_diagnostics(comparison)
    columns = [
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
        "paper_method",
        "metric",
        "nobs",
        "repetitions",
        "observed_value",
        "paper_value",
        "absolute_error",
        "relative_error",
    ]
    columns.extend(column for column in _TAIL_DIAGNOSTIC_COLUMNS if column in comparison)
    columns.extend(
        column for column in _PAPER_IMPLIED_TAIL_COLUMNS if column in comparison
    )
    sort_columns = [
        "source_table",
        "dgp",
        "error_distribution",
        "strong_fraction",
        "n_groups",
        "config_index",
        "method",
        "metric",
    ]
    return comparison[columns].sort_values(sort_columns, ignore_index=True)


def _summary_metric_rows(summary: pd.DataFrame) -> pd.DataFrame:
    required = {"dgp", "error_distribution", "n_groups", "method"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            "summary is missing required paper comparison columns: "
            + ", ".join(sorted(missing))
        )

    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        n_groups = int(row["n_groups"])
        n_per_group = int(row.get("n_per_group", np.nan))
        nobs = float(row.get("nobs", n_groups * n_per_group))
        strong_fraction = float(row.get("strong_fraction", 0.0))
        weak_fraction = float(row.get("weak_fraction", 0.0))
        base = {
            "scenario": row.get("scenario", ""),
            "config_index": row.get("config_index", np.nan),
            "dgp": str(row["dgp"]),
            "error_distribution": _normalize_error_distribution(
                str(row["error_distribution"])
            ),
            "n_groups": n_groups,
            "n_per_group": n_per_group,
            "strong_fraction": strong_fraction,
            "weak_fraction": weak_fraction,
            "method": str(row["method"]),
            "nobs": nobs,
            "repetitions": _optional_int(row.get("repetitions", np.nan)),
        }
        base.update(
            {
                column: row[column]
                for column in _TAIL_DIAGNOSTIC_COLUMNS
                if column in row and pd.notna(row[column])
            }
        )
        if "scaled_mse" in row and pd.notna(row["scaled_mse"]):
            rows.append(
                {
                    **base,
                    "metric": "scaled_mse",
                    "observed_value": float(row["scaled_mse"]),
                }
            )
        if "scaled_mad" in row and pd.notna(row["scaled_mad"]):
            scaled_mad = float(row["scaled_mad"])
        elif "mad" in row and pd.notna(row["mad"]):
            scaled_mad = nobs * float(row["mad"])
        else:
            scaled_mad = np.nan
        if np.isfinite(scaled_mad):
            rows.append({**base, "metric": "scaled_mad", "observed_value": scaled_mad})
    return pd.DataFrame(rows)


def _add_paper_implied_tail_diagnostics(comparison: pd.DataFrame) -> pd.DataFrame:
    comparison = comparison.copy()
    for column in _PAPER_IMPLIED_TAIL_COLUMNS:
        comparison[column] = np.nan

    required = {
        "metric",
        "observed_value",
        "paper_value",
        "max_scaled_sq_error",
        "max_abs_error",
        "nobs",
        "repetitions",
    }
    if not required.issubset(comparison.columns):
        return comparison

    numeric = comparison[list(required - {"metric"})].apply(
        pd.to_numeric,
        errors="coerce",
    )
    mask = (
        comparison["metric"].eq("scaled_mse")
        & (numeric["paper_value"] > numeric["observed_value"])
        & (numeric["paper_value"] > 0.0)
        & (numeric["nobs"] > 0.0)
        & (numeric["repetitions"] > 0.0)
        & (numeric["max_scaled_sq_error"] >= 0.0)
    )
    if not mask.any():
        return comparison

    observed_total_scaled_sq = numeric.loc[mask, "observed_value"] * numeric.loc[
        mask, "repetitions"
    ]
    paper_total_scaled_sq = numeric.loc[mask, "paper_value"] * numeric.loc[
        mask, "repetitions"
    ]
    implied_scaled_sq = paper_total_scaled_sq - (
        observed_total_scaled_sq - numeric.loc[mask, "max_scaled_sq_error"]
    )
    valid = implied_scaled_sq > 0.0
    if not valid.any():
        return comparison

    valid_index = implied_scaled_sq[valid].index
    implied_scaled_sq = implied_scaled_sq.loc[valid_index]
    implied_abs_error = np.sqrt(implied_scaled_sq / numeric.loc[valid_index, "nobs"])
    comparison.loc[valid_index, "paper_implied_single_tail_abs_error"] = (
        implied_abs_error
    )
    comparison.loc[valid_index, "paper_implied_single_tail_scaled_sq_error"] = (
        implied_scaled_sq
    )
    comparison.loc[valid_index, "paper_implied_single_tail_mse_share"] = (
        implied_scaled_sq / paper_total_scaled_sq.loc[valid_index]
    )

    observed_max = numeric.loc[valid_index, "max_abs_error"]
    positive_observed_max = observed_max > 0.0
    ratio_index = valid_index[positive_observed_max.to_numpy()]
    comparison.loc[ratio_index, "paper_implied_tail_to_observed_max_ratio"] = (
        implied_abs_error.loc[ratio_index] / observed_max.loc[ratio_index]
    )
    return comparison


def _optional_int(value: Any) -> int | float:
    if pd.isna(value):
        return np.nan
    return int(value)


def _target_rows(
    *,
    rows: list[dict[str, Any]],
    source_table: str,
    dgp: str,
    error_distribution: str,
    strong_fraction: float,
    weak_fraction: float,
    values_by_group_count: dict[int, dict[str, list[float]]],
) -> None:
    for n_groups, metrics in values_by_group_count.items():
        for metric, values in metrics.items():
            for method, value in zip(IMPLEMENTED_PAPER_METHODS, values):
                rows.append(
                    {
                        "source_table": source_table,
                        "dgp": dgp,
                        "error_distribution": error_distribution,
                        "n_groups": n_groups,
                        "n_per_group": 500,
                        "strong_fraction": strong_fraction,
                        "weak_fraction": weak_fraction,
                        "method": method,
                        "paper_method": PAPER_METHOD_LABELS[method],
                        "metric": metric,
                        "paper_value": value,
                    }
                )


def _liml_mad_rows(
    *,
    rows: list[dict[str, Any]],
    source_table: str,
    dgp: str,
    error_distribution: str,
    strong_fraction: float,
    weak_fraction: float,
    values_by_group_count: dict[int, float],
) -> None:
    for n_groups, value in values_by_group_count.items():
        rows.append(
            {
                "source_table": source_table,
                "dgp": dgp,
                "error_distribution": error_distribution,
                "n_groups": n_groups,
                "n_per_group": 500,
                "strong_fraction": strong_fraction,
                "weak_fraction": weak_fraction,
                "method": "liml_interacted",
                "paper_method": PAPER_METHOD_LABELS["liml_interacted"],
                "metric": "scaled_mad",
                "paper_value": value,
            }
        )


def _extend_table2(rows: list[dict[str, Any]]) -> None:
    _target_rows(
        rows=rows,
        source_table="Table 2",
        dgp="dgp1",
        error_distribution="normal",
        strong_fraction=0.05,
        weak_fraction=0.0,
        values_by_group_count={
            40: {
                "scaled_mse": [436.882, 21.215, 22.669, 20.390, 20.390],
                "scaled_mad": [1923.779, 443.467, 423.261, 430.740, 430.740],
            },
            100: {
                "scaled_mse": [458.475, 23.663, 20.486, 19.609, 19.599],
                "scaled_mad": [3318.415, 761.088, 646.376, 627.586, 627.586],
            },
            200: {
                "scaled_mse": [387.056, 32.099, 23.843, 21.686, 21.683],
                "scaled_mad": [4341.239, 1267.731, 1043.499, 1021.058, 1018.517],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 2",
        dgp="dgp1",
        error_distribution="normal",
        strong_fraction=0.05,
        weak_fraction=0.0,
        values_by_group_count={40: 432.387, 100: 593.404, 200: 1001.641},
    )
    _target_rows(
        rows=rows,
        source_table="Table 2",
        dgp="dgp1",
        error_distribution="normal",
        strong_fraction=0.25,
        weak_fraction=0.0,
        values_by_group_count={
            40: {
                "scaled_mse": [16.240, 4.221, 4.334, 4.215, 4.215],
                "scaled_mad": [385.748, 206.675, 202.563, 198.870, 198.870],
            },
            100: {
                "scaled_mse": [17.488, 4.594, 4.390, 4.399, 4.399],
                "scaled_mad": [661.972, 311.558, 306.842, 320.676, 320.676],
            },
            200: {
                "scaled_mse": [15.208, 4.578, 4.156, 4.051, 4.050],
                "scaled_mad": [875.906, 469.002, 427.221, 421.565, 421.565],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 2",
        dgp="dgp1",
        error_distribution="normal",
        strong_fraction=0.25,
        weak_fraction=0.0,
        values_by_group_count={40: 203.090, 100: 307.353, 200: 418.238},
    )
    _target_rows(
        rows=rows,
        source_table="Table 2",
        dgp="dgp1",
        error_distribution="chisq3",
        strong_fraction=0.05,
        weak_fraction=0.0,
        values_by_group_count={
            40: {
                "scaled_mse": [6113.771, 135.939, 199.797, 129.725, 136.969],
                "scaled_mad": [5278.398, 1145.955, 1321.611, 1048.631, 1052.631],
            },
            100: {
                "scaled_mse": [2791.348, 189.829, 191.282, 124.294, 142.841],
                "scaled_mad": [7135.788, 2340.853, 1928.876, 1541.093, 1632.401],
            },
            200: {
                "scaled_mse": [2663.635, 314.390, 183.905, 121.197, 132.861],
                "scaled_mad": [10743.992, 4569.327, 2901.436, 2378.734, 2545.821],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 2",
        dgp="dgp1",
        error_distribution="chisq3",
        strong_fraction=0.05,
        weak_fraction=0.0,
        values_by_group_count={40: 1127.648, 100: 1796.948, 200: 2497.460},
    )
    _target_rows(
        rows=rows,
        source_table="Table 2",
        dgp="dgp1",
        error_distribution="chisq3",
        strong_fraction=0.25,
        weak_fraction=0.0,
        values_by_group_count={
            40: {
                "scaled_mse": [106.285, 25.804, 26.797, 25.590, 25.747],
                "scaled_mad": [1072.737, 492.951, 532.019, 496.556, 501.732],
            },
            100: {
                "scaled_mse": [94.806, 30.146, 29.370, 26.310, 27.498],
                "scaled_mad": [1439.007, 799.533, 803.419, 813.880, 790.668],
            },
            200: {
                "scaled_mse": [98.786, 34.913, 27.915, 25.958, 26.671],
                "scaled_mad": [2165.718, 1327.275, 1159.424, 1110.788, 1140.030],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 2",
        dgp="dgp1",
        error_distribution="chisq3",
        strong_fraction=0.25,
        weak_fraction=0.0,
        values_by_group_count={40: 486.464, 100: 792.318, 200: 1123.004},
    )


def _extend_table3(rows: list[dict[str, Any]]) -> None:
    _target_rows(
        rows=rows,
        source_table="Table 3",
        dgp="dgp2",
        error_distribution="normal",
        strong_fraction=0.025,
        weak_fraction=0.025,
        values_by_group_count={
            40: {
                "scaled_mse": [1435.660, 43.079, 47.703, 40.468, 41.944],
                "scaled_mad": [3239.714, 621.016, 661.890, 610.296, 623.647],
            },
            100: {
                "scaled_mse": [1905.386, 64.325, 54.582, 46.870, 48.225],
                "scaled_mad": [6199.708, 1318.320, 1069.955, 1036.453, 1017.512],
            },
            200: {
                "scaled_mse": [1120.562, 75.823, 47.525, 40.191, 40.989],
                "scaled_mad": [7228.982, 2043.003, 1493.240, 1459.994, 1445.584],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 3",
        dgp="dgp2",
        error_distribution="normal",
        strong_fraction=0.025,
        weak_fraction=0.025,
        values_by_group_count={40: 603.854, 100: 1016.949, 200: 1477.341},
    )
    _target_rows(
        rows=rows,
        source_table="Table 3",
        dgp="dgp2",
        error_distribution="normal",
        strong_fraction=0.125,
        weak_fraction=0.125,
        values_by_group_count={
            40: {
                "scaled_mse": [45.407, 8.481, 8.708, 8.414, 8.546],
                "scaled_mad": [647.919, 265.315, 276.036, 260.195, 272.046],
            },
            100: {
                "scaled_mse": [51.337, 9.325, 8.816, 8.750, 9.021],
                "scaled_mad": [1145.194, 464.062, 431.866, 431.265, 426.228],
            },
            200: {
                "scaled_mse": [42.263, 9.673, 8.297, 7.873, 8.114],
                "scaled_mad": [1456.950, 673.573, 604.213, 582.558, 565.383],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 3",
        dgp="dgp2",
        error_distribution="normal",
        strong_fraction=0.125,
        weak_fraction=0.125,
        values_by_group_count={40: 261.139, 100: 428.505, 200: 570.043},
    )
    _target_rows(
        rows=rows,
        source_table="Table 3",
        dgp="dgp2",
        error_distribution="chisq3",
        strong_fraction=0.025,
        weak_fraction=0.025,
        values_by_group_count={
            40: {
                "scaled_mse": [720188.116, 260.451, 564.523, 246.761, 259.075],
                "scaled_mad": [8722.373, 1709.158, 2091.322, 1523.892, 1644.423],
            },
            100: {
                "scaled_mse": [98979469.114, 543.929, 729.808, 293.429, 369.645],
                "scaled_mad": [14080.389, 4194.958, 3740.958, 2470.492, 2619.759],
            },
            200: {
                "scaled_mse": [9393.866, 741.439, 476.687, 223.481, 271.115],
                "scaled_mad": [18096.993, 7615.012, 4307.353, 3019.094, 3377.892],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 3",
        dgp="dgp2",
        error_distribution="chisq3",
        strong_fraction=0.025,
        weak_fraction=0.025,
        values_by_group_count={40: 1703.432, 100: 2883.843, 200: 3582.159},
    )
    _target_rows(
        rows=rows,
        source_table="Table 3",
        dgp="dgp2",
        error_distribution="chisq3",
        strong_fraction=0.125,
        weak_fraction=0.125,
        values_by_group_count={
            40: {
                "scaled_mse": [303.819, 51.212, 59.449, 52.709, 54.365],
                "scaled_mad": [1761.756, 716.053, 754.384, 710.452, 697.723],
            },
            100: {
                "scaled_mse": [279.308, 68.797, 67.898, 57.511, 60.779],
                "scaled_mad": [2458.124, 1280.568, 1188.038, 1107.445, 1163.068],
            },
            200: {
                "scaled_mse": [275.248, 84.233, 59.538, 51.447, 54.105],
                "scaled_mad": [3568.601, 2127.199, 1583.559, 1398.683, 1518.890],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 3",
        dgp="dgp2",
        error_distribution="chisq3",
        strong_fraction=0.125,
        weak_fraction=0.125,
        values_by_group_count={40: 731.689, 100: 1127.071, 200: 1544.208},
    )


def _extend_table4(rows: list[dict[str, Any]]) -> None:
    _target_rows(
        rows=rows,
        source_table="Table 4",
        dgp="dgp3",
        error_distribution="normal",
        strong_fraction=0.0,
        weak_fraction=0.0,
        values_by_group_count={
            40: {
                "scaled_mse": [467.057, 35.303, 38.713, 33.382, 33.810],
                "scaled_mad": [1992.620, 581.345, 552.848, 528.262, 556.787],
            },
            100: {
                "scaled_mse": [390.835, 25.589, 21.891, 20.821, 21.321],
                "scaled_mad": [3071.019, 786.978, 686.456, 648.956, 668.154],
            },
            200: {
                "scaled_mse": [316.284, 33.967, 24.779, 22.619, 23.071],
                "scaled_mad": [3923.494, 1328.891, 1061.998, 1039.967, 1033.464],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 4",
        dgp="dgp3",
        error_distribution="normal",
        strong_fraction=0.0,
        weak_fraction=0.0,
        values_by_group_count={40: 530.137, 100: 632.800, 200: 1061.145},
    )
    _target_rows(
        rows=rows,
        source_table="Table 4",
        dgp="dgp3",
        error_distribution="chisq3",
        strong_fraction=0.0,
        weak_fraction=0.0,
        values_by_group_count={
            40: {
                "scaled_mse": [9623.419, 206.304, 398.164, 213.024, 260.572],
                "scaled_mad": [5459.984, 1440.192, 1897.443, 1441.298, 1512.635],
            },
            100: {
                "scaled_mse": [2318.217, 201.848, 204.249, 132.450, 158.688],
                "scaled_mad": [6620.035, 2313.303, 1980.954, 1658.628, 1879.294],
            },
            200: {
                "scaled_mse": [2150.827, 327.106, 190.661, 125.153, 138.088],
                "scaled_mad": [9760.407, 4617.979, 2970.616, 2480.836, 2663.850],
            },
        },
    )
    _liml_mad_rows(
        rows=rows,
        source_table="Table 4",
        dgp="dgp3",
        error_distribution="chisq3",
        strong_fraction=0.0,
        weak_fraction=0.0,
        values_by_group_count={40: 1517.961, 100: 1876.630, 200: 2604.448},
    )


def _normalize_error_distribution(value: str) -> str:
    key = value.lower().replace("_", "").replace("-", "")
    if key in {"chisq3", "chi2", "chisquare3"}:
        return "chisq3"
    if key == "normal":
        return "normal"
    return value
