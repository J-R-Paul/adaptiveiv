from __future__ import annotations

# ruff: noqa: E402

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from adaptiveiv import AdaptiveIV, simulate_paper_section4_dgp


def main() -> None:
    args = parse_args()
    settings = resolve_settings(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_inference_validation(settings)
    summary = summarize_inference_results(results)
    checks = validate_inference_summary(summary, preset=args.preset)

    results_path = output_dir / "inference_results.csv"
    summary_path = output_dir / "summary.csv"
    checks_path = output_dir / "checks.csv"
    report_path = output_dir / "report.md"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    checks.to_csv(checks_path, index=False)
    report_path.write_text(
        render_report(args, settings, summary, checks),
        encoding="utf-8",
    )

    print(f"Inference validation report: {report_path}")
    print(f"Simulation rows: {len(results)}")
    print(f"Checks passed: {int(checks['passed'].sum())}/{len(checks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptiveiv paper-baseline inference validation."
    )
    parser.add_argument("--preset", choices=["smoke", "release"], default="smoke")
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--n-groups", type=int)
    parser.add_argument("--n-per-group", type=int)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--output-dir", default="validation/outputs/inference")
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    if args.preset == "release":
        repetitions = 50
        n_groups = 40
        n_per_group = 120
    else:
        repetitions = 5
        n_groups = 12
        n_per_group = 60
    return {
        "preset": args.preset,
        "repetitions": args.repetitions or repetitions,
        "n_groups": args.n_groups or n_groups,
        "n_per_group": args.n_per_group or n_per_group,
        "seed": args.seed,
        "beta": 0.0,
        "strong_fraction": 0.35,
    }


def run_inference_validation(settings: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for replication in range(settings["repetitions"]):
        seed = int(settings["seed"] + replication)
        data = simulate_paper_section4_dgp(
            dgp="dgp1",
            n_groups=settings["n_groups"],
            n_per_group=settings["n_per_group"],
            beta=settings["beta"],
            strong_fraction=settings["strong_fraction"],
            error_distribution="normal",
            seed=seed,
        )
        model = AdaptiveIV.from_formula("Y ~ 1 + X + [W ~ Z]", data=data, groups="group")
        try:
            result = model.fit(random_state=seed, cov_type="homoskedastic")
            interval = result.conf_int()
            beta_hat = float(result.params["W"])
            bse = float(result.bse["W"])
            inference_diagnostics = result.inference_diagnostics.set_index("component")
            lower = float(interval.loc["W", "lower"])
            upper = float(interval.loc["W", "upper"])
            pvalue = float(result.pvalues["W"])
            rows.append(
                {
                    "replication": replication,
                    "seed": seed,
                    "beta_hat": beta_hat,
                    "true_beta": settings["beta"],
                    "bse": bse,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "covered_95": lower <= settings["beta"] <= upper,
                    "rejected_5pct": pvalue < 0.05,
                    "pvalue": pvalue,
                    "nobs": result.nobs,
                    "ngroups": result.ngroups,
                    "selected_total": result.selection_summary.get("selected_total", 0),
                    "cov_estimator": result.cov_estimator,
                    "variance_component_a": float(
                        inference_diagnostics.loc["a", "variance"]
                    ),
                    "variance_component_b": float(
                        inference_diagnostics.loc["b", "variance"]
                    ),
                    "variance_average": float(
                        inference_diagnostics.loc["average", "variance"]
                    ),
                    "df_resid": int(inference_diagnostics.loc["average", "df_resid"]),
                    "finite": bool(np.isfinite(beta_hat) and np.isfinite(bse)),
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - retained in artifacts.
            rows.append(
                {
                    "replication": replication,
                    "seed": seed,
                    "beta_hat": np.nan,
                    "true_beta": settings["beta"],
                    "bse": np.nan,
                    "ci_lower": np.nan,
                    "ci_upper": np.nan,
                    "covered_95": False,
                    "rejected_5pct": False,
                    "pvalue": np.nan,
                    "nobs": len(data),
                    "ngroups": int(data["group"].nunique()),
                    "selected_total": 0,
                    "cov_estimator": "",
                    "variance_component_a": np.nan,
                    "variance_component_b": np.nan,
                    "variance_average": np.nan,
                    "df_resid": np.nan,
                    "finite": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return pd.DataFrame(rows)


def summarize_inference_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    finite = results["finite"].astype(bool)
    finite_results = results.loc[finite]
    return pd.DataFrame(
        [
            {
                "method": "adaptive",
                "cov_estimator": "paper_homoskedastic",
                "repetitions": int(len(results)),
                "finite_share": float(finite.mean()) if len(results) else 0.0,
                "coverage_95": float(finite_results["covered_95"].mean())
                if len(finite_results)
                else np.nan,
                "rejection_5pct": float(finite_results["rejected_5pct"].mean())
                if len(finite_results)
                else np.nan,
                "mean_bse": float(finite_results["bse"].mean())
                if len(finite_results)
                else np.nan,
                "mean_selected_total": float(finite_results["selected_total"].mean())
                if len(finite_results)
                else np.nan,
            }
        ]
    )


def validate_inference_summary(summary: pd.DataFrame, *, preset: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(
            [{"check": "summary_not_empty", "passed": False, "detail": "empty summary"}]
        )
    row = summary.iloc[0]
    coverage = float(row["coverage_95"])
    rejection = float(row["rejection_5pct"])
    finite_share = float(row["finite_share"])
    if preset == "release":
        coverage_low, coverage_high = 0.85, 1.0
        rejection_high = 0.15
        finite_min = 0.95
    else:
        coverage_low, coverage_high = 0.0, 1.0
        rejection_high = 1.0
        finite_min = 0.5
    return pd.DataFrame(
        [
            {
                "check": "finite_share",
                "passed": finite_share >= finite_min,
                "detail": f"finite_share={finite_share:.6g}; required>={finite_min:.6g}",
            },
            {
                "check": "coverage_95_range",
                "passed": coverage_low <= coverage <= coverage_high,
                "detail": (
                    f"coverage_95={coverage:.6g}; "
                    f"required between {coverage_low:.6g} and {coverage_high:.6g}"
                ),
            },
            {
                "check": "rejection_5pct_upper_bound",
                "passed": rejection <= rejection_high,
                "detail": (
                    f"rejection_5pct={rejection:.6g}; "
                    f"required<={rejection_high:.6g}"
                ),
            },
        ]
    )


def render_report(
    args: argparse.Namespace,
    settings: dict[str, Any],
    summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    lines = [
        "# adaptiveiv Inference Validation Report",
        "",
        "## Settings",
        "",
        f"- Preset: {args.preset}",
        f"- Repetitions: {settings['repetitions']}",
        f"- Groups: {settings['n_groups']}",
        f"- Observations per group: {settings['n_per_group']}",
        f"- Base seed: {settings['seed']}",
        "- DGP: dgp1, normal errors, beta=0",
        "- Covariance estimator: paper_homoskedastic",
        "",
        "## Summary",
        "",
        _markdown_table(summary),
        "",
        "## Checks",
        "",
        _markdown_table(checks),
        "",
        "## Scope limits",
        "",
        "- This validates the first paper-baseline homoskedastic inference slice.",
        "- It does not validate robust, clustered, bootstrap, repeated-split, or absolute-selection inference.",
        "- Smoke presets check plumbing; release presets provide stronger but still reviewable evidence.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.select_dtypes(include=["floating"]).columns:
        display[column] = display[column].map(lambda value: f"{value:.6g}")
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(display.columns)) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in display.to_numpy()
    ]
    return "\n".join([header, separator, *rows])


if __name__ == "__main__":
    main()
