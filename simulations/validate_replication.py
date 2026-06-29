from __future__ import annotations

# ruff: noqa: E402

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from adaptiveiv.simulation import simulate_paper_section4_dgp
from adaptiveiv.validation import (
    estimate_methods_once,
    summarize_simulation_results,
    validate_simulation_summary,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_validation(args)
    summary = summarize_simulation_results(results)
    checks = validate_simulation_summary(summary)

    results_path = output_dir / "simulation_results.csv"
    summary_path = output_dir / "summary.csv"
    checks_path = output_dir / "checks.csv"
    report_path = output_dir / "report.md"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    checks.to_csv(checks_path, index=False)
    report_path.write_text(render_report(args, summary, checks), encoding="utf-8")

    print(f"Validation report: {report_path}")
    print(f"Simulation rows: {len(results)}")
    print(f"Checks passed: {int(checks['passed'].sum())}/{len(checks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptiveiv paper-style Monte Carlo validation."
    )
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--n-groups", type=int, default=40)
    parser.add_argument("--n-per-group", type=int, default=120)
    parser.add_argument("--n-splits", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--output-dir", default="validation/outputs/latest")
    return parser.parse_args()


def run_validation(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for scenario in default_scenarios(args.n_groups):
        for replication in range(args.repetitions):
            seed = args.seed + 10_000 * scenario["index"] + replication
            data = simulate_paper_section4_dgp(
                dgp=scenario["dgp"],
                n_groups=args.n_groups,
                n_per_group=args.n_per_group,
                strong_fraction=scenario["strong_fraction"],
                weak_fraction=scenario["weak_fraction"],
                error_distribution=scenario["error_distribution"],
                seed=seed,
            )
            estimates = estimate_methods_once(data, random_state=seed, n_splits=args.n_splits)
            estimates.insert(0, "replication", replication)
            estimates.insert(0, "scenario", scenario["name"])
            estimates["dgp"] = scenario["dgp"]
            estimates["error_distribution"] = scenario["error_distribution"]
            estimates["n_groups"] = args.n_groups
            estimates["n_per_group"] = args.n_per_group
            estimates["strong_fraction"] = scenario["strong_fraction"]
            estimates["weak_fraction"] = scenario["weak_fraction"]
            estimates["seed"] = seed
            rows.append(estimates)
    return pd.concat(rows, ignore_index=True)


def default_scenarios(n_groups: int) -> list[dict[str, Any]]:
    sparse_share = max(1 / n_groups, 0.05)
    return [
        {
            "index": 0,
            "name": "dgp1_sparse_normal",
            "dgp": "dgp1",
            "strong_fraction": sparse_share,
            "weak_fraction": 0.0,
            "error_distribution": "normal",
        },
        {
            "index": 1,
            "name": "dgp2_sparse_normal",
            "dgp": "dgp2",
            "strong_fraction": sparse_share,
            "weak_fraction": sparse_share,
            "error_distribution": "normal",
        },
        {
            "index": 2,
            "name": "dgp3_nonseparated_normal",
            "dgp": "dgp3",
            "strong_fraction": 0.0,
            "weak_fraction": 0.0,
            "error_distribution": "normal",
        },
    ]


def render_report(
    args: argparse.Namespace,
    summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    lines = [
        "# adaptiveiv Replication Validation Report",
        "",
        "## Settings",
        "",
        f"- Repetitions per scenario: {args.repetitions}",
        f"- Groups: {args.n_groups}",
        f"- Observations per group: {args.n_per_group}",
        f"- Split repetitions per fit: {args.n_splits}",
        f"- Base seed: {args.seed}",
        "- Estimators: pooled, fully_interacted, split_interacted, adaptive, oracle",
        "",
        "## Summary Metrics",
        "",
        _markdown_table(
            summary[
                [
                    "scenario",
                    "method",
                    "repetitions",
                    "finite_share",
                    "scaled_mse",
                    "mad",
                    "mean_selected_total",
                ]
            ]
        ),
        "",
        "## Qualitative Checks",
        "",
        _markdown_table(checks[["scenario", "check", "passed", "detail"]]),
        "",
        "## Scope limits",
        "",
        "- This report validates the estimators currently implemented in `adaptiveiv`.",
        "- It is a qualitative replication of the paper's Section 4 Monte Carlo logic.",
        "- It does not reproduce UJIVE, IJIVE, lasso, empirical applications, or broad inference.",
        "- Validation fits use `cov_type=\"none\"`; inference is validated separately.",
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
