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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from simulations.diagnose_pooled_tail_seeds import pooled_tail_seed_diagnostic


def main() -> None:
    args = parse_args()
    comparison = pd.read_csv(args.comparison)
    targets = failed_pooled_mse_targets(
        comparison,
        relative_tolerance=args.relative_tolerance,
    )
    seeds_by_config = random_seeds_for_targets(
        targets,
        seed_count=args.seed_count,
        random_seed=args.random_seed,
    )
    candidates = scan_target_seed_candidates(
        targets,
        seeds_by_config=seeds_by_config,
        top_k=args.top_k,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(output_dir / "pooled_mse_targets.csv", index=False)
    candidates.to_csv(output_dir / "pooled_mse_target_seed_candidates.csv", index=False)
    (output_dir / "report.md").write_text(
        render_report(targets, candidates, args.seed_count),
        encoding="utf-8",
    )
    print(f"Pooled MSE target diagnostic report: {output_dir / 'report.md'}")
    print(f"Failed pooled MSE rows: {len(targets)}")
    print(f"Candidate seed rows: {len(candidates)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan seeds near paper-implied pooled chi-square MSE tails."
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path(
            "validation/outputs/paper_tables/full_combined_reconstructed_dgp3/"
            "paper_comparison.csv"
        ),
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.25)
    parser.add_argument("--seed-count", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260624)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default="validation/outputs/paper_tables/pooled_mse_tail_targets",
    )
    return parser.parse_args()


def failed_pooled_mse_targets(
    comparison: pd.DataFrame,
    *,
    relative_tolerance: float = 0.25,
) -> pd.DataFrame:
    failed = comparison.loc[
        comparison["method"].eq("pooled")
        & comparison["metric"].eq("scaled_mse")
        & comparison["relative_error"].abs().gt(relative_tolerance)
    ].copy()
    if failed.empty:
        return pd.DataFrame(
            columns=[
                "config_index",
                "source_table",
                "dgp",
                "error_distribution",
                "n_groups",
                "n_per_group",
                "strong_fraction",
                "weak_fraction",
                "observed_value",
                "paper_value",
                "relative_error",
                "target_status",
                "target_abs_beta",
            ]
        )

    implied = failed.get("paper_implied_single_tail_abs_error", pd.Series(np.nan))
    failed["target_abs_beta"] = pd.to_numeric(implied, errors="coerce")
    failed["target_status"] = np.where(
        np.isfinite(failed["target_abs_beta"]),
        "missing_larger_tail",
        np.where(
            failed["observed_value"] > failed["paper_value"],
            "observed_exceeds_paper",
            "missing_implied_tail_diagnostic",
        ),
    )
    columns = [
        "config_index",
        "source_table",
        "dgp",
        "error_distribution",
        "n_groups",
        "n_per_group",
        "strong_fraction",
        "weak_fraction",
        "observed_value",
        "paper_value",
        "relative_error",
        "target_status",
        "target_abs_beta",
    ]
    optional_columns = [
        "max_abs_error",
        "paper_implied_tail_to_observed_max_ratio",
        "paper_implied_single_tail_mse_share",
    ]
    return failed[[column for column in columns + optional_columns if column in failed]]


def random_seeds_for_targets(
    targets: pd.DataFrame,
    *,
    seed_count: int,
    random_seed: int,
) -> dict[int, list[int]]:
    if seed_count < 1:
        raise ValueError("seed_count must be positive")
    rng = np.random.default_rng(random_seed)
    seeds_by_config: dict[int, list[int]] = {}
    for config_index in targets.loc[
        targets["target_status"].eq("missing_larger_tail"),
        "config_index",
    ].astype(int):
        seeds = rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=seed_count,
            dtype=np.uint64,
        )
        seeds_by_config[int(config_index)] = [int(seed) for seed in seeds]
    return seeds_by_config


def scan_target_seed_candidates(
    targets: pd.DataFrame,
    *,
    seeds_by_config: dict[int, list[int]],
    top_k: int = 10,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in targets.to_dict(orient="records"):
        if target["target_status"] != "missing_larger_tail":
            continue
        config_index = int(target["config_index"])
        target_abs_beta = float(target["target_abs_beta"])
        diagnostics = [
            {
                **pooled_tail_seed_diagnostic(target, int(seed)),
                "target_abs_beta": target_abs_beta,
            }
            for seed in seeds_by_config.get(config_index, [])
        ]
        if not diagnostics:
            continue
        frame = pd.DataFrame(diagnostics)
        frame["target_distance"] = (
            frame["abs_beta_hat"] - target_abs_beta
        ).abs()
        frame["target_relative_distance"] = frame["target_distance"] / target_abs_beta
        frame = frame.sort_values(
            ["target_distance", "abs_beta_hat"],
            ascending=[True, False],
            ignore_index=True,
        )
        rows.extend(frame.head(top_k).to_dict(orient="records"))
    return pd.DataFrame(rows)


def render_report(
    targets: pd.DataFrame,
    candidates: pd.DataFrame,
    seed_count: int,
) -> str:
    lines = [
        "# adaptiveiv Pooled MSE Tail Target Diagnostic",
        "",
        f"- Failed pooled MSE rows: {len(targets)}",
        f"- Random seeds scanned per missing-tail row: {seed_count}",
        "",
        "## Target Rows",
        "",
        _markdown_table(targets),
        "",
        "## Closest Candidate Seeds",
        "",
        _markdown_table(candidates),
        "",
        "## Interpretation",
        "",
        "- `missing_larger_tail` rows are under the paper MSE target and can be",
        "  investigated by looking for real seeds near the paper-implied tail.",
        "- `observed_exceeds_paper` rows already have larger MSE than the paper target;",
        "  adding a larger single tail event cannot reconcile them.",
        "- Candidate seeds are diagnostics, not proof that the paper's original",
        "  observation-level seeds have been recovered.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.to_dict(orient="records"):
        values = [
            _format_markdown_value(row[column], column_name=str(column))
            for column in frame.columns
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_markdown_value(value: Any, *, column_name: str = "") -> str:
    if pd.isna(value):
        return "nan"
    if "seed" in column_name and isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
