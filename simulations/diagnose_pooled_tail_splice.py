from __future__ import annotations

# ruff: noqa: E402

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def main() -> None:
    args = parse_args()
    comparison = pd.read_csv(args.comparison)
    candidates = read_candidate_csvs([Path(path) for path in args.candidate_csv])
    splice = tail_splice_candidates(
        comparison,
        candidates,
        relative_tolerance=args.relative_tolerance,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splice.to_csv(output_dir / "pooled_tail_splice_candidates.csv", index=False)
    (output_dir / "report.md").write_text(render_report(splice), encoding="utf-8")
    print(f"Pooled tail splice diagnostic report: {output_dir / 'report.md'}")
    print(f"Splice candidate rows: {len(splice)}")
    if not splice.empty:
        within = int(splice["spliced_within_tolerance"].sum())
        print(f"Candidates within tolerance: {within}/{len(splice)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Counterfactual one-tail splice diagnostic for pooled MSE rows."
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path(
            "validation/outputs/paper_tables/full_combined_reconstructed_dgp3/"
            "paper_comparison.csv"
        ),
    )
    parser.add_argument(
        "--candidate-csv",
        action="append",
        required=True,
        help="CSV containing config_index, seed, and abs_beta_hat columns.",
    )
    parser.add_argument("--relative-tolerance", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        default="validation/outputs/paper_tables/pooled_tail_splice_reconstruction",
    )
    return parser.parse_args()


def read_candidate_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["candidate_source"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def tail_splice_candidates(
    comparison: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    relative_tolerance: float = 0.25,
) -> pd.DataFrame:
    failed = comparison.loc[
        comparison["method"].eq("pooled")
        & comparison["metric"].eq("scaled_mse")
        & comparison["relative_error"].abs().gt(relative_tolerance)
    ].copy()
    if failed.empty or candidates.empty:
        return pd.DataFrame()

    required_candidate_cols = {"config_index", "seed", "abs_beta_hat"}
    missing = required_candidate_cols.difference(candidates.columns)
    if missing:
        raise ValueError(
            "candidate CSVs missing required columns: " + ", ".join(sorted(missing))
        )

    rows: list[dict[str, Any]] = []
    for target in failed.to_dict(orient="records"):
        config_index = int(target["config_index"])
        target_candidates = candidates.loc[
            pd.to_numeric(candidates["config_index"], errors="coerce").eq(config_index)
        ].copy()
        if target_candidates.empty:
            continue
        nobs = float(target["nobs"])
        repetitions = float(target["repetitions"])
        observed_total = float(target["observed_value"]) * repetitions
        current_top_scaled_sq = float(target["max_scaled_sq_error"])
        paper_value = float(target["paper_value"])
        for candidate in target_candidates.to_dict(orient="records"):
            candidate_abs_beta = float(candidate["abs_beta_hat"])
            candidate_scaled_sq = nobs * candidate_abs_beta**2
            spliced_total = observed_total - current_top_scaled_sq + candidate_scaled_sq
            spliced_scaled_mse = spliced_total / repetitions
            spliced_relative_error = (spliced_scaled_mse - paper_value) / abs(paper_value)
            rows.append(
                {
                    "config_index": config_index,
                    "source_table": target.get("source_table", ""),
                    "dgp": target.get("dgp", ""),
                    "error_distribution": target.get("error_distribution", ""),
                    "n_groups": int(target["n_groups"]),
                    "method": "pooled",
                    "seed": int(candidate["seed"]),
                    "candidate_source": candidate.get("candidate_source", ""),
                    "candidate_abs_beta_hat": candidate_abs_beta,
                    "candidate_scaled_sq_error": candidate_scaled_sq,
                    "current_observed_value": float(target["observed_value"]),
                    "paper_value": paper_value,
                    "current_relative_error": float(target["relative_error"]),
                    "current_max_scaled_sq_error": current_top_scaled_sq,
                    "spliced_scaled_mse": spliced_scaled_mse,
                    "spliced_relative_error": spliced_relative_error,
                    "spliced_abs_relative_error": abs(spliced_relative_error),
                    "spliced_within_tolerance": bool(
                        abs(spliced_relative_error) <= relative_tolerance
                    ),
                }
            )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["config_index", "spliced_abs_relative_error"],
        ignore_index=True,
    )


def render_report(splice: pd.DataFrame) -> str:
    lines = [
        "# adaptiveiv Pooled Tail Splice Diagnostic",
        "",
        "This is counterfactual reconstruction evidence. It replaces the current",
        "largest pooled squared-error contribution with a candidate seed's pooled",
        "squared-error contribution, then recomputes scaled MSE. It does not mutate",
        "the full validation outputs and does not prove recovery of the paper's",
        "original simulation seeds.",
        "",
        "## Ranked Splice Candidates",
        "",
        _markdown_table(splice),
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
