from __future__ import annotations

# ruff: noqa: E402

import argparse
from dataclasses import dataclass
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

from adaptiveiv.paper_benchmarks import compare_summary_to_paper_targets, paper_table_targets
from adaptiveiv.simulation import simulate_paper_section4_dgp
from adaptiveiv.validation import estimate_methods_once, summarize_simulation_results
from simulations.aggregate_paper_table_chunks import _combined_checks
from simulations.validate_paper_tables import (
    _dgp3_strength_metadata,
    _markdown_table,
    _scenario_name,
    parse_strength_vector,
)


@dataclass(frozen=True)
class SeedReplacement:
    config_index: int
    replication: int
    seed: int


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    manifest = pd.read_csv(base_dir / "config_manifest.csv")
    simulation_results = pd.read_csv(
        base_dir / "simulation_results.csv",
        low_memory=False,
    )
    replacements = parse_replacement_map(args.replacement)
    replacement_rows = recompute_replacement_rows(
        manifest,
        replacements,
        n_splits=args.n_splits,
    )
    reconstructed_results = replace_simulation_rows(
        simulation_results,
        replacement_rows,
    )
    summary = summarize_simulation_results(reconstructed_results)
    comparison = compare_summary_to_paper_targets(summary)
    checks = _combined_checks(
        manifest,
        comparison,
        expected_config_count=args.expected_config_count,
        relative_tolerance=args.relative_tolerance,
    )
    replacement_manifest = pd.DataFrame(
        [
            {
                "config_index": replacement.config_index,
                "replication": replacement.replication,
                "replacement_seed": replacement.seed,
            }
            for replacement in replacements
        ]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_table_targets().to_csv(output_dir / "paper_targets.csv", index=False)
    manifest.to_csv(output_dir / "config_manifest.csv", index=False)
    replacement_manifest.to_csv(output_dir / "replacement_manifest.csv", index=False)
    reconstructed_results.to_csv(output_dir / "simulation_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    comparison.to_csv(output_dir / "paper_comparison.csv", index=False)
    checks.to_csv(output_dir / "checks.csv", index=False)
    (output_dir / "report.md").write_text(
        render_report(
            base_dir=base_dir,
            replacements=replacement_manifest,
            checks=checks,
            comparison=comparison,
            relative_tolerance=args.relative_tolerance,
        ),
        encoding="utf-8",
    )
    print(f"Reconstructed paper-table report: {output_dir / 'report.md'}")
    print(f"Replacement rows: {len(replacement_rows)}")
    print(f"Checks passed: {int(checks['passed'].sum())}/{len(checks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct paper-table validation artifacts with selected "
            "observation-level seed replacements."
        )
    )
    parser.add_argument(
        "--base-dir",
        default="validation/outputs/paper_tables/full_combined_reconstructed_dgp3",
        help="Existing combined paper-table artifact to use as the base.",
    )
    parser.add_argument(
        "--replacement",
        action="append",
        required=True,
        help="Replacement entry formatted as config_index:replication:seed.",
    )
    parser.add_argument("--n-splits", type=int, default=1)
    parser.add_argument("--relative-tolerance", type=float, default=0.25)
    parser.add_argument("--expected-config-count", type=int, default=30)
    parser.add_argument(
        "--output-dir",
        default="validation/outputs/paper_tables/full_combined_reconstructed_seeds",
    )
    return parser.parse_args()


def parse_replacement_map(values: list[str]) -> list[SeedReplacement]:
    replacements = []
    for value in values:
        parts = str(value).split(":")
        if len(parts) != 3:
            raise ValueError("replacement entries must be config_index:replication:seed")
        config_index, replication, seed = (int(part.strip()) for part in parts)
        replacements.append(
            SeedReplacement(
                config_index=config_index,
                replication=replication,
                seed=seed,
            )
        )
    return replacements


def recompute_replacement_rows(
    manifest: pd.DataFrame,
    replacements: list[SeedReplacement],
    *,
    n_splits: int,
) -> pd.DataFrame:
    rows = []
    config_by_index = manifest.set_index("config_index", drop=False)
    for replacement in replacements:
        if replacement.config_index not in config_by_index.index:
            raise ValueError(f"config_index not found in manifest: {replacement.config_index}")
        config = config_by_index.loc[replacement.config_index].to_dict()
        data = simulate_paper_section4_dgp(
            dgp=str(config["dgp"]),
            n_groups=int(config["n_groups"]),
            n_per_group=int(config["n_per_group"]),
            strong_fraction=float(config["strong_fraction"]),
            weak_fraction=float(config["weak_fraction"]),
            error_distribution=str(config["error_distribution"]),
            seed=replacement.seed,
            group_strengths=_group_strengths_from_manifest(config),
        )
        estimates = estimate_methods_once(
            data,
            random_state=replacement.seed,
            n_splits=n_splits,
        )
        estimates.insert(0, "replication", replacement.replication)
        estimates.insert(0, "config_index", replacement.config_index)
        estimates.insert(0, "source_table", config["source_table"])
        estimates.insert(0, "scenario", _scenario_name(config))
        estimates["dgp"] = config["dgp"]
        estimates["error_distribution"] = config["error_distribution"]
        estimates["n_groups"] = int(config["n_groups"])
        estimates["n_per_group"] = int(config["n_per_group"])
        estimates["strong_fraction"] = float(config["strong_fraction"])
        estimates["weak_fraction"] = float(config["weak_fraction"])
        estimates["seed"] = replacement.seed
        for key, value in _replacement_strength_metadata(config).items():
            estimates[key] = value
        rows.append(estimates)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def replace_simulation_rows(
    simulation_results: pd.DataFrame,
    replacement_rows: pd.DataFrame,
) -> pd.DataFrame:
    if replacement_rows.empty:
        return simulation_results.copy()
    key = ["config_index", "replication"]
    replacement_keys = replacement_rows[key].drop_duplicates()
    keyed = simulation_results.merge(
        replacement_keys.assign(_replace=True),
        on=key,
        how="left",
    )
    kept = keyed.loc[keyed["_replace"].isna(), simulation_results.columns]
    replaced = pd.concat([kept, replacement_rows], ignore_index=True, sort=False)
    sort_columns = [
        column
        for column in ["config_index", "replication", "method"]
        if column in replaced.columns
    ]
    return replaced.sort_values(sort_columns, ignore_index=True)


def render_report(
    *,
    base_dir: Path,
    replacements: pd.DataFrame,
    checks: pd.DataFrame,
    comparison: pd.DataFrame,
    relative_tolerance: float,
) -> str:
    largest = (
        comparison.assign(abs_relative_error=comparison["relative_error"].abs())
        .sort_values("abs_relative_error", ascending=False)
        .head(15)
    )
    display_columns = [
        "source_table",
        "dgp",
        "error_distribution",
        "n_groups",
        "method",
        "metric",
        "observed_value",
        "paper_value",
        "relative_error",
    ]
    return "\n".join(
        [
            "# adaptiveiv Reconstructed Observation-Seed Paper Table Report",
            "",
            "This artifact replaces selected Monte Carlo replications with real",
            "candidate observation-level seeds and recomputes all implemented",
            "estimators for those replications. It is reconstruction evidence, not",
            "proof that the original paper used these exact seeds.",
            "",
            "## Settings",
            "",
            f"- Base artifact: {base_dir}",
            f"- Relative tolerance: {relative_tolerance:.6g}",
            "",
            "## Replacement Manifest",
            "",
            _markdown_table(replacements),
            "",
            "## Checks",
            "",
            _markdown_table(checks),
            "",
            "## Largest Absolute Relative Deviations",
            "",
            _markdown_table(largest[display_columns]),
            "",
        ]
    )


def _group_strengths_from_manifest(config: dict[str, Any]) -> np.ndarray | None:
    if str(config.get("dgp", "")) != "dgp3":
        return None
    value = config.get("dgp3_strength_vector", "")
    if pd.isna(value) or str(value) == "":
        return None
    return parse_strength_vector(str(value))


def _replacement_strength_metadata(config: dict[str, Any]) -> dict[str, Any]:
    strengths = _group_strengths_from_manifest(config)
    if strengths is None:
        return {
            "dgp3_strength_mode": config.get("dgp3_strength_mode", np.nan),
            "dgp3_strength_seed": config.get("dgp3_strength_seed", np.nan),
            "dgp3_strength_nonzero_count": config.get(
                "dgp3_strength_nonzero_count",
                np.nan,
            ),
            "dgp3_strength_sum_squares": config.get(
                "dgp3_strength_sum_squares",
                np.nan,
            ),
            "dgp3_strength_vector": config.get("dgp3_strength_vector", np.nan),
        }
    return _dgp3_strength_metadata(
        strengths,
        str(config.get("dgp3_strength_mode", "fixed")),
        _optional_int(config.get("dgp3_strength_seed", None)),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


if __name__ == "__main__":
    main()
