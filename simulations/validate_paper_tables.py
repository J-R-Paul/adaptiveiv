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

from adaptiveiv.paper_benchmarks import (
    compare_summary_to_paper_targets,
    paper_table_targets,
    section4_paper_configurations,
)
from adaptiveiv.simulation import simulate_paper_section4_dgp
from adaptiveiv.simulation import paper_group_strengths
from adaptiveiv.validation import estimate_methods_once, summarize_simulation_results


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = selected_configurations(args)
    results = run_paper_table_validation(args, configs)
    summary = summarize_simulation_results(results)
    comparison = compare_summary_to_paper_targets(summary)
    relative_tolerance = _relative_tolerance(args)
    checks = paper_comparison_checks(
        comparison,
        relative_tolerance,
        expected_matches=expected_paper_target_matches(configs),
    )
    targets = paper_table_targets()

    targets.to_csv(output_dir / "paper_targets.csv", index=False)
    paper_table_config_manifest(args, configs).to_csv(
        output_dir / "config_manifest.csv",
        index=False,
    )
    results.to_csv(output_dir / "simulation_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    comparison.to_csv(output_dir / "paper_comparison.csv", index=False)
    checks.to_csv(output_dir / "checks.csv", index=False)
    (output_dir / "report.md").write_text(
        render_report(args, configs, comparison, checks),
        encoding="utf-8",
    )

    print(f"Paper-table validation report: {output_dir / 'report.md'}")
    print(f"Configurations: {len(configs)}")
    print(f"Simulation rows: {len(results)}")
    print(f"Matched comparison rows: {len(comparison)}")
    print(f"Checks passed: {int(checks['passed'].sum())}/{len(checks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare adaptiveiv simulations with paper Table 2-4 targets."
    )
    parser.add_argument(
        "--preset",
        choices=["smoke", "release", "full"],
        default="smoke",
        help=(
            "smoke runs one DGP1 configuration with 2 repetitions; release runs "
            "all implemented paper configurations with 100 repetitions; full uses "
            "the paper's 500 repetitions."
        ),
    )
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--n-splits", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--relative-tolerance", type=float)
    parser.add_argument("--max-configs", type=int)
    parser.add_argument(
        "--config-start",
        type=int,
        default=0,
        help="Zero-based first paper configuration to run after filters.",
    )
    parser.add_argument(
        "--config-stop",
        type=int,
        help="Zero-based half-open stop index after filters.",
    )
    parser.add_argument("--only-table", choices=["Table 2", "Table 3", "Table 4"])
    parser.add_argument("--only-dgp", choices=["dgp1", "dgp2", "dgp3"])
    parser.add_argument("--only-error", choices=["normal", "chisq3"])
    parser.add_argument(
        "--redraw-dgp3-strengths",
        action="store_true",
        help=(
            "Redraw DGP3 group first-stage strengths every replication. By default "
            "the paper-table runner treats DGP parameters as fixed within each "
            "configuration and only redraws observation-level data."
        ),
    )
    parser.add_argument(
        "--dgp3-strength-seed-base",
        type=int,
        help=(
            "Optional base seed for fixed DGP3 first-stage strength vectors. "
            "When omitted, --seed is used so existing runs remain reproducible. "
            "The final strength seed is base + 100000 * config_index."
        ),
    )
    parser.add_argument(
        "--dgp3-strength-seed",
        type=int,
        help=(
            "Optional exact seed for fixed DGP3 first-stage strength vectors. "
            "This takes precedence over --dgp3-strength-seed-base and is useful "
            "for targeted Table 4 reconstruction diagnostics."
        ),
    )
    parser.add_argument(
        "--dgp3-strength-seed-map",
        help=(
            "Optional comma-separated config-index seed map for fixed DGP3 "
            "strength vectors, for example '24=11890,26=45610'. Entries take "
            "precedence over --dgp3-strength-seed and --dgp3-strength-seed-base."
        ),
    )
    parser.add_argument("--output-dir", default="validation/outputs/paper_tables")
    return parser.parse_args()


def selected_configurations(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = [
        {**config, "config_index": index}
        for index, config in enumerate(section4_paper_configurations())
    ]
    if args.only_table is not None:
        configs = [config for config in configs if config["source_table"] == args.only_table]
    if args.only_dgp is not None:
        configs = [config for config in configs if config["dgp"] == args.only_dgp]
    if args.only_error is not None:
        configs = [
            config
            for config in configs
            if config["error_distribution"] == args.only_error
        ]

    config_start = args.config_start
    config_stop = args.config_stop
    if config_start < 0:
        raise ValueError("config-start must be nonnegative")
    if config_stop is not None and config_stop < config_start:
        raise ValueError("config-stop must be greater than or equal to config-start")
    configs = configs[config_start:config_stop]

    max_configs = args.max_configs
    if max_configs is None and args.preset == "smoke":
        max_configs = 1
    if max_configs is not None:
        configs = configs[:max_configs]
    return configs


def run_paper_table_validation(
    args: argparse.Namespace,
    configs: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    repetitions = _repetitions(args)
    for config in configs:
        config_index = int(config["config_index"])
        group_strengths, strength_mode, strength_seed = _dgp3_strength_config(args, config)
        strength_metadata = _dgp3_strength_metadata(
            group_strengths,
            strength_mode,
            strength_seed,
        )
        for replication in range(repetitions):
            seed = args.seed + 100_000 * config_index + replication
            data = simulate_paper_section4_dgp(
                dgp=config["dgp"],
                n_groups=int(config["n_groups"]),
                n_per_group=int(config["n_per_group"]),
                strong_fraction=float(config["strong_fraction"]),
                weak_fraction=float(config["weak_fraction"]),
                error_distribution=config["error_distribution"],
                seed=seed,
                group_strengths=group_strengths,
            )
            estimates = estimate_methods_once(
                data,
                random_state=seed,
                n_splits=args.n_splits,
            )
            estimates.insert(0, "replication", replication)
            estimates.insert(0, "config_index", config["config_index"])
            estimates.insert(0, "source_table", config["source_table"])
            estimates.insert(0, "scenario", _scenario_name(config))
            estimates["dgp"] = config["dgp"]
            estimates["error_distribution"] = config["error_distribution"]
            estimates["n_groups"] = config["n_groups"]
            estimates["n_per_group"] = config["n_per_group"]
            estimates["strong_fraction"] = config["strong_fraction"]
            estimates["weak_fraction"] = config["weak_fraction"]
            estimates["seed"] = seed
            for key, value in strength_metadata.items():
                estimates[key] = value
            rows.append(estimates)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def paper_table_config_manifest(
    args: argparse.Namespace,
    configs: list[dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for config in configs:
        group_strengths, strength_mode, strength_seed = _dgp3_strength_config(args, config)
        rows.append(
            {
                **config,
                **_dgp3_strength_metadata(
                    group_strengths,
                    strength_mode,
                    strength_seed,
                ),
            }
        )
    return pd.DataFrame(rows)


def _dgp3_strength_config(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> tuple[Any, str, int | None]:
    if config["dgp"] != "dgp3" or getattr(args, "redraw_dgp3_strengths", False):
        return None, "redraw" if config["dgp"] == "dgp3" else "not_applicable", None
    config_index = int(config["config_index"])
    seed_map = _parse_dgp3_strength_seed_map(
        getattr(args, "dgp3_strength_seed_map", None)
    )
    exact_seed = seed_map.get(config_index, getattr(args, "dgp3_strength_seed", None))
    if exact_seed is not None:
        strength_seed = int(exact_seed)
    else:
        seed_base = getattr(args, "dgp3_strength_seed_base", None)
        if seed_base is None:
            seed_base = args.seed
        strength_seed = int(seed_base) + 100_000 * config_index
    return paper_group_strengths(
        config["dgp"],
        n_groups=int(config["n_groups"]),
        strong_fraction=float(config["strong_fraction"]),
        weak_fraction=float(config["weak_fraction"]),
        seed=strength_seed,
    ), "fixed", strength_seed


def _parse_dgp3_strength_seed_map(value: str | None) -> dict[int, int]:
    if value is None or str(value).strip() == "":
        return {}
    parsed: dict[int, int] = {}
    for item in str(value).split(","):
        if "=" not in item:
            raise ValueError("dgp3-strength-seed-map entries must be config=seed")
        config_text, seed_text = item.split("=", 1)
        parsed[int(config_text.strip())] = int(seed_text.strip())
    return parsed


def _dgp3_strength_metadata(
    group_strengths: Any,
    strength_mode: str,
    strength_seed: int | None,
) -> dict[str, Any]:
    if group_strengths is None:
        return {
            "dgp3_strength_mode": strength_mode,
            "dgp3_strength_seed": strength_seed,
            "dgp3_strength_nonzero_count": None,
            "dgp3_strength_sum_squares": None,
            "dgp3_strength_vector": "",
        }
    strengths = pd.Series(group_strengths, dtype=float).to_numpy()
    return {
        "dgp3_strength_mode": strength_mode,
        "dgp3_strength_seed": strength_seed,
        "dgp3_strength_nonzero_count": int((strengths != 0.0).sum()),
        "dgp3_strength_sum_squares": float(strengths @ strengths),
        "dgp3_strength_vector": _format_strength_vector(strengths),
    }


def _format_strength_vector(group_strengths: Any) -> str:
    strengths = pd.Series(group_strengths, dtype=float).to_numpy()
    return ",".join(f"{value:.17g}" for value in strengths)


def parse_strength_vector(value: str) -> Any:
    if value == "":
        return pd.Series([], dtype=float).to_numpy()
    return pd.Series([float(part) for part in value.split(",")], dtype=float).to_numpy()


def paper_comparison_checks(
    comparison: pd.DataFrame,
    relative_tolerance: float,
    *,
    expected_matches: int | None = None,
) -> pd.DataFrame:
    if comparison.empty:
        return pd.DataFrame(
            [{"check": "paper_targets_matched", "passed": False, "detail": "no matches"}]
        )
    rows = []
    for metric, metric_comparison in comparison.groupby("metric", sort=False):
        max_abs_relative_error = float(metric_comparison["relative_error"].abs().max())
        rows.append(
            {
                "check": f"{metric}_within_relative_tolerance",
                "passed": bool(max_abs_relative_error <= relative_tolerance),
                "detail": (
                    f"max absolute relative error={max_abs_relative_error:.6g}; "
                    f"tolerance={relative_tolerance:.6g}"
                ),
            }
        )
    rows.append(
        {
            "check": "paper_targets_matched",
            "passed": expected_matches is None or len(comparison) == expected_matches,
            "detail": (
                f"matched rows={len(comparison)}"
                if expected_matches is None
                else f"matched rows={len(comparison)}; expected={expected_matches}"
            ),
        }
    )
    return pd.DataFrame(rows)


def expected_paper_target_matches(configs: list[dict[str, Any]]) -> int:
    if not configs:
        return 0
    targets = paper_table_targets()
    config_frame = pd.DataFrame(configs)
    key = [
        "source_table",
        "dgp",
        "error_distribution",
        "n_groups",
        "n_per_group",
        "strong_fraction",
        "weak_fraction",
    ]
    expected = config_frame[key].drop_duplicates().merge(
        targets,
        on=key,
        how="inner",
        validate="one_to_many",
    )
    return int(len(expected))


def render_report(
    args: argparse.Namespace,
    configs: list[dict[str, Any]],
    comparison: pd.DataFrame,
    checks: pd.DataFrame,
) -> str:
    lines = [
        "# adaptiveiv Paper Table Validation Report",
        "",
        "## Settings",
        "",
        f"- Preset: {args.preset}",
        f"- Repetitions per configuration: {_repetitions(args)}",
        f"- Split repetitions per fit: {args.n_splits}",
        f"- Configurations run: {len(configs)}",
        f"- Base seed: {args.seed}",
        f"- Relative tolerance: {_relative_tolerance(args):.6g}",
        *_dgp3_strength_setting_lines(args),
        "",
        "## Checks",
        "",
        _markdown_table(checks),
        "",
        "## Largest Absolute Relative Deviations",
        "",
        _markdown_table(_largest_deviations(comparison)),
        "",
        "## Scope limits",
        "",
        "- Targets are transcribed from Abadie, Gu, and Shen Tables 2-4.",
        "- Comparisons cover transcribed targets for implemented methods: "
        "2SLS-P, 2SLS-INT, 2SLS-SSINT, 2SLS-INF, 2SLS-ADPT, and LIML-INT "
        "MAD. LIML-INT MSE is not compared because the paper does not report "
        "LIML moments.",
        "- Rejection-rate targets are not compared here because broad inference "
        "validation is not yet implemented for every benchmark estimator.",
        "- Small presets are smoke checks. The full preset matches the paper's "
        "500 simulation repetitions and is the appropriate release evidence.",
        "",
    ]
    return "\n".join(lines)


def _dgp3_strength_setting_lines(args: argparse.Namespace) -> list[str]:
    lines = []
    if getattr(args, "redraw_dgp3_strengths", False):
        lines.append("- DGP3 strength mode: redraw per replication")
    else:
        lines.append("- DGP3 strength mode: fixed within configuration")
    if getattr(args, "dgp3_strength_seed_base", None) is not None:
        lines.append(f"- DGP3 strength seed base: {args.dgp3_strength_seed_base}")
    if getattr(args, "dgp3_strength_seed", None) is not None:
        lines.append(f"- DGP3 exact strength seed: {args.dgp3_strength_seed}")
    seed_map = getattr(args, "dgp3_strength_seed_map", None)
    if seed_map is not None and str(seed_map).strip() != "":
        lines.append(f"- DGP3 strength seed map: {seed_map}")
    return lines


def _largest_deviations(comparison: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    tail_columns = [
        "max_abs_error",
        "q95_abs_error",
        "q99_abs_error",
        "max_scaled_sq_error",
        "top_abs_error_mse_share",
        "paper_implied_single_tail_abs_error",
        "paper_implied_single_tail_scaled_sq_error",
        "paper_implied_tail_to_observed_max_ratio",
        "paper_implied_single_tail_mse_share",
    ]
    display_columns = [
        "source_table",
        "dgp",
        "error_distribution",
        "n_groups",
        "strong_fraction",
        "weak_fraction",
        "method",
        "metric",
        "observed_value",
        "paper_value",
        "relative_error",
    ]
    display_columns.extend(column for column in tail_columns if column in comparison)
    return (
        comparison.assign(abs_relative_error=comparison["relative_error"].abs())
        .sort_values("abs_relative_error", ascending=False)
        .head(limit)[display_columns]
    )


def _scenario_name(config: dict[str, Any]) -> str:
    return (
        f"{config['dgp']}_{config['error_distribution']}_"
        f"g{config['n_groups']}_ps{config['strong_fraction']}_"
        f"pw{config['weak_fraction']}"
    )


def _repetitions(args: argparse.Namespace) -> int:
    if args.repetitions is not None:
        return args.repetitions
    if args.preset == "full":
        return 500
    if args.preset == "release":
        return 100
    return 2


def _relative_tolerance(args: argparse.Namespace) -> float:
    if args.relative_tolerance is not None:
        return args.relative_tolerance
    if args.preset == "full":
        return 0.25
    if args.preset == "release":
        return 0.75
    return 10.0


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
