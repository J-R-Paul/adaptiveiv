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

from adaptiveiv.paper_benchmarks import paper_table_targets, section4_paper_configurations
from simulations.diagnose_pooled_tail_seeds import _draw_centered_error

CONVENTIONS: tuple[str, ...] = (
    "shuffled_separate",
    "fixed_order",
    "shuffled_coupled",
)


def main() -> None:
    args = parse_args()
    configs = _selected_configs(args)
    rows = []
    for config in configs:
        rows.extend(
            audit_pooled_conventions(
                config,
                seed_base=args.seed,
                repetitions=args.repetitions,
                conventions=args.convention or CONVENTIONS,
            ).to_dict(orient="records")
        )
    results = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "pooled_rng_convention_audit.csv", index=False)
    (output_dir / "report.md").write_text(render_report(results), encoding="utf-8")
    print(f"Pooled RNG convention audit report: {output_dir / 'report.md'}")
    print(f"Rows: {len(results)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit pooled 2SLS sensitivity to DGP1/DGP2 RNG conventions."
    )
    parser.add_argument(
        "--config-index",
        type=int,
        action="append",
        help="Paper configuration index to audit. May be supplied repeatedly.",
    )
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument(
        "--convention",
        choices=CONVENTIONS,
        action="append",
        help="Convention to audit. Defaults to all conventions.",
    )
    parser.add_argument(
        "--output-dir",
        default="validation/outputs/paper_tables/pooled_rng_convention_audit",
    )
    return parser.parse_args()


def audit_pooled_conventions(
    config: dict[str, Any],
    *,
    seed_base: int,
    repetitions: int,
    conventions: tuple[str, ...] | list[str] = CONVENTIONS,
) -> pd.DataFrame:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    rows = []
    nobs = int(config["n_groups"]) * int(config["n_per_group"])
    paper_value = _paper_pooled_scaled_mse(config)
    for convention in conventions:
        diagnostics = [
            pooled_convention_diagnostic(
                config,
                seed_base + 100_000 * int(config["config_index"]) + replication,
                convention=convention,
            )
            for replication in range(repetitions)
        ]
        frame = pd.DataFrame(diagnostics)
        scaled_mse = float(nobs * np.mean(frame["beta_hat"].to_numpy(dtype=float) ** 2))
        max_index = frame["abs_beta_hat"].idxmax()
        max_row = frame.loc[max_index]
        rows.append(
            {
                "config_index": int(config["config_index"]),
                "source_table": config["source_table"],
                "dgp": config["dgp"],
                "error_distribution": config["error_distribution"],
                "n_groups": int(config["n_groups"]),
                "n_per_group": int(config["n_per_group"]),
                "strong_fraction": float(config["strong_fraction"]),
                "weak_fraction": float(config["weak_fraction"]),
                "convention": convention,
                "repetitions": repetitions,
                "scaled_mse": scaled_mse,
                "paper_value": paper_value,
                "relative_error": (scaled_mse - paper_value) / abs(paper_value),
                "max_abs_beta_hat": float(max_row["abs_beta_hat"]),
                "max_abs_beta_seed": int(max_row["seed"]),
                "max_abs_beta_replication": int(max_row["replication"]),
                "max_abs_beta_denominator": float(max_row["pooled_denominator"]),
            }
        )
    return pd.DataFrame(rows)


def pooled_convention_diagnostic(
    config: dict[str, Any],
    seed: int,
    *,
    convention: str,
) -> dict[str, Any]:
    if convention not in CONVENTIONS:
        raise ValueError(f"Unknown convention: {convention}")
    beta_hat, denominator, numerator = _pooled_iv_cross_products(
        config,
        seed,
        convention,
    )
    return {
        "config_index": int(config.get("config_index", -1)),
        "seed": int(seed),
        "replication": int(seed - 20260623 - 100_000 * int(config["config_index"]))
        if "config_index" in config
        else -1,
        "convention": convention,
        "beta_hat": float(beta_hat),
        "abs_beta_hat": float(abs(beta_hat)),
        "pooled_numerator": float(numerator),
        "pooled_denominator": float(denominator),
    }


def render_report(results: pd.DataFrame) -> str:
    lines = [
        "# adaptiveiv Pooled RNG Convention Audit",
        "",
        "## Convention Summary",
        "",
        _markdown_table(results),
        "",
        "## Interpretation",
        "",
        "- `shuffled_separate` is the current package convention: group strengths",
        "  are shuffled using a strength RNG initialized with the dataset seed, while",
        "  observation-level draws start from the same seed in a separate RNG.",
        "- `fixed_order` leaves strong/weak groups in deterministic group order.",
        "- `shuffled_coupled` consumes the group-strength shuffle on the same RNG",
        "  stream used for observation-level draws.",
        "- This is a pooled-only diagnostic; it does not replace full paper-table",
        "  validation.",
        "",
    ]
    return "\n".join(lines)


def _selected_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = [
        {**config, "config_index": index}
        for index, config in enumerate(section4_paper_configurations())
    ]
    if args.config_index is None:
        return configs
    wanted = set(args.config_index)
    return [config for config in configs if int(config["config_index"]) in wanted]


def _paper_pooled_scaled_mse(config: dict[str, Any]) -> float:
    targets = paper_table_targets()
    row = targets.loc[
        targets["dgp"].eq(config["dgp"])
        & targets["error_distribution"].eq(config["error_distribution"])
        & targets["n_groups"].eq(int(config["n_groups"]))
        & targets["n_per_group"].eq(int(config["n_per_group"]))
        & targets["strong_fraction"].eq(float(config["strong_fraction"]))
        & targets["weak_fraction"].eq(float(config["weak_fraction"]))
        & targets["method"].eq("pooled")
        & targets["metric"].eq("scaled_mse")
    ]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["paper_value"])


def _strengths_for_convention(
    config: dict[str, Any],
    seed: int,
    convention: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n_groups = int(config["n_groups"])
    n_strong = int(round(n_groups * float(config["strong_fraction"])))
    n_weak = int(round(n_groups * float(config["weak_fraction"])))
    strengths = np.zeros(n_groups, dtype=float)
    if config["dgp"] == "dgp1":
        strengths[:n_strong] = 1.0
    elif config["dgp"] == "dgp2":
        strengths[:n_strong] = 1.0
        strengths[n_strong : n_strong + n_weak] = 0.2
    else:
        raise ValueError("pooled RNG convention audit supports DGP1 and DGP2 only")

    if convention == "fixed_order":
        return strengths
    if convention == "shuffled_coupled":
        rng.shuffle(strengths)
        return strengths
    strength_rng = np.random.default_rng(seed)
    strength_rng.shuffle(strengths)
    return strengths


def _pooled_iv_cross_products(
    config: dict[str, Any],
    seed: int,
    convention: str,
) -> tuple[float, float, float]:
    n_groups = int(config["n_groups"])
    n_per_group = int(config["n_per_group"])
    nobs = n_groups * n_per_group
    rng = np.random.default_rng(seed)
    strengths = _strengths_for_convention(config, seed, convention, rng)
    error_key = str(config["error_distribution"]).lower().replace("_", "")
    sqrt_term = float(np.sqrt(1.0 - 0.25**2))

    sum_x = sum_xx = 0.0
    sum_z = sum_w = sum_y = 0.0
    sum_zx = sum_wx = sum_yx = 0.0
    sum_zw = sum_zy = 0.0
    for rho_g in strengths:
        z = rng.normal(size=n_per_group)
        x = rng.normal(size=n_per_group)
        v = _draw_centered_error(rng, n_per_group, error_key)
        e = _draw_centered_error(rng, n_per_group, error_key)
        u = 0.25 * v + sqrt_term * e
        w = rho_g * z + x + v
        y = x + u

        sum_x += float(x.sum())
        sum_xx += float(x @ x)
        sum_z += float(z.sum())
        sum_w += float(w.sum())
        sum_y += float(y.sum())
        sum_zx += float(z @ x)
        sum_wx += float(w @ x)
        sum_yx += float(y @ x)
        sum_zw += float(z @ w)
        sum_zy += float(z @ y)

    controls_cross = np.array([[nobs, sum_x], [sum_x, sum_xx]], dtype=float)
    controls_inv = np.linalg.inv(controls_cross)
    z_controls = np.array([sum_z, sum_zx], dtype=float)
    w_controls = np.array([sum_w, sum_wx], dtype=float)
    y_controls = np.array([sum_y, sum_yx], dtype=float)
    denominator = sum_zw - float(z_controls @ controls_inv @ w_controls)
    numerator = sum_zy - float(z_controls @ controls_inv @ y_controls)
    return float(numerator / denominator), float(denominator), float(numerator)


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
