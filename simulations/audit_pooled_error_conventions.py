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

ERROR_CONVENTIONS: tuple[str, ...] = (
    "centered",
    "raw",
    "standardized",
    "raw_standardized",
)


def main() -> None:
    args = parse_args()
    configs = _selected_configs(args)
    rows = []
    for config in configs:
        rows.extend(
            audit_pooled_error_conventions(
                config,
                seed_base=args.seed,
                repetitions=args.repetitions,
                conventions=args.convention or ERROR_CONVENTIONS,
            ).to_dict(orient="records")
        )
    results = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "pooled_error_convention_audit.csv", index=False)
    (output_dir / "report.md").write_text(render_report(results), encoding="utf-8")
    print(f"Pooled error convention audit report: {output_dir / 'report.md'}")
    print(f"Rows: {len(results)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit pooled 2SLS sensitivity to chi-square error conventions."
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
        choices=ERROR_CONVENTIONS,
        action="append",
        help="Error convention to audit. Defaults to all conventions.",
    )
    parser.add_argument(
        "--output-dir",
        default="validation/outputs/paper_tables/pooled_error_convention_audit",
    )
    return parser.parse_args()


def audit_pooled_error_conventions(
    config: dict[str, Any],
    *,
    seed_base: int,
    repetitions: int,
    conventions: tuple[str, ...] | list[str] = ERROR_CONVENTIONS,
) -> pd.DataFrame:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    if config["dgp"] not in {"dgp1", "dgp2"}:
        raise ValueError("pooled error convention audit supports DGP1 and DGP2 only")
    if config["error_distribution"] != "chisq3":
        raise ValueError("pooled error convention audit is for chi-square rows only")

    rows = []
    nobs = int(config["n_groups"]) * int(config["n_per_group"])
    paper_value = _paper_pooled_scaled_mse(config)
    for convention in conventions:
        diagnostics = [
            pooled_error_convention_diagnostic(
                config,
                seed_base + 100_000 * int(config["config_index"]) + replication,
                convention=convention,
            )
            for replication in range(repetitions)
        ]
        frame = pd.DataFrame(diagnostics)
        beta_hat = frame["beta_hat"].to_numpy(dtype=float)
        scaled_mse = float(nobs * np.mean(beta_hat**2))
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
                "error_convention": convention,
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
    results = pd.DataFrame(rows)
    results["abs_relative_error"] = results["relative_error"].abs()
    results = results.sort_values(
        ["config_index", "abs_relative_error", "error_convention"],
        ignore_index=True,
    )
    results["rank"] = results.groupby("config_index").cumcount() + 1
    return results.drop(columns=["abs_relative_error"])


def pooled_error_convention_diagnostic(
    config: dict[str, Any],
    seed: int,
    *,
    convention: str,
) -> dict[str, Any]:
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
        "error_convention": convention,
        "beta_hat": float(beta_hat),
        "abs_beta_hat": float(abs(beta_hat)),
        "pooled_numerator": float(numerator),
        "pooled_denominator": float(denominator),
    }


def render_report(results: pd.DataFrame) -> str:
    lines = [
        "# adaptiveiv Pooled Error Convention Audit",
        "",
        "This diagnostic checks whether the remaining pooled chi-square paper-table",
        "MSE deviations are explained by centering or scaling conventions for",
        "chi-square errors. It does not mutate validation outputs.",
        "",
        "## Convention Summary",
        "",
        _markdown_table(results),
        "",
        "## Interpretation",
        "",
        "- `centered` uses `chi2(3) - 3`, the package's maintained convention.",
        "- `raw` uses uncentered `chi2(3)` draws. With an intercept in pooled 2SLS,",
        "  raw and centered errors should coincide up to numerical precision.",
        "- `standardized` uses `(chi2(3) - 3) / sqrt(6)`.",
        "- `raw_standardized` uses `chi2(3) / sqrt(6)`.",
        "- This is a pooled-only diagnostic for the heavy-tail paper-table blocker.",
        "",
    ]
    return "\n".join(lines)


def _selected_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = [
        {**config, "config_index": index}
        for index, config in enumerate(section4_paper_configurations())
        if config["error_distribution"] == "chisq3" and config["dgp"] in {"dgp1", "dgp2"}
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


def _strengths_for_config(config: dict[str, Any], seed: int) -> np.ndarray:
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
        raise ValueError("pooled error convention audit supports DGP1 and DGP2 only")
    np.random.default_rng(seed).shuffle(strengths)
    return strengths


def _pooled_iv_cross_products(
    config: dict[str, Any],
    seed: int,
    convention: str,
) -> tuple[float, float, float]:
    if convention not in ERROR_CONVENTIONS:
        raise ValueError(f"Unknown error convention: {convention}")

    n_per_group = int(config["n_per_group"])
    rng = np.random.default_rng(seed)
    strengths = _strengths_for_config(config, seed)
    sqrt_term = float(np.sqrt(1.0 - 0.25**2))

    sum_x = sum_xx = 0.0
    sum_z = sum_w = sum_y = 0.0
    sum_zx = sum_wx = sum_yx = 0.0
    sum_zw = sum_zy = 0.0
    for rho_g in strengths:
        z = rng.normal(size=n_per_group)
        x = rng.normal(size=n_per_group)
        v = _draw_error(rng, n_per_group, convention)
        e = _draw_error(rng, n_per_group, convention)
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

    nobs = int(config["n_groups"]) * n_per_group
    design_cross = np.array([[nobs, sum_x], [sum_x, sum_xx]], dtype=float)
    inv_design_cross = np.linalg.pinv(design_cross)

    def residual_cross(a_sum: float, ax_sum: float, b_sum: float, bx_sum: float) -> float:
        a_design = np.array([a_sum, ax_sum], dtype=float)
        b_design = np.array([b_sum, bx_sum], dtype=float)
        return -float(a_design @ inv_design_cross @ b_design)

    z_w = sum_zw + residual_cross(sum_z, sum_zx, sum_w, sum_wx)
    z_y = sum_zy + residual_cross(sum_z, sum_zx, sum_y, sum_yx)
    beta_hat = z_y / z_w
    return float(beta_hat), float(z_w), float(z_y)


def _draw_error(
    rng: np.random.Generator,
    size: int,
    convention: str,
) -> np.ndarray:
    draws = rng.chisquare(df=3, size=size)
    if convention == "centered":
        return draws - 3.0
    if convention == "raw":
        return draws
    if convention == "standardized":
        return (draws - 3.0) / np.sqrt(6.0)
    if convention == "raw_standardized":
        return draws / np.sqrt(6.0)
    raise ValueError(f"Unknown error convention: {convention}")


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
