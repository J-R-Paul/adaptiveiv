from __future__ import annotations

# ruff: noqa: E402

import argparse
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from adaptiveiv.paper_benchmarks import section4_paper_configurations
from adaptiveiv.simulation import paper_group_strengths


def main() -> None:
    args = parse_args()
    configs = section4_paper_configurations()
    config = {**configs[args.config_index], "config_index": args.config_index}
    rng = np.random.default_rng(args.random_seed)
    seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=args.seed_count,
        dtype=np.uint64,
    )
    scan, counts = scan_pooled_tail_seeds(
        config,
        [int(seed) for seed in seeds],
        top_k=args.top_k,
        thresholds=args.threshold,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scan.to_csv(output_dir / "pooled_tail_seed_scan.csv", index=False)
    pd.DataFrame([counts]).to_csv(output_dir / "pooled_tail_seed_counts.csv", index=False)
    (output_dir / "report.md").write_text(
        render_report(config, scan, counts),
        encoding="utf-8",
    )
    print(f"Pooled-tail seed diagnostic report: {output_dir / 'report.md'}")
    print(f"Seeds scanned: {counts['seed_count']}")
    print(f"Maximum |beta_hat|: {counts['max_abs_beta_hat']:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast seed scan for pooled 2SLS tail events in paper DGPs."
    )
    parser.add_argument("--config-index", type=int, default=19)
    parser.add_argument("--seed-count", type=int, default=1_000)
    parser.add_argument("--random-seed", type=int, default=20260623)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=[10.0, 100.0, 500.0, 1000.0],
        help="Absolute beta threshold to count. May be supplied repeatedly.",
    )
    parser.add_argument(
        "--output-dir",
        default="validation/outputs/paper_tables/pooled_tail_seed_scan",
    )
    return parser.parse_args()


def scan_pooled_tail_seeds(
    config: dict[str, Any],
    seeds: Iterable[int],
    *,
    top_k: int = 20,
    thresholds: Iterable[float] = (10.0, 100.0, 500.0, 1000.0),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    diagnostics = [
        pooled_tail_seed_diagnostic(config, int(seed))
        for seed in seeds
    ]
    frame = pd.DataFrame(diagnostics)
    if frame.empty:
        return frame, {"seed_count": 0, "max_abs_beta_hat": np.nan}
    frame = frame.sort_values("abs_beta_hat", ascending=False, ignore_index=True)
    counts: dict[str, Any] = {
        "seed_count": int(len(frame)),
        "max_abs_beta_hat": float(frame["abs_beta_hat"].max()),
        "max_abs_beta_seed": int(frame.loc[frame["abs_beta_hat"].idxmax(), "seed"]),
    }
    for threshold in thresholds:
        threshold_value = float(threshold)
        key = f"abs_beta_gt_{threshold_value:g}"
        counts[key] = int((frame["abs_beta_hat"] > threshold_value).sum())
    return frame.head(top_k).copy(), counts


def pooled_tail_seed_diagnostic(
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    beta_hat, denominator, numerator = _pooled_iv_cross_products(config, seed)
    return {
        "config_index": int(config.get("config_index", -1)),
        "source_table": config["source_table"],
        "dgp": config["dgp"],
        "error_distribution": config["error_distribution"],
        "n_groups": int(config["n_groups"]),
        "n_per_group": int(config["n_per_group"]),
        "strong_fraction": float(config["strong_fraction"]),
        "weak_fraction": float(config["weak_fraction"]),
        "seed": int(seed),
        "beta_hat": float(beta_hat),
        "abs_beta_hat": float(abs(beta_hat)),
        "pooled_numerator": float(numerator),
        "pooled_denominator": float(denominator),
    }


def render_report(
    config: dict[str, Any],
    scan: pd.DataFrame,
    counts: dict[str, Any],
) -> str:
    lines = [
        "# adaptiveiv Pooled Tail Seed Diagnostic",
        "",
        "## Configuration",
        "",
        f"- Config index: {config.get('config_index', 'n/a')}",
        f"- Source table: {config['source_table']}",
        f"- DGP: {config['dgp']}",
        f"- Error distribution: {config['error_distribution']}",
        f"- Groups: {config['n_groups']}",
        f"- Group size: {config['n_per_group']}",
        f"- Strong fraction: {config['strong_fraction']}",
        f"- Weak fraction: {config['weak_fraction']}",
        "",
        "## Counts",
        "",
        _markdown_table(pd.DataFrame([counts])),
        "",
        "## Largest Absolute Pooled Estimates",
        "",
        _markdown_table(scan),
        "",
    ]
    return "\n".join(lines)


def _pooled_iv_cross_products(
    config: dict[str, Any],
    seed: int,
) -> tuple[float, float, float]:
    n_groups = int(config["n_groups"])
    n_per_group = int(config["n_per_group"])
    nobs = n_groups * n_per_group
    strengths = paper_group_strengths(
        config["dgp"],
        n_groups=n_groups,
        strong_fraction=float(config["strong_fraction"]),
        weak_fraction=float(config["weak_fraction"]),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
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


def _draw_centered_error(
    rng: np.random.Generator,
    size: int,
    error_distribution: str,
) -> np.ndarray:
    if error_distribution == "normal":
        return rng.normal(size=size)
    if error_distribution in {"chisq3", "chi2", "chisquare3"}:
        return rng.chisquare(df=3, size=size) - 3.0
    raise ValueError("error_distribution must be 'normal' or 'chisq3'")


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
