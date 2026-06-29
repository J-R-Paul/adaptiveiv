from __future__ import annotations

# ruff: noqa: E402

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from adaptiveiv.paper_benchmarks import (
    compare_summary_to_paper_targets,
    paper_table_targets,
)
from adaptiveiv.validation import summarize_simulation_results
from simulations.validate_paper_tables import (
    expected_paper_target_matches,
    _largest_deviations,
    _markdown_table,
    paper_comparison_checks,
)


def main() -> None:
    args = parse_args()
    checks = aggregate_paper_table_chunks(
        [Path(path) for path in args.chunk_dirs],
        Path(args.output_dir),
        expected_config_count=args.expected_config_count,
        relative_tolerance=args.relative_tolerance,
    )
    print(f"Combined paper-table report: {Path(args.output_dir) / 'report.md'}")
    print(f"Checks passed: {int(checks['passed'].sum())}/{len(checks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate chunked adaptiveiv paper-table validation outputs."
    )
    parser.add_argument("chunk_dirs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-config-count", type=int, default=30)
    parser.add_argument("--relative-tolerance", type=float, default=0.25)
    return parser.parse_args()


def aggregate_paper_table_chunks(
    chunk_dirs: list[Path],
    output_dir: Path,
    *,
    expected_config_count: int = 30,
    relative_tolerance: float = 0.25,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    results = []
    for chunk_dir in chunk_dirs:
        manifests.append(_read_required_csv(chunk_dir / "config_manifest.csv"))
        results.append(_read_required_csv(chunk_dir / "simulation_results.csv"))

    manifest = pd.concat(manifests, ignore_index=True).drop_duplicates()
    simulation_results = pd.concat(results, ignore_index=True)
    summary = summarize_simulation_results(simulation_results)
    comparison = compare_summary_to_paper_targets(summary)
    checks = _combined_checks(
        manifest,
        comparison,
        expected_config_count=expected_config_count,
        relative_tolerance=relative_tolerance,
    )
    targets = paper_table_targets()

    targets.to_csv(output_dir / "paper_targets.csv", index=False)
    manifest.to_csv(output_dir / "config_manifest.csv", index=False)
    simulation_results.to_csv(output_dir / "simulation_results.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    comparison.to_csv(output_dir / "paper_comparison.csv", index=False)
    checks.to_csv(output_dir / "checks.csv", index=False)
    (output_dir / "report.md").write_text(
        render_combined_report(
            chunk_dirs,
            manifest,
            comparison,
            checks,
            expected_config_count=expected_config_count,
            relative_tolerance=relative_tolerance,
        ),
        encoding="utf-8",
    )
    return checks


def render_combined_report(
    chunk_dirs: list[Path],
    manifest: pd.DataFrame,
    comparison: pd.DataFrame,
    checks: pd.DataFrame,
    *,
    expected_config_count: int,
    relative_tolerance: float,
) -> str:
    lines = [
        "# adaptiveiv Combined Paper Table Validation Report",
        "",
        "## Settings",
        "",
        f"- Chunk directories: {len(chunk_dirs)}",
        f"- Configurations observed: {manifest['config_index'].nunique()}",
        f"- Expected configurations: {expected_config_count}",
        f"- Matched comparison rows: {len(comparison)}",
        f"- Relative tolerance: {relative_tolerance:.6g}",
        "",
        "## Checks",
        "",
        _markdown_table(checks),
        "",
        "## Largest Absolute Relative Deviations",
        "",
        _markdown_table(_largest_deviations(comparison)),
        "",
    ]
    return "\n".join(lines)


def _combined_checks(
    manifest: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    expected_config_count: int,
    relative_tolerance: float,
) -> pd.DataFrame:
    checks = paper_comparison_checks(
        comparison,
        relative_tolerance,
        expected_matches=expected_paper_target_matches(
            manifest.to_dict(orient="records")
        ),
    )
    observed = sorted(manifest["config_index"].astype(int).unique().tolist())
    expected = list(range(expected_config_count))
    missing = [index for index in expected if index not in observed]
    unexpected = [index for index in observed if index not in expected]
    coverage = pd.DataFrame(
        [
            {
                "check": "config_coverage",
                "passed": not missing and not unexpected,
                "detail": (
                    f"observed={observed}; missing={missing}; unexpected={unexpected}"
                ),
            }
        ]
    )
    return pd.concat([checks, coverage], ignore_index=True)


def _read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
