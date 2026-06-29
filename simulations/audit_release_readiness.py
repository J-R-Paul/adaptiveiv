from __future__ import annotations

# ruff: noqa: E402

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from adaptiveiv.paper_benchmarks import paper_method_coverage, paper_table_targets


@dataclass(frozen=True)
class ReleaseReadinessAudit:
    ready: bool
    checks: pd.DataFrame

    @property
    def failed_gates(self) -> list[str]:
        required = self.checks["required"].astype(bool)
        failed = self.checks.loc[required & ~self.checks["passed"], "gate"].tolist()
        return [str(gate) for gate in failed]


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    audit = assess_release_readiness(
        project_root=project_root,
        replication_dir=Path(args.replication_dir),
        inference_dir=Path(args.inference_dir),
        paper_tables_dir=Path(args.paper_tables_dir),
        reconstructed_paper_tables_dir=Path(args.reconstructed_paper_tables_dir),
        dist_dir=Path(args.dist_dir),
    )
    write_release_audit(audit, output_dir)
    print(f"Release audit report: {output_dir / 'report.md'}")
    print(f"Ready: {audit.ready}")
    if audit.failed_gates:
        print(f"Failed gates: {', '.join(audit.failed_gates)}")
    if not audit.ready and not args.no_fail:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit adaptiveiv release-readiness artifacts."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument(
        "--replication-dir",
        default="validation/outputs/latest",
        help="Directory containing qualitative replication checks.csv.",
    )
    parser.add_argument(
        "--inference-dir",
        default="validation/outputs/inference",
        help="Directory containing inference validation checks.csv.",
    )
    parser.add_argument(
        "--paper-tables-dir",
        default="validation/outputs/paper_tables/full_combined_reconstructed_dgp3",
        help="Directory containing full paper-table comparison checks.csv.",
    )
    parser.add_argument(
        "--reconstructed-paper-tables-dir",
        default=(
            "validation/outputs/paper_tables/"
            "full_combined_reconstructed_observation_seeds"
        ),
        help=(
            "Directory containing reconstructed observation-seed paper-table "
            "evidence checks.csv."
        ),
    )
    parser.add_argument(
        "--dist-dir",
        default="dist",
        help="Directory containing built sdist and wheel artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default="validation/outputs/release_audit",
        help="Directory for release audit outputs.",
    )
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="Write outputs and exit zero even when release gates fail.",
    )
    return parser.parse_args()


def assess_release_readiness(
    *,
    project_root: Path,
    replication_dir: Path | None = None,
    inference_dir: Path | None = None,
    paper_tables_dir: Path | None = None,
    reconstructed_paper_tables_dir: Path | None = None,
    dist_dir: Path | None = None,
) -> ReleaseReadinessAudit:
    root = Path(project_root)
    replication_dir = _resolve(root, replication_dir or Path("validation/outputs/latest"))
    inference_dir = _resolve(root, inference_dir or Path("validation/outputs/inference"))
    paper_tables_dir = _resolve(
        root,
        paper_tables_dir
        or Path("validation/outputs/paper_tables/full_combined_reconstructed_dgp3"),
    )
    reconstructed_paper_tables_dir = _resolve(
        root,
        reconstructed_paper_tables_dir
        or Path(
            "validation/outputs/paper_tables/"
            "full_combined_reconstructed_observation_seeds"
        ),
    )
    dist_dir = _resolve(root, dist_dir or Path("dist"))

    rows = [
        _checks_csv_gate(
            "replication_validation",
            replication_dir / "checks.csv",
            "qualitative replication checks",
            required=True,
        ),
        _checks_csv_gate(
            "inference_validation",
            inference_dir / "checks.csv",
            "paper_homoskedastic inference checks",
            required=True,
        ),
        _checks_csv_gate(
            "paper_table_validation",
            paper_tables_dir / "checks.csv",
            "non-blocking full paper-table numerical replication checks",
            required=False,
        ),
        _paper_table_artifact_freshness_gate(paper_tables_dir, required=False),
        _paper_method_coverage_gate(required=False),
        _paper_replication_limitation_gate(required=False),
        _checks_csv_gate(
            "reconstructed_paper_table_evidence",
            reconstructed_paper_tables_dir / "checks.csv",
            "supporting reconstructed observation-seed paper-table evidence",
            required=False,
        ),
        _dist_gate(dist_dir, required=True),
    ]
    checks = pd.DataFrame(rows)
    required = checks["required"].astype(bool)
    ready = bool(checks.loc[required, "passed"].all())
    return ReleaseReadinessAudit(ready=ready, checks=checks)


def write_release_audit(audit: ReleaseReadinessAudit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.checks.to_csv(output_dir / "checks.csv", index=False)
    (output_dir / "release_readiness.json").write_text(
        json.dumps(
            {
                "ready": audit.ready,
                "failed_gates": audit.failed_gates,
                "checks": audit.checks.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(render_report(audit), encoding="utf-8")


def render_report(audit: ReleaseReadinessAudit) -> str:
    lines = [
        "# adaptiveiv Release Readiness Audit",
        "",
        f"- Ready for public release: {audit.ready}",
        f"- Failed gates: {', '.join(audit.failed_gates) or 'none'}",
        "",
        "## Checks",
        "",
        _markdown_table(audit.checks),
        "",
    ]
    return "\n".join(lines)


def _checks_csv_gate(
    gate: str,
    path: Path,
    description: str,
    *,
    required: bool,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "gate": gate,
            "passed": False,
            "required": required,
            "description": description,
            "source": str(path),
            "detail": (
                f"missing required file: {path}"
                if required
                else f"missing optional evidence file: {path}"
            ),
        }
    checks = pd.read_csv(path)
    if "passed" not in checks.columns or "check" not in checks.columns:
        return {
            "gate": gate,
            "passed": False,
            "required": required,
            "description": description,
            "source": str(path),
            "detail": "checks.csv must contain check and passed columns",
        }
    passed = _passed_series(checks["passed"])
    failed = checks.loc[~passed]
    if failed.empty:
        detail = f"{len(checks)} checks passed"
    else:
        detail = "; ".join(
            _format_failed_check(row) for _, row in failed.iterrows()
        )
    return {
        "gate": gate,
        "passed": bool(failed.empty),
        "required": required,
        "description": description,
        "source": str(path),
        "detail": detail,
    }


def _paper_method_coverage_gate(*, required: bool) -> dict[str, Any]:
    coverage = paper_method_coverage()
    implemented = coverage["implemented"].astype(bool)
    targets = coverage["paper_table_targets_transcribed"].astype(bool)
    missing = coverage.loc[~implemented, "paper_method"].tolist()
    missing_targets = coverage.loc[implemented & ~targets, "paper_method"].tolist()
    passed = not missing and not missing_targets
    detail = (
        f"{int(implemented.sum())}/{len(coverage)} paper methods implemented; "
        f"{int(targets.sum())}/{len(coverage)} have transcribed table targets"
    )
    if missing:
        detail += "; missing implementations=" + ", ".join(
            str(method) for method in missing
        )
    if missing_targets:
        detail += "; missing targets=" + ", ".join(
            str(method) for method in missing_targets
        )
    return {
        "gate": "paper_method_coverage",
        "passed": passed,
        "required": required,
        "description": "full original-paper Section 4 estimator coverage",
        "source": "adaptiveiv.paper_benchmarks.paper_method_coverage",
        "detail": detail,
    }


def _paper_table_artifact_freshness_gate(
    paper_tables_dir: Path,
    *,
    required: bool,
) -> dict[str, Any]:
    path = paper_tables_dir / "paper_targets.csv"
    if not path.exists():
        return {
            "gate": "paper_table_artifact_freshness",
            "passed": False,
            "required": required,
            "description": "paper-table artifact target set matches current code",
            "source": str(path),
            "detail": (
                f"missing required file: {path}"
                if required
                else f"missing optional evidence file: {path}"
            ),
        }
    artifact_targets = pd.read_csv(path)
    current_target_count = len(paper_table_targets())
    artifact_target_count = len(artifact_targets)
    passed = artifact_target_count == current_target_count
    return {
        "gate": "paper_table_artifact_freshness",
        "passed": passed,
        "required": required,
        "description": "paper-table artifact target set matches current code",
        "source": str(path),
        "detail": (
            f"artifact rows={artifact_target_count}; "
            f"current target rows={current_target_count}"
        ),
    }


def _paper_replication_limitation_gate(*, required: bool) -> dict[str, Any]:
    return {
        "gate": "paper_replication_limitation",
        "passed": True,
        "required": required,
        "description": "known original-paper replication limitation",
        "source": "validation/README.md",
        "detail": (
            "Current package does not exactly replicate all original Tables 2-4; "
            "likely due to unrecovered simulation seeds/state and unimplemented "
            "external comparators."
        ),
    }


def _dist_gate(dist_dir: Path, *, required: bool) -> dict[str, Any]:
    if not dist_dir.exists():
        return {
            "gate": "distribution_artifacts",
            "passed": False,
            "required": required,
            "description": "built source distribution and wheel",
            "source": str(dist_dir),
            "detail": f"missing required directory: {dist_dir}",
        }
    sdists = sorted(dist_dir.glob("*.tar.gz"))
    wheels = sorted(dist_dir.glob("*.whl"))
    missing: list[str] = []
    if not sdists:
        missing.append("sdist (*.tar.gz)")
    if not wheels:
        missing.append("wheel (*.whl)")
    return {
        "gate": "distribution_artifacts",
        "passed": not missing,
        "required": required,
        "description": "built source distribution and wheel",
        "source": str(dist_dir),
        "detail": (
            f"sdist={sdists[-1].name}; wheel={wheels[-1].name}"
            if not missing
            else "missing " + ", ".join(missing)
        ),
    }


def _passed_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def _format_failed_check(row: pd.Series) -> str:
    check = str(row["check"])
    detail = str(row.get("detail", "")).strip()
    return f"{check}: {detail}" if detail else check


def _resolve(project_root: Path, path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root / path


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = [str(row[column]) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
