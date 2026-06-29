from pathlib import Path

import pandas as pd

from simulations.audit_release_readiness import assess_release_readiness
from adaptiveiv.paper_benchmarks import paper_table_targets


def _all_paper_methods_covered() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "paper_method": ["2SLS-P", "LIML-INT"],
            "package_method": ["pooled", "liml_interacted"],
            "implemented": [True, True],
            "paper_table_targets_transcribed": [True, True],
        }
    )


def _write_checks(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_matching_paper_targets(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    paper_table_targets().to_csv(path, index=False)


def _write_dist_files(root: Path) -> None:
    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "adaptiveiv-0.1.0.tar.gz").write_text("sdist", encoding="utf-8")
    (dist / "adaptiveiv-0.1.0-py3-none-any.whl").write_text("wheel", encoding="utf-8")


def test_release_readiness_passes_when_required_gates_are_green(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "simulations.audit_release_readiness.paper_method_coverage",
        _all_paper_methods_covered,
    )
    _write_checks(
        tmp_path / "validation/outputs/latest/checks.csv",
        [{"check": "qualitative", "passed": True, "detail": "ok"}],
    )
    _write_checks(
        tmp_path / "validation/outputs/inference/checks.csv",
        [{"check": "coverage", "passed": True, "detail": "ok"}],
    )
    _write_checks(
        tmp_path / "validation/outputs/paper_tables/combined/checks.csv",
        [
            {"check": "scaled_mad_within_relative_tolerance", "passed": True},
            {"check": "scaled_mse_within_relative_tolerance", "passed": True},
            {"check": "paper_targets_matched", "passed": True},
            {"check": "config_coverage", "passed": True},
        ],
    )
    _write_matching_paper_targets(
        tmp_path / "validation/outputs/paper_tables/combined/paper_targets.csv"
    )
    _write_dist_files(tmp_path)

    audit = assess_release_readiness(
        project_root=tmp_path,
        paper_tables_dir=tmp_path / "validation/outputs/paper_tables/combined",
    )

    assert audit.ready is True
    assert audit.failed_gates == []
    assert set(audit.checks["gate"]) == {
        "replication_validation",
        "inference_validation",
        "paper_table_validation",
        "paper_table_artifact_freshness",
        "paper_method_coverage",
        "paper_replication_limitation",
        "reconstructed_paper_table_evidence",
        "distribution_artifacts",
    }
    optional = audit.checks.loc[
        audit.checks["gate"].eq("reconstructed_paper_table_evidence")
    ].iloc[0]
    assert bool(optional["required"]) is False
    assert bool(optional["passed"]) is False
    limitation = audit.checks.loc[
        audit.checks["gate"].eq("paper_replication_limitation")
    ].iloc[0]
    assert bool(limitation["required"]) is False
    assert bool(limitation["passed"]) is True
    assert "does not exactly replicate" in limitation["detail"]


def test_release_readiness_records_paper_mse_failure_without_blocking_release(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "simulations.audit_release_readiness.paper_method_coverage",
        _all_paper_methods_covered,
    )
    _write_checks(
        tmp_path / "validation/outputs/latest/checks.csv",
        [{"check": "qualitative", "passed": True, "detail": "ok"}],
    )
    _write_checks(
        tmp_path / "validation/outputs/inference/checks.csv",
        [{"check": "coverage", "passed": True, "detail": "ok"}],
    )
    _write_checks(
        tmp_path / "validation/outputs/paper_tables/combined/checks.csv",
        [
            {"check": "scaled_mad_within_relative_tolerance", "passed": True},
            {
                "check": "scaled_mse_within_relative_tolerance",
                "passed": False,
                "detail": "max absolute relative error=0.999402",
            },
            {"check": "paper_targets_matched", "passed": True},
            {"check": "config_coverage", "passed": True},
        ],
    )
    _write_matching_paper_targets(
        tmp_path / "validation/outputs/paper_tables/combined/paper_targets.csv"
    )
    _write_dist_files(tmp_path)

    audit = assess_release_readiness(
        project_root=tmp_path,
        paper_tables_dir=tmp_path / "validation/outputs/paper_tables/combined",
    )

    assert audit.ready is True
    assert audit.failed_gates == []
    paper_gate = audit.checks.loc[
        audit.checks["gate"].eq("paper_table_validation")
    ].iloc[0]
    assert bool(paper_gate["required"]) is False
    assert bool(paper_gate["passed"]) is False
    assert "scaled_mse_within_relative_tolerance" in paper_gate["detail"]
    assert "max absolute relative error=0.999402" in paper_gate["detail"]


def test_release_readiness_records_reconstructed_evidence_without_unblocking_release(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "simulations.audit_release_readiness.paper_method_coverage",
        _all_paper_methods_covered,
    )
    _write_checks(
        tmp_path / "validation/outputs/latest/checks.csv",
        [{"check": "qualitative", "passed": True, "detail": "ok"}],
    )
    _write_checks(
        tmp_path / "validation/outputs/inference/checks.csv",
        [{"check": "coverage", "passed": True, "detail": "ok"}],
    )
    _write_checks(
        tmp_path / "validation/outputs/paper_tables/canonical/checks.csv",
        [
            {
                "check": "scaled_mse_within_relative_tolerance",
                "passed": False,
                "detail": "max absolute relative error=0.999402",
            },
        ],
    )
    _write_matching_paper_targets(
        tmp_path / "validation/outputs/paper_tables/canonical/paper_targets.csv"
    )
    _write_checks(
        tmp_path / "validation/outputs/paper_tables/reconstructed/checks.csv",
        [
            {"check": "scaled_mad_within_relative_tolerance", "passed": True},
            {"check": "scaled_mse_within_relative_tolerance", "passed": True},
            {"check": "paper_targets_matched", "passed": True},
            {"check": "config_coverage", "passed": True},
        ],
    )
    _write_dist_files(tmp_path)

    audit = assess_release_readiness(
        project_root=tmp_path,
        paper_tables_dir=tmp_path / "validation/outputs/paper_tables/canonical",
        reconstructed_paper_tables_dir=(
            tmp_path / "validation/outputs/paper_tables/reconstructed"
        ),
    )

    assert audit.ready is True
    assert audit.failed_gates == []
    reconstructed = audit.checks.loc[
        audit.checks["gate"].eq("reconstructed_paper_table_evidence")
    ].iloc[0]
    assert bool(reconstructed["passed"]) is True
    assert bool(reconstructed["required"]) is False
    assert reconstructed["detail"] == "4 checks passed"


def test_release_readiness_records_missing_original_paper_methods_as_limitation(
    tmp_path,
):
    _write_checks(
        tmp_path / "validation/outputs/latest/checks.csv",
        [{"check": "qualitative", "passed": True}],
    )
    _write_checks(
        tmp_path / "validation/outputs/inference/checks.csv",
        [{"check": "coverage", "passed": True}],
    )
    _write_checks(
        tmp_path / "validation/outputs/paper_tables/full_combined_reconstructed_dgp3/checks.csv",
        [
            {"check": "scaled_mad_within_relative_tolerance", "passed": True},
            {"check": "scaled_mse_within_relative_tolerance", "passed": True},
            {"check": "paper_targets_matched", "passed": True},
            {"check": "config_coverage", "passed": True},
        ],
    )
    _write_matching_paper_targets(
        tmp_path
        / "validation/outputs/paper_tables/full_combined_reconstructed_dgp3/paper_targets.csv"
    )
    _write_dist_files(tmp_path)

    audit = assess_release_readiness(project_root=tmp_path)

    assert audit.ready is True
    assert audit.failed_gates == []
    coverage = audit.checks.loc[audit.checks["gate"].eq("paper_method_coverage")].iloc[0]
    assert bool(coverage["required"]) is False
    assert bool(coverage["passed"]) is False
    assert "6/9 paper methods implemented" in coverage["detail"]
    assert "6/9 have transcribed table targets" in coverage["detail"]
    assert "missing targets=LIML-INT" not in coverage["detail"]
    assert "2SLS-SSL" in coverage["detail"]
    assert "UJIVE" in coverage["detail"]
    assert "IJIVE" in coverage["detail"]


def test_release_readiness_records_stale_paper_targets_without_blocking_release(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "simulations.audit_release_readiness.paper_method_coverage",
        _all_paper_methods_covered,
    )
    _write_checks(
        tmp_path / "validation/outputs/latest/checks.csv",
        [{"check": "qualitative", "passed": True, "detail": "ok"}],
    )
    _write_checks(
        tmp_path / "validation/outputs/inference/checks.csv",
        [{"check": "coverage", "passed": True, "detail": "ok"}],
    )
    _write_checks(
        tmp_path / "validation/outputs/paper_tables/combined/checks.csv",
        [
            {"check": "scaled_mad_within_relative_tolerance", "passed": True},
            {"check": "scaled_mse_within_relative_tolerance", "passed": True},
            {"check": "paper_targets_matched", "passed": True},
            {"check": "config_coverage", "passed": True},
        ],
    )
    pd.DataFrame({"method": ["pooled"]}).to_csv(
        tmp_path / "validation/outputs/paper_tables/combined/paper_targets.csv",
        index=False,
    )
    _write_dist_files(tmp_path)

    audit = assess_release_readiness(
        project_root=tmp_path,
        paper_tables_dir=tmp_path / "validation/outputs/paper_tables/combined",
    )

    assert audit.ready is True
    assert audit.failed_gates == []
    gate = audit.checks.loc[
        audit.checks["gate"].eq("paper_table_artifact_freshness")
    ].iloc[0]
    assert bool(gate["required"]) is False
    assert bool(gate["passed"]) is False
    assert "artifact rows=1" in gate["detail"]
    assert "current target rows=330" in gate["detail"]


def test_release_readiness_records_missing_paper_table_artifact_as_limitation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "simulations.audit_release_readiness.paper_method_coverage",
        _all_paper_methods_covered,
    )
    _write_checks(
        tmp_path / "validation/outputs/latest/checks.csv",
        [{"check": "qualitative", "passed": True}],
    )
    _write_checks(
        tmp_path / "validation/outputs/inference/checks.csv",
        [{"check": "coverage", "passed": True}],
    )
    _write_dist_files(tmp_path)

    audit = assess_release_readiness(project_root=tmp_path)

    assert audit.ready is True
    assert audit.failed_gates == []
    paper_gate = audit.checks.loc[
        audit.checks["gate"].eq("paper_table_validation")
    ].iloc[0]
    assert bool(paper_gate["passed"]) is False
    assert bool(paper_gate["required"]) is False
    assert "missing optional evidence file" in paper_gate["detail"]
