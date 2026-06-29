import os
import subprocess

import pandas as pd


def _uv_env():
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/uv-cache")
    return env


def test_simulation_example_runs_through_uv_without_pythonpath_hack():
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-editable",
            "--group",
            "dev",
            "python",
            "examples/simulation_example.py",
        ],
        check=True,
        capture_output=True,
        env=_uv_env(),
        text=True,
    )

    assert "Adaptive estimate" in completed.stdout
    assert "Selected groups" in completed.stdout


def test_replication_validation_script_writes_outputs(tmp_path):
    output_dir = tmp_path / "validation"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-editable",
            "--group",
            "dev",
            "python",
            "simulations/validate_replication.py",
            "--repetitions",
            "2",
            "--n-groups",
            "8",
            "--n-per-group",
            "30",
            "--seed",
            "2026",
            "--n-splits",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        env=_uv_env(),
        text=True,
    )

    assert "Validation report" in completed.stdout
    results = pd.read_csv(output_dir / "simulation_results.csv")
    summary = pd.read_csv(output_dir / "summary.csv")
    report = (output_dir / "report.md").read_text()

    assert {"scenario", "replication", "method", "beta_hat"}.issubset(results.columns)
    assert {"scenario", "method", "scaled_mse", "mean_selected_total"}.issubset(
        summary.columns
    )
    assert results.loc[results["method"] == "adaptive", "n_splits"].eq(2).all()
    assert "# adaptiveiv Replication Validation Report" in report
    assert "Scope limits" in report


def test_inference_validation_script_writes_outputs(tmp_path):
    output_dir = tmp_path / "inference"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-editable",
            "--group",
            "dev",
            "python",
            "simulations/validate_inference.py",
            "--preset",
            "smoke",
            "--repetitions",
            "3",
            "--n-groups",
            "8",
            "--n-per-group",
            "40",
            "--seed",
            "3030",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        env=_uv_env(),
        text=True,
    )

    assert "Inference validation report" in completed.stdout
    results = pd.read_csv(output_dir / "inference_results.csv")
    summary = pd.read_csv(output_dir / "summary.csv")
    checks = pd.read_csv(output_dir / "checks.csv")
    report = (output_dir / "report.md").read_text()

    assert {"beta_hat", "bse", "covered_95", "rejected_5pct"}.issubset(
        results.columns
    )
    assert {
        "variance_component_a",
        "variance_component_b",
        "variance_average",
        "df_resid",
    }.issubset(results.columns)
    assert {"coverage_95", "rejection_5pct", "finite_share"}.issubset(
        summary.columns
    )
    assert {"check", "passed", "detail"}.issubset(checks.columns)
    assert "# adaptiveiv Inference Validation Report" in report
    assert "paper_homoskedastic" in report


def test_paper_table_validation_script_writes_outputs(tmp_path):
    output_dir = tmp_path / "paper_tables"
    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-editable",
            "--group",
            "dev",
            "python",
            "simulations/validate_paper_tables.py",
            "--preset",
            "smoke",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        env=_uv_env(),
        text=True,
    )

    assert "Paper-table validation report" in completed.stdout
    targets = pd.read_csv(output_dir / "paper_targets.csv")
    comparison = pd.read_csv(output_dir / "paper_comparison.csv")
    checks = pd.read_csv(output_dir / "checks.csv")
    report = (output_dir / "report.md").read_text()

    assert {"source_table", "method", "metric", "paper_value"}.issubset(
        targets.columns
    )
    assert {"observed_value", "paper_value", "relative_error"}.issubset(
        comparison.columns
    )
    assert checks["passed"].all()
    assert "# adaptiveiv Paper Table Validation Report" in report


def test_paper_table_chunk_aggregation_script_writes_outputs(tmp_path):
    chunk_a = tmp_path / "chunk_a"
    chunk_b = tmp_path / "chunk_b"
    output_dir = tmp_path / "combined"
    for start, stop, path in [(0, 1, chunk_a), (1, 2, chunk_b)]:
        subprocess.run(
            [
                "uv",
                "run",
                "--no-editable",
                "--group",
                "dev",
                "python",
                "simulations/validate_paper_tables.py",
                "--preset",
                "smoke",
                "--config-start",
                str(start),
                "--config-stop",
                str(stop),
                "--max-configs",
                "1",
                "--output-dir",
                str(path),
            ],
            check=True,
            capture_output=True,
            env=_uv_env(),
            text=True,
        )

    completed = subprocess.run(
        [
            "uv",
            "run",
            "--no-editable",
            "--group",
            "dev",
            "python",
            "simulations/aggregate_paper_table_chunks.py",
            str(chunk_a),
            str(chunk_b),
            "--expected-config-count",
            "2",
            "--relative-tolerance",
            "10",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        env=_uv_env(),
        text=True,
    )

    assert "Combined paper-table report" in completed.stdout
    manifest = pd.read_csv(output_dir / "config_manifest.csv")
    comparison = pd.read_csv(output_dir / "paper_comparison.csv")
    checks = pd.read_csv(output_dir / "checks.csv")
    report = (output_dir / "report.md").read_text()

    assert sorted(manifest["config_index"].tolist()) == [0, 1]
    assert sorted(comparison["config_index"].unique().tolist()) == [0, 1]
    assert checks.loc[checks["check"] == "config_coverage", "passed"].iloc[0]
    assert "# adaptiveiv Combined Paper Table Validation Report" in report
