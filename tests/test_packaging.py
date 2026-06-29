import tarfile
import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_from_uv_environment():
    import adaptiveiv
    from adaptiveiv.paper_benchmarks import paper_table_targets

    assert adaptiveiv.__version__ == "0.1.0"
    assert adaptiveiv.AdaptiveIV.__name__ == "AdaptiveIV"
    assert adaptiveiv.AdaptiveIVResults.__name__ == "AdaptiveIVResults"
    assert len(paper_table_targets()) == 330


def test_public_package_metadata_declares_license_classifiers_and_extras():
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]

    assert project["license"] == "MIT"
    assert "License :: OSI Approved :: MIT License" in project["classifiers"]
    assert "Topic :: Scientific/Engineering :: Information Analysis" in project[
        "classifiers"
    ]
    assert "docs" in project["optional-dependencies"]
    assert "examples" in project["optional-dependencies"]


def test_sdist_includes_license_and_public_validation_scripts():
    sdist_path = PROJECT_ROOT / "dist" / "adaptiveiv-0.1.0.tar.gz"

    with tarfile.open(sdist_path) as archive:
        names = set(archive.getnames())

    assert "adaptiveiv-0.1.0/LICENSE" in names
    assert "adaptiveiv-0.1.0/validation/README.md" in names
    assert "adaptiveiv-0.1.0/simulations/audit_release_readiness.py" in names
    assert "adaptiveiv-0.1.0/simulations/audit_pooled_rng_conventions.py" in names
    assert "adaptiveiv-0.1.0/simulations/audit_pooled_error_conventions.py" in names
    assert "adaptiveiv-0.1.0/simulations/diagnose_pooled_mse_targets.py" in names
    assert "adaptiveiv-0.1.0/simulations/diagnose_pooled_tail_splice.py" in names
    assert "adaptiveiv-0.1.0/simulations/diagnose_pooled_tail_seeds.py" in names
    assert "adaptiveiv-0.1.0/simulations/reconstruct_paper_table_seeds.py" in names
    assert not any(name.startswith("adaptiveiv-0.1.0/specs/") for name in names)
    assert not any(
        name.startswith("adaptiveiv-0.1.0/validation/outputs/") for name in names
    )
    assert not any(name.startswith("adaptiveiv-0.1.0/site/") for name in names)
    assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)
