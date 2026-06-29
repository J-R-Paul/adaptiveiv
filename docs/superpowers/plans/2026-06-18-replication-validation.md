# Replication Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a rerunnable Monte Carlo validation layer for the Abadie-Gu-Shen adaptive IV package.

**Architecture:** Keep estimator code separate from validation orchestration. DGP construction lives in `src/adaptiveiv/simulation.py`; aggregation helpers live in `src/adaptiveiv/validation.py`; runnable release validation lives under `simulations/`; committed documentation lives under `validation/` and `docs/`.

**Tech Stack:** Python 3.10+, `numpy`, `pandas`, existing `AdaptiveIV` estimators, `pytest`, `uv`.

---

### Task 1: Paper DGP Constructors

**Files:**
- Modify: `src/adaptiveiv/simulation.py`
- Modify: `src/adaptiveiv/__init__.py`
- Test: `tests/test_replication_validation.py`

- [ ] **Step 1: Write failing DGP tests**

Create tests that import `paper_group_strengths` and `simulate_paper_section4_dgp`, then check DGP1/DGP2/DGP3 group-strength distributions, reproducibility, and centered chi-squared errors.

- [ ] **Step 2: Verify red**

Run: `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_replication_validation.py -q`

Expected: import failure for the new functions.

- [ ] **Step 3: Implement constructors**

Add `paper_group_strengths(...)` and `simulate_paper_section4_dgp(...)` with parameters for DGP name, groups, group size, strong/weak shares, `rho_uv`, `beta`, error distribution, and seed. The returned DataFrame must include `Y`, `W`, `Z`, `X`, `u`, `v`, `rho_g`, `group`, and `dgp`.

- [ ] **Step 4: Verify green**

Run the same targeted pytest command. Expected: all new DGP tests pass.

### Task 2: Validation Metrics And Estimator Runner

**Files:**
- Create: `src/adaptiveiv/validation.py`
- Modify: `src/adaptiveiv/__init__.py`
- Test: `tests/test_replication_validation.py`

- [ ] **Step 1: Write failing metric tests**

Add tests for `estimate_methods_once`, `summarize_simulation_results`, and `validate_simulation_summary`. Tests should require pooled, fully interacted, split interacted, adaptive, and oracle estimates; metrics should include bias, MSE, scaled MSE, MAD, finite share, mean selected groups, and validation flags.

- [ ] **Step 2: Verify red**

Run: `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_replication_validation.py -q`

Expected: import failure for validation helpers.

- [ ] **Step 3: Implement helpers**

Implement a small validation module that fits implemented estimators, computes oracle estimates by passing known nonzero or strong groups through the split-sample machinery, aggregates results, and returns transparent validation flags.

- [ ] **Step 4: Verify green**

Run the targeted pytest command. Expected: all replication validation tests pass.

### Task 3: Rerunnable Script And Report

**Files:**
- Create: `simulations/validate_replication.py`
- Create: `validation/README.md`
- Test: `tests/test_docs_examples.py`

- [ ] **Step 1: Write failing script smoke test**

Add a test that runs the validation script with a tiny grid and temporary output directory, then checks that `simulation_results.csv`, `summary.csv`, and `report.md` exist and contain the expected columns/headings.

- [ ] **Step 2: Verify red**

Run: `PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_docs_examples.py -q`

Expected: script missing.

- [ ] **Step 3: Implement script and validation README**

The script should accept `--repetitions`, `--n-groups`, `--n-per-group`, `--seed`, and `--output-dir`. It should write CSV outputs and a Markdown report with settings, qualitative checks, and scope limits.

- [ ] **Step 4: Verify green**

Run the targeted docs/examples tests. Expected: script smoke passes.

### Task 4: Documentation And Package Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/index.md`
- Modify: `docs/limitations.md`
- Modify: `pyproject.toml` if artifact inclusion needs adjustment.
- Modify: `specs/2026-06-18-replication-validation/validation.md`

- [ ] **Step 1: Document rerun command**

Add `uv run --group dev python simulations/validate_replication.py ...` to README/docs and state the validation scope honestly.

- [ ] **Step 2: Run full verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest -q
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev ruff check .
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev mypy src/adaptiveiv
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group docs mkdocs build --strict --site-dir /tmp/adaptiveiv-validation-docs
```

Expected: all pass, aside from the known upstream Material for MkDocs warning.

- [ ] **Step 3: Run default validation smoke**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --group dev python simulations/validate_replication.py --repetitions 3 --n-groups 12 --n-per-group 40 --output-dir /tmp/adaptiveiv-validation-smoke
```

Expected: CSV and report artifacts are produced.

- [ ] **Step 4: Update spec status**

Record validation evidence in `specs/2026-06-18-replication-validation/validation.md`. Keep the spec non-terminal unless the package-level publication bar is fully met.
