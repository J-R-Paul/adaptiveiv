# adaptiveiv Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `adaptiveiv` into a uv-managed, statsmodels/linearmodels-friendly research package that satisfies `plan.md`.

**Architecture:** Keep a small public surface around `AdaptiveIV`, estimator helpers, and `AdaptiveIVResults`. Move statistical primitives into focused modules so the estimator, formula parsing, diagnostics, benchmark estimators, tests, examples, and packaging can be validated independently.

**Tech Stack:** Python, uv, NumPy, pandas, statsmodels, patsy, pytest, hatchling.

---

## File Structure

- `src/adaptiveiv/model.py`: public `AdaptiveIV` model class, input validation, `fit`, and `from_formula`.
- `src/adaptiveiv/estimators.py`: group statistics, adaptive threshold, split-sample estimator, and benchmark estimators.
- `src/adaptiveiv/results.py`: statsmodels-like result object and summaries.
- `src/adaptiveiv/formula.py`: formula parser for `Y ~ 1 + X + [W ~ Z]`.
- `src/adaptiveiv/diagnostics.py`: diagnostics dataclasses and DataFrame builders.
- `src/adaptiveiv/simulation.py`: paper-style data generating processes for examples/tests.
- `src/adaptiveiv/__init__.py`: public exports.
- `tests/`: pytest tests outside the runtime package.
- `examples/simulation_example.py`: valid-instrument worked example.
- `README.md`: installation, quick start, assumptions, diagnostics, and citation.
- `pyproject.toml`: uv/hatch metadata, runtime dependencies, dependency groups, pytest/ruff config, and package include/exclude rules.

## Task 1: Packaging and Test Harness

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/test_packaging.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write failing packaging tests**

```python
def test_package_imports_from_installed_environment():
    import adaptiveiv

    assert adaptiveiv.AdaptiveIV.__name__ == "AdaptiveIV"
```

- [ ] **Step 2: Run packaging test and verify red**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_packaging.py -q`

Expected: FAIL because pytest may be missing and/or the package is not importable through uv.

- [ ] **Step 3: Fix package metadata**

Use `pyproject.toml` as the source of truth. Specify packages under the hatch wheel target, move tests out of `src/adaptiveiv`, add dependency groups for `dev`, `docs`, and `examples`, add pytest config, and lower the Python floor to a realistic supported range if dependencies permit.

- [ ] **Step 4: Verify green**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_packaging.py -q`

Expected: PASS.

## Task 2: Formula API

**Files:**
- Modify: `src/adaptiveiv/formula.py`
- Modify: `src/adaptiveiv/model.py`
- Modify: `src/adaptiveiv/__init__.py`
- Create: `tests/test_formula_api.py`

- [ ] **Step 1: Write failing formula tests**

Tests should cover `AdaptiveIV.from_formula("Y ~ 1 + X1 + X2 + [W ~ Z]", data=df, groups="group")`, agreement with the direct dataframe API, quoted variable names supported by patsy, and clear errors for missing or ambiguous `[W ~ Z]` blocks.

- [ ] **Step 2: Verify red**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_formula_api.py -q`

Expected: FAIL because `from_formula` and parser behavior do not exist.

- [ ] **Step 3: Implement minimal formula parser**

Parse exactly one bracketed IV block, return dependent variable, exogenous controls, endogenous variable, excluded instrument, and intercept flag. Reject multiple endogenous variables or instruments for the first release.

- [ ] **Step 4: Verify green**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_formula_api.py -q`

Expected: PASS.

## Task 3: Statistical Primitives and Diagnostics

**Files:**
- Create: `src/adaptiveiv/estimators.py`
- Create: `src/adaptiveiv/diagnostics.py`
- Modify: `src/adaptiveiv/model.py`
- Create: `tests/test_estimators.py`
- Create: `tests/test_diagnostics.py`

- [ ] **Step 1: Write failing tests for group stats and threshold selection**

Tests should check residualized instruments are orthogonal to controls within group, `rho_hat = (Z'W)/(Z'Z)`, `mu_hat = (Z'W)/sqrt(Z'Z)`, adaptive risk uses `mu_check = mu_hat / sqrt(kappa)`, and the selected count implied by `delta_hat` matches the risk-minimizing `K_hat`.

- [ ] **Step 2: Write failing diagnostics tests**

Tests should require `group_diagnostics`, `selection_summary`, split estimates, thresholds, numerator/denominator components, and structured warnings on the results object.

- [ ] **Step 3: Verify red**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_estimators.py tests/test_diagnostics.py -q`

Expected: FAIL because modules and diagnostics are missing.

- [ ] **Step 4: Implement primitives and diagnostics**

Implement focused pure functions for residualization, group statistics, variance estimates, adaptive thresholding, split estimates, and diagnostics table construction. Use `warnings.warn` instead of `print`.

- [ ] **Step 5: Verify green**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_estimators.py tests/test_diagnostics.py -q`

Expected: PASS.

## Task 4: Public Model and Results API

**Files:**
- Modify: `src/adaptiveiv/model.py`
- Modify: `src/adaptiveiv/results.py`
- Modify: `src/adaptiveiv/__init__.py`
- Create: `tests/test_model_api.py`
- Create: `tests/test_results.py`

- [ ] **Step 1: Write failing public API tests**

Tests should require the direct constructor names from `plan.md`: `dependent`, `endogenous`, `instruments`, `exog`, and `groups`. They should also require backward-compatible aliases where cheap: `exog_endog`, `exog_exog`, and `instrument`.

- [ ] **Step 2: Write failing results tests**

Tests should require `params`, `bse`, `cov`, `tvalues`, `pvalues`, `conf_int()`, `summary()`, `first_stage`, `group_diagnostics`, `selected_groups`, `selection_summary`, `method`, `nobs`, `ngroups`, and `cov_type`. Unsupported inferential quantities must not appear as fake finite values.

- [ ] **Step 3: Verify red**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_model_api.py tests/test_results.py -q`

Expected: FAIL because the current API and result object are incomplete.

- [ ] **Step 4: Implement public API**

Route `AdaptiveIV.fit()` through the estimator primitives. Support `method="adaptive"`, `"select"`, `"split-interacted"`, `"interacted"`, and `"pooled"` where feasible for validation and comparison. Support `cov_type="homoskedastic"` initially and explicit not-available behavior for unsupported covariance modes.

- [ ] **Step 5: Verify green**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_model_api.py tests/test_results.py -q`

Expected: PASS.

## Task 5: Simulation and Benchmark Validation

**Files:**
- Create: `src/adaptiveiv/simulation.py`
- Create: `tests/test_simulation_validation.py`
- Modify: `examples/simulation_example.py`

- [ ] **Step 1: Write failing paper-style simulation tests**

Tests should generate valid-instrument DGPs with known beta, strong/weak/zero first-stage groups, and no direct correlation between the excluded instrument and the structural error. The adaptive estimator should recover beta within a broad finite-sample tolerance on a fixed seed and select mostly strong groups in a well-separated DGP.

- [ ] **Step 2: Write failing benchmark tests**

Tests should compare pooled and fully interacted helper estimators with statsmodels matrix calculations in simple cases.

- [ ] **Step 3: Verify red**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_simulation_validation.py -q`

Expected: FAIL because simulation helpers and benchmark estimators are missing or incomplete.

- [ ] **Step 4: Implement simulation and examples**

Add DGP helpers and update the example so it runs through `uv`, uses a valid instrument, prints adaptive and comparison estimates, and shows selected groups.

- [ ] **Step 5: Verify green**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_simulation_validation.py -q`

Expected: PASS.

## Task 6: Documentation and Release Checks

**Files:**
- Modify: `README.md`
- Modify: `.gitignore` if a Git repository is initialized later
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing docs smoke test**

Add a test that executes the README quick-start code path or the example script through `subprocess`.

- [ ] **Step 2: Verify red**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_docs_examples.py -q`

Expected: FAIL until documentation/example code is runnable.

- [ ] **Step 3: Write docs**

README must cover installation with uv, quick start, formula and direct APIs, estimator overview, diagnostics, inference caveats, limitations, and citation.

- [ ] **Step 4: Verify full package**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv build
```

Expected: both commands exit 0.

## Task 7: Independent Review

**Files:**
- No predetermined write set.

- [ ] **Step 1: Dispatch independent verification subagent**

Ask the subagent to read `plan.md`, inspect the implementation, run or review the verification commands, and report gaps against each major spec section.

- [ ] **Step 2: Address findings**

Fix Critical and Important findings before claiming completion.

- [ ] **Step 3: Final verification**

Re-run the full package verification commands after addressing findings.

