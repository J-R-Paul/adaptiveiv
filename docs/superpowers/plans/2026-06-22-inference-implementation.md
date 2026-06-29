# Inference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the first public inference slice from `specs/2026-06-22-inference-design/design.md`.

**Architecture:** Add a small inference primitive in `src/adaptiveiv/estimators.py` that computes paper-baseline homoskedastic component variances from select-and-interact moments. Thread the resulting scalar standard error and metadata through `AdaptiveIV.fit()` into `AdaptiveIVResults`, while keeping repeated splits and absolute-sign selection point-estimate-only via `cov_type="none"`.

**Tech Stack:** Python 3.10+, NumPy, pandas, pytest, uv.

---

### Task 1: Result Object Inference Metadata

**Files:**
- Modify: `tests/test_model_results_api.py`
- Modify: `src/adaptiveiv/results.py`

- [ ] **Step 1: Write failing tests**

Add tests that require `fit(cov_type="homoskedastic")` to expose finite `bse`,
`cov`, `tvalues`, `pvalues`, `conf_int()`, `cov_estimator`, and
`inference_notes`, and require `fit(cov_type="none")` to keep inferential
accessors unavailable.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_model_results_api.py -q
```

Expected: the new inference tests fail because the result object has no
inference metadata and still returns no standard errors.

- [ ] **Step 3: Implement result metadata**

Extend `AdaptiveIVResults.__init__` with `cov_estimator`, `inference_notes`,
and optional `df_resid`. Make the summary print inference columns only when
`inference_available` is true and print a no-inference note otherwise.

- [ ] **Step 4: Verify green for result metadata**

Run the same targeted pytest command. Expected: result-object tests that do not
need covariance math pass after covariance is threaded in Task 2.

### Task 2: Homoskedastic Select-and-Interact Covariance

**Files:**
- Modify: `tests/test_estimators.py`
- Modify: `tests/test_model_results_api.py`
- Modify: `src/adaptiveiv/estimators.py`
- Modify: `src/adaptiveiv/model.py`

- [ ] **Step 1: Write failing tests**

Add tests for the component variance formula:

```text
sigma_u^2 * sum(q_i^2) / (sum(q_i W_i)^2)
```

where `q_i = rho_hat_from_selection_split[group] * z_resid_i`, and for the
single-repetition average variance:

```text
(var_a + var_b) / 4
```

with an inference note that the cross-component covariance is treated as
first-order negligible for this first slice.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_estimators.py tests/test_model_results_api.py -q
```

Expected: tests fail because no covariance primitive exists.

- [ ] **Step 3: Implement covariance primitive**

Create a dataclass such as `InferenceEstimate` and a function such as
`homoskedastic_split_variance()` in `estimators.py`. It should return no
inference if a component has no selected groups, zero denominator, non-finite
variance, or nonpositive residual degrees of freedom.

- [ ] **Step 4: Thread covariance through `AdaptiveIV.fit()`**

Accept `cov_type=None`, `"none"`, `"homoskedastic"`, and `"unadjusted"`.
For `n_splits=1`, `selection_rule="positive"`, and select-and-interact methods,
compute `bse`. For `n_splits>1` or `selection_rule="absolute"` with a non-none
covariance request, raise a clear `ValueError`.

- [ ] **Step 5: Verify green**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_estimators.py tests/test_model_results_api.py -q
```

Expected: targeted tests pass.

### Task 3: Validation Helpers And Docs

**Files:**
- Modify: `src/adaptiveiv/validation.py`
- Modify: `README.md`
- Modify: `docs/api.md`
- Modify: `docs/limitations.md`
- Create: `docs/inference.md`
- Modify: `mkdocs.yml`
- Modify: `tests/test_docs_examples.py`

- [ ] **Step 1: Write failing tests**

Add doc/API tests requiring repeated-split validation helpers to pass
`cov_type="none"` and requiring the docs to mention that repeated split
stability is not a standard error.

- [ ] **Step 2: Verify red**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_docs_examples.py tests/test_replication_validation.py -q
```

Expected: tests fail until helper/docs are updated.

- [ ] **Step 3: Update helper and docs**

Use `cov_type="none"` for repeated-split validation helper calls. Add a
dedicated inference docs page and short README/API/limitations updates.

- [ ] **Step 4: Verify green**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_docs_examples.py tests/test_replication_validation.py -q
```

Expected: tests pass.

### Task 4: Full Verification

**Files:**
- No direct file edits expected.

- [ ] **Step 1: Run package test suite**

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest -q
```

- [ ] **Step 2: Run lint and typing**

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev ruff check .
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group dev mypy src/adaptiveiv
```

- [ ] **Step 3: Build docs**

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --group docs mkdocs build --strict --site-dir /tmp/adaptiveiv-inference-docs
```

- [ ] **Step 4: Run validation smoke**

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --group dev python simulations/validate_replication.py --repetitions 2 --n-groups 12 --n-per-group 40 --n-splits 3 --output-dir /tmp/adaptiveiv-inference-validation-smoke
```

- [ ] **Step 5: Build artifacts**

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv build
```

Expected: all commands complete successfully.
