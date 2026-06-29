# Validation

## Required Evidence

- Failing tests are observed before implementation.
- Focused estimator and validation tests pass after implementation.
- Full tests, ruff, mypy, docs build, offline build, and no-editable release
  audit pass or report the expected remaining required blockers.

## Current Result

Passed for this implementation slice.

Red test evidence before implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_estimators.py::test_liml_interacted_matches_just_identified_iv_without_exog tests/test_replication_validation.py::test_estimate_methods_once_can_request_liml_interacted_benchmark -q
```

Result: import error because `fit_liml_interacted` did not exist.

Focused green checks after implementation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_estimators.py::test_liml_interacted_matches_just_identified_iv_without_exog tests/test_replication_validation.py::test_estimate_methods_once_can_request_liml_interacted_benchmark tests/test_replication_validation.py::test_paper_method_coverage_marks_unimplemented_original_paper_methods tests/test_release_readiness.py::test_release_readiness_flags_missing_original_paper_methods -q
```

Result: `4 passed`.

Broader checks:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_estimators.py tests/test_replication_validation.py tests/test_release_readiness.py -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/ruff check .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/mypy src/adaptiveiv
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --group docs mkdocs build --strict --site-dir /tmp/adaptiveiv-docs-check
UV_CACHE_DIR=/tmp/uv-cache uv build --offline
```

Results: `63 passed` for focused modules; `111 passed` for full tests; ruff
passed; mypy passed; docs build passed; sdist and wheel built.

Packaged/no-editable evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --reinstall-package adaptiveiv --offline --group dev python - <<'PY'
import numpy as np
import adaptiveiv.estimators as e
from adaptiveiv import simulate_paper_section4_dgp
from adaptiveiv.validation import estimate_methods_once
print(e.__file__, hasattr(e, 'fit_liml_interacted'))
data = simulate_paper_section4_dgp(dgp='dgp1', n_groups=8, n_per_group=80, strong_fraction=0.5, seed=1234)
row = estimate_methods_once(data, random_state=44, methods=['liml_interacted']).iloc[0]
print(row['method'], bool(row['finite']), np.isfinite(row['beta_hat']))
PY
```

Result: `fit_liml_interacted` available from `site-packages`; printed
`liml_interacted True True`.

Release audit:

```bash
PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=/tmp/uv-cache uv run --no-editable --reinstall-package adaptiveiv --offline --group dev python simulations/audit_release_readiness.py --no-fail --output-dir validation/outputs/release_audit
```

Result: `Ready: False`; failed gates: `paper_table_validation`,
`paper_method_coverage`. The method coverage detail is
`6/9 paper methods implemented; 5/9 have transcribed table targets; missing
implementations=2SLS-SSL, UJIVE, IJIVE; missing targets=LIML-INT`.
