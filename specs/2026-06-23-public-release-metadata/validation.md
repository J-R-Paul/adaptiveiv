# Validation

## Required Evidence

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_packaging.py -q`
- Built sdist and wheel metadata inspection.
- Full test, lint, typing, docs, build, and clean no-editable smoke gates.

## Current Result

Done.

Red checks before implementation:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_packaging.py -q`: 2 failed, 1 passed. Failures showed missing `project.license` metadata and missing `LICENSE` in the built sdist.

Green evidence after implementation:

- `UV_CACHE_DIR=/tmp/uv-cache uv build`: built `dist/adaptiveiv-0.1.0.tar.gz` and `dist/adaptiveiv-0.1.0-py3-none-any.whl`.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src .venv/bin/python -m pytest tests/test_packaging.py -q`: 3 passed.
- Built `PKG-INFO` reports `License-Expression: MIT`, `License-File: LICENSE`, public classifiers, and `docs`/`examples` extras.
- Built wheel contains `adaptiveiv-0.1.0.dist-info/licenses/LICENSE` and matching metadata.
