# Plan

## Approach

Use tests to pin the public metadata contract, then update package metadata and
rebuild artifacts.

## Work Locations

- `pyproject.toml`
- `LICENSE`
- `tests/test_packaging.py`
- `dist/`
- `specs/log.md`

## Next Concrete Step

Run the full package gates and a clean no-editable smoke after rebuilding the
sdist and wheel.
