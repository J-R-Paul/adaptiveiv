# Plan

## Approach

Add a paper-target layer that is distinct from the existing qualitative
validation runner. The package should expose transcribed targets and comparison
helpers from `adaptiveiv.paper_benchmarks`, while the runnable validation script
lives under `simulations/`.

## Work Location

- `src/adaptiveiv/paper_benchmarks.py`: Section 4 target table and comparison
  helpers.
- `simulations/validate_paper_tables.py`: CLI runner for smoke, release, and
  full presets.
- `src/adaptiveiv/validation.py`: preserve paper configuration columns during
  summary aggregation.
- `simulations/validate_replication.py`: write paper configuration columns into
  raw validation output.
- `tests/test_replication_validation.py`: target coverage and comparison tests.
- `README.md`, `docs/index.md`, `docs/limitations.md`, `validation/README.md`:
  validation documentation.

## Next Concrete Step

Run focused tests and smoke validation, then run full package quality gates. The
full 500-repetition paper preset remains a separate long-running release check.
