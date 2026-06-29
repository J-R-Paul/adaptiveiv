# Plan

Status: done

1. Add a regression test that monkeypatches the analytic inference routine and
   verifies `cov_type="none"` does not call it.
2. Thread the resolved inference request into split-repetition fitting.
3. Preserve the existing unavailable-inference result state for point-estimate
   fits.
4. Verify focused inference API behavior, full tests, lint, typing, docs, and
   the inference validation smoke runner.
