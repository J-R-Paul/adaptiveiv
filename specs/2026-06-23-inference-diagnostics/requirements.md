# Requirements

Spec: 2026-06-23-inference-diagnostics
Created: 2026-06-23
Status: done
Type: inference implementation
Summary: Expose paper-homoskedastic inference diagnostics as a public, auditable result table and validation artifact.

## Goal

Make the first supported inference slice easier to inspect from normal Python
workflows. Users should be able to see the component-level variances used to
construct the averaged adaptive standard error, and validation outputs should
carry those quantities in flat CSV columns.

## Scope

- Add a public `AdaptiveIVResults.inference_diagnostics` table for supported
  `paper_homoskedastic` fits.
- Keep the table unavailable for point-estimate-only and unsupported inference
  fits.
- Include component and averaged variances in `validate_inference.py` outputs.
- Document the result-object contract in the API and inference docs.

## Out Of Scope

- Changing the homoskedastic covariance formula.
- Adding robust, clustered, bootstrap, repeated-split, absolute-selection, or
  benchmark-estimator inference.
- Treating split-estimate stability diagnostics as standard errors.

## Success Looks Like

- The diagnostics table has one row for split component `a`, one for split
  component `b`, and one for the final averaged estimator.
- The averaged row matches `results.cov` and `results.bse`.
- Unsupported inference results raise `NotImplementedError` for
  `inference_diagnostics`.
- Inference validation CSVs include component and averaged variance columns.

## Outcome

Completed on 2026-06-23. Supported `paper_homoskedastic` fits now expose
`results.inference_diagnostics`; inference validation artifacts include the
component and averaged variance columns used to audit `results.bse`.
