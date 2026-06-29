# Plan

## Approach

Thread the existing internal `InferenceEstimate` component variance information
through the public result object and validation script without widening the
eligible inference configurations.

## Work Locations

- `src/adaptiveiv/estimators.py`: carry component degrees of freedom alongside
  component variances.
- `src/adaptiveiv/model.py`: construct a diagnostics table from the inference
  object.
- `src/adaptiveiv/results.py`: expose the diagnostics table through a guarded
  property.
- `simulations/validate_inference.py`: write flat variance and degrees-of-
  freedom columns.
- `tests/test_model_results_api.py` and `tests/test_docs_examples.py`: public
  API and artifact checks.
- `docs/api.md` and `docs/inference.md`: user-facing contract.

## Next Concrete Step

Run the focused result/API and validation-artifact tests, then run the broader
package gates.
