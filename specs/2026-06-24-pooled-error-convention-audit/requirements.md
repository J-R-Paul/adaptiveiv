# Requirements

Spec: 2026-06-24-pooled-error-convention-audit
Created: 2026-06-24
Status: done
Type: numerical validation
Summary: Audit whether chi-square error centering or scaling conventions explain failed pooled MSE paper-table rows.

## Goal

Test a concrete remaining root-cause hypothesis for the paper-table release
blocker: the failed pooled chi-square MSE rows might be caused by a mismatch in
how the paper generated chi-square errors.

## Scope

- Compare pooled 2SLS scaled MSE under multiple chi-square error conventions:
  centered, raw, standardized, and raw standardized.
- Cover DGP1 and DGP2 chi-square paper configurations, with optional filtering
  by `config_index`.
- Use the same fast pooled cross-product logic as the existing pooled tail and
  RNG diagnostics.
- Write a CSV and Markdown report that rank conventions by absolute relative
  error to the paper target.
- Document that this is a diagnostic and does not change the maintained DGP.

## Out Of Scope

- Changing `simulate_paper_section4_dgp`.
- Mutating full paper-table validation artifacts.
- Claiming release readiness.
- Auditing DGP3 fixed-strength vectors.

## Success Looks Like

- The diagnostic reproduces the current centered convention for relevant rows.
- It shows whether standardized or raw chi-square conventions improve the
  failed pooled rows without damaging the broader chi-square fit.
- Tests cover convention ranking and report rendering.

## Outcome

Completed on 2026-06-24. The diagnostic writes
`pooled_error_convention_audit.csv` and `report.md` artifacts. For the three
currently failed pooled MSE rows, raw and centered chi-square conventions are
numerically equivalent after pooled 2SLS intercept residualization, while
standardized chi-square conventions move the MSE much farther away from the
paper targets. The broader DGP1/DGP2 chi-square audit gives the same pattern, so
the remaining release blocker is not explained by chi-square centering or
scaling.
