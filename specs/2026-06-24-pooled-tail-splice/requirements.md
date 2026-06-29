# Requirements

Spec: 2026-06-24-pooled-tail-splice
Created: 2026-06-24
Status: done
Type: numerical validation
Summary: Test whether real high-tail candidate seeds can reconcile failed pooled chi-square MSE rows by one-tail replacement.

## Goal

Quantify how much of the remaining pooled chi-square paper-table MSE mismatch
can be explained by rare tail draws already found by the seed diagnostics. The
diagnostic should compute counterfactual paper-table MSE values after replacing
the current largest squared-error contribution with a candidate seed's pooled
estimate.

## Scope

- Read full paper-comparison output and candidate pooled-tail seed CSVs.
- Identify failed pooled `scaled_mse` rows.
- For each candidate seed, compute the spliced scaled MSE obtained by replacing
  the current maximum scaled squared error with the candidate seed's scaled
  squared error.
- Rank candidate seeds by absolute relative error to the paper target.
- Write CSV and Markdown artifacts.

## Out Of Scope

- Mutating full validation outputs.
- Claiming candidate seeds are the paper's original simulation seeds.
- Relaxing release readiness.
- Replacing a full Monte Carlo run.

## Success Looks Like

- Config 18 can be checked against the close target seed found by the 1,000-seed
  diagnostic.
- Config 19 can use the existing 10,000-seed tail scan to show whether real
  large-tail seeds make the paper MSE plausible under a one-tail replacement.
- The report clearly labels the output as counterfactual reconstruction
  evidence, not release confirmation.

## Outcome

Completed on 2026-06-24. The diagnostic writes ranked CSV and Markdown
artifacts under
`validation/outputs/paper_tables/pooled_tail_splice_reconstruction`. It found
32 candidate counterfactual rows, with two candidates inside the 25 percent
paper-table tolerance: config 18 with seed `3585863902` and config 19 with seed
`2612616469`. This supports the rare-tail explanation for those failed pooled
MSE rows, but it does not identify the paper's original seeds and does not
change release readiness.
