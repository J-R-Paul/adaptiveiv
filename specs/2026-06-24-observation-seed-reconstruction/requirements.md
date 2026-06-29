# Requirements

Spec: 2026-06-24-observation-seed-reconstruction
Created: 2026-06-24
Status: done
Type: numerical validation
Summary: Test whether real observation-seed replacements can bring the full paper-table validation inside release tolerance.

## Goal

Move beyond pooled-only splice arithmetic by testing reconstructed
observation-level seed replacements at the full validation-row level. For each
problem pooled chi-square row, replace one existing Monte Carlo replication with
a real candidate seed, recompute all implemented estimators for that
replication, and reaggregate the paper-table comparison.

## Scope

- Read an existing combined paper-table validation artifact.
- Accept a replacement map of `config_index:replication:seed` entries.
- Recompute all implemented estimators for each replacement using the same
  paper configuration and DGP3 fixed-strength provenance when applicable.
- Write replaced simulation results, summaries, comparison rows, checks, and a
  Markdown report.
- Clearly label the output as reconstructed validation evidence, not recovered
  original paper seeds.

## Out Of Scope

- Claiming the replacement seeds are the paper's original seeds.
- Mutating the strict full-combined validation artifact in place.
- Relaxing release readiness tolerances.
- Replacing DGP3 strength-vector reconstruction.

## Success Looks Like

- The diagnostic can recompute replacements for configs 7, 18, and 19.
- The report says whether the resulting full comparison passes the existing
  25 percent paper-table tolerance.
- Tests cover replacement-map parsing and row replacement behavior.

## Outcome

Completed on 2026-06-24. The diagnostic recomputed full estimator rows for
configs 7, 18, and 19 using real observation-level replacement seeds and wrote a
separate reconstructed artifact at
`validation/outputs/paper_tables/full_combined_reconstructed_observation_seeds`.
That artifact passes all four paper-table checks under the existing 25 percent
relative tolerance. It remains reconstruction evidence only; the canonical
release audit should stay tied to the unreplaced full-combined paper-table
artifact until original simulation state is recovered or a release policy
explicitly accepts reconstructed seed evidence.
