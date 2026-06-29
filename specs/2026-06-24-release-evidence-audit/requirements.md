# Requirements

Spec: 2026-06-24-release-evidence-audit
Created: 2026-06-24
Status: done
Type: release validation
Summary: Make release readiness evidence distinguish canonical paper confirmation from reconstructed seed evidence.

## Goal

Improve the public-release audit so it records all current validation evidence
without weakening the release gate. The canonical paper-table numerical
confirmation must remain release-blocking, while reconstructed observation-seed
evidence should be visible as supporting evidence only.

## Scope

- Keep `paper_table_validation` tied to the unreplaced full-combined artifact.
- Add a separate non-blocking audit row for reconstructed observation-seed
  paper-table evidence when that artifact exists.
- Ensure `ready` and `failed_gates` are computed from required release gates
  only, not from supporting evidence rows.
- Strengthen packaging tests so generated validation outputs and internal specs
  are not shipped in the source distribution.
- Document the distinction in validation docs.

## Out Of Scope

- Treating reconstructed seed evidence as original paper confirmation.
- Relaxing the 25 percent paper-table tolerance.
- Marking the package release-ready while the canonical paper-table gate fails.
- Adding network-dependent release gates.

## Success Looks Like

- Release audit reports the reconstructed paper-table artifact separately.
- The audit still reports `Ready: False` when canonical paper-table validation
  fails.
- Tests cover non-blocking evidence rows and sdist exclusions.
- Verification commands pass after the change.

## Outcome

Completed on 2026-06-24. The release audit now distinguishes required release
gates from optional supporting evidence. The unreplaced full-combined
paper-table artifact remains the required numerical-confirmation gate, while
the reconstructed observation-seed artifact is reported as non-required
supporting evidence. The current refreshed audit reports `Ready: False` with
failed gate `paper_table_validation`, and separately reports
`reconstructed_paper_table_evidence=True`.
