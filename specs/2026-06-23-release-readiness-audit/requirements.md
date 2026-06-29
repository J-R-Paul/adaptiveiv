# Requirements

Spec: 2026-06-23-release-readiness-audit
Created: 2026-06-23
Status: done
Type: release validation
Summary: Add a machine-readable release-readiness audit that makes validation blockers explicit before public release.

## Goal

Create a single release audit entry point that inspects generated validation and
packaging artifacts and reports whether `adaptiveiv` is ready for public
release. The audit should not rerun the expensive paper-table simulations; it
should read their current outputs and make the remaining release blockers
explicit.

## Scope

- Check qualitative replication validation artifacts.
- Check inference validation artifacts.
- Check full paper-table numerical confirmation artifacts.
- Check that wheel and sdist artifacts exist.
- Write machine-readable and Markdown audit outputs.
- Exit nonzero when required release gates fail, with an opt-out flag for
  producing reports during an in-progress release.

## Out Of Scope

- Replacing the expensive paper-table validation runner.
- Relaxing the paper-table MSE gate.
- Declaring public readiness while any required gate is red or missing.
- Uploading to PyPI.

## Success Looks Like

- A green artifact set reports `ready=true`.
- A paper-table `scaled_mse` failure reports `ready=false` and names the failed
  check.
- Missing required artifacts report `ready=false`.
- The current package state can generate an explicit release audit report that
  surfaces the known paper-table MSE blocker.

## Outcome

Completed on 2026-06-23. The package now has a release-readiness audit that
reads validation artifacts and distribution artifacts, writes CSV/JSON/Markdown
outputs, and exits nonzero unless all release gates pass. Current artifacts
correctly report `ready=false` because the full paper-table numerical
confirmation gate still fails on `scaled_mse`.
