# Spec: Release scope for paper replication limits

Status: done
Type: release scope

## Goal

Update release readiness so exact original-paper Tables 2-4 replication is a
documented limitation rather than a required release gate.

## Scope

- Keep qualitative replication, inference validation, and distribution artifacts
  as required release gates.
- Continue to report exact paper-table validation, paper-target artifact
  freshness, and full paper-method coverage as non-blocking evidence.
- Add a clear limitation note: the current package does not exactly replicate
  all original paper tables, likely because original simulation seeds/state are
  unrecovered and some external comparators remain unimplemented.
- Update current docs and backlog so users do not mistake the package for full
  original-paper reproduction.

## Success Statement

The release audit can be ready when supported package validation and artifacts
pass, while still recording non-blocking paper-replication limitations. Public
docs state the limitation plainly.

## Result

Completed on 2026-06-24. Exact original-paper Tables 2-4 replication is now a
non-blocking limitation in the release audit and public docs.
