# Spec: LIML-INT paper targets and artifact freshness

Status: done
Type: validation coverage

## Goal

Add the published LIML-INT `N x MAD` targets from Abadie, Gu, and Shen Tables
2-4 to the package's Section 4 paper-table target set, now that the package has
a point-estimate-only fully interacted LIML benchmark.

## Scope

- Transcribe only the paper-reported LIML-INT `N x MAD` values.
- Do not invent LIML-INT `N x MSE` targets, because the paper explicitly does
  not report LIML moments.
- Include LIML-INT in default paper-table validation runs so every transcribed
  LIML target can be matched by generated summary rows.
- Make the release audit fail when an existing paper-table artifact's
  `paper_targets.csv` is stale relative to the current target table.
- Update documentation and project backlog to reflect that LIML target
  transcription is no longer the remaining blocker.

## Success Statement

The current target table contains 330 rows: the existing 300 MSE/MAD rows for
five implemented 2SLS-style estimators plus 30 LIML-INT MAD rows. Paper-table
smoke validation matches the new target count for one configuration. The release
audit reports stale paper-table target artifacts as a required blocker until the
full paper-table artifact is regenerated.
