# Backlog

## Complete Monte Carlo Table Replication

Why it matters: exact Tables 2-4 replication would strengthen academic
credibility beyond the qualitative package validation now in place.

Current state: implemented-method paper targets and observed-vs-paper comparison
are available through `simulations/validate_paper_tables.py`. A first full
500-repetition chunked pass covered all 30 implemented-method configurations and
matched the previous 300 paper-target rows, but failed the aggregate 25% release
tolerance because of Table 4 / DGP3 deviations and rare heavy-tail pooled MSE
rows. The in-code target table now has 330 rows after adding LIML-INT MAD
targets, so the full paper-table artifact must be regenerated before it can be
treated as current exact-replication evidence. This is now tracked as a known
paper-replication limitation rather than a public-release blocker.

Remaining dependency or trigger: identify the exact remaining DGP3 replication
convention or source code needed to close Table 4 after the untruncated-mixture
correction; decide how to handle volatile pooled chi-square MSE rows; regenerate
the full paper-table validation artifact against the expanded target table;
implement or depend on UJIVE, IJIVE, and 2SLS-SSL comparators to reproduce every
paper column rather than the implemented-method subset. The release audit now
reports this as non-blocking evidence with a limitation note: the package does
not exactly replicate all original paper tables, likely because original
simulation seeds/state are unrecovered and some external comparators remain
unimplemented.

Source spec: `2026-06-18-replication-validation`.

## Empirical Application Replication

Why it matters: reproducing the schooling and voter-turnout applications would
show the package is useful beyond synthetic DGPs.

Dependency or trigger: locate, license, and document the Stephens-Yang and
Charles-Stephens replication data.

Source spec: `2026-06-18-replication-validation`.

## Extended Inference Modes

Why it matters: applied users will eventually want robust, clustered,
repeated-split, absolute-selection, and benchmark-estimator covariance options.
These should not inherit paper-baseline standard errors without separate
identification of the sampling target and simulation validation.

Dependency or trigger: complete a new design/validation spec for each extension.

Source spec: `2026-06-22-inference-design`.
