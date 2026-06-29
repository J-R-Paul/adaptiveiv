# Project Log

## 2026-06-18 - 2026-06-18-replication-validation

Outcome: done.

Added paper-style DGP constructors, validation helpers, a rerunnable Monte Carlo
validation script, tests, docs, and package artifact inclusion for validation
materials. Recommended lightweight validation generated
`validation/outputs/latest/report.md` with 150 estimator rows and 9/9 qualitative
checks passed. Future work: exact Tables 2-4 replication with LIML/UJIVE/IJIVE/lasso
and empirical application replication should be separate specs.

## 2026-06-18 - 2026-06-18-repeated-splits

Outcome: done.

Added `AdaptiveIV.fit(n_splits=...)` for repeated split-sample estimation,
per-repetition component diagnostics, split-stability summaries, validation
script support for `--n-splits`, and docs for the repeated validation command.
The package still does not expose standard errors or p-values; inference remains
a separate validation problem.

## 2026-06-22 - 2026-06-22-inference-design

Outcome: done.

Completed a detailed inference design spec for `adaptiveiv`. The design makes
paper-baseline homoskedastic inference the first public target, keeps repeated
split stability separate from standard errors, leaves robust/clustered/bootstrap
extensions unsupported until separately validated, defines the result-object and
validation contract needed before exposing finite `bse`, p-values, or confidence
intervals. The first paper-baseline implementation slice was completed later the
same day.

## 2026-06-22 - inference implementation

Outcome: done.

Implemented the first inference slice from the design spec: paper-baseline
homoskedastic inference for supported select-and-interact fits with
`n_splits=1` and `selection_rule="positive"`. Added explicit
`cov_type="none"` point-estimate-only mode, result metadata
(`cov_estimator`, `inference_notes`), an inference validation runner, docs, and
tests. Repeated-split, absolute-selection, robust, clustered, bootstrap, and
benchmark-estimator inference remain unsupported unless separately validated.

## 2026-06-23 - 2026-06-23-paper-table-validation

Outcome: done.

Added a direct paper-table comparison layer for the implemented Section 4
estimators. `adaptiveiv.paper_benchmarks` now contains transcribed Tables 2-4
targets for `2SLS-P`, `2SLS-INT`, `2SLS-SSINT`, `2SLS-INF`, and `2SLS-ADPT`.
`simulations/validate_paper_tables.py` writes paper targets, config manifests,
simulated summaries, observed-vs-paper deviations, checks, and a Markdown
report. `simulations/aggregate_paper_table_chunks.py` combines chunks and checks
configuration coverage. Smoke validation, chunked artifact validation, a
100-repetition Table 2 DGP1 normal chunk, and the first full 500-repetition
paper configuration passed. The remaining release-evidence step is to run and
aggregate the full 500-repetition chunks for all 30 implemented-method paper
configurations.

## 2026-06-23 - 2026-06-23-inference-implementation-hardening

Outcome: done.

Hardened the first `paper_homoskedastic` inference implementation against the
completed inference design spec. Component-level residual variance now uses each
split component's own select-and-interact coefficient, and `conf_int(alpha=...)`
supports arbitrary two-sided normal-approximation intervals for valid alpha
values. Focused estimator/API tests, the docs inference-validation example, and
the inference smoke runner passed.

## 2026-06-23 - adaptive selector paper-table hardening

Outcome: in progress.

Changed the adaptive selector to choose the paper's risk-minimizing `K_hat` from
full-sample order statistics, then select the top `K_hat` positive-strength
groups in each cross-fit selection split. Also fixed paper-table validation
seeds so chunks use the original paper `config_index`, making chunked runs
stable under slicing. Corrected the DGP3 generator to use the paper's
untruncated normal mixture for nonzero first-stage coefficients. Targeted
100-repetition diagnostics for the previously worst chi-square adaptive rows
passed. Full 500-repetition chunks cover all 30 implemented-method paper
configurations and match all 300 paper-target rows, but aggregate release checks
remain blocked by Table 4 / DGP3 deviations and rare heavy-tail pooled MSE rows.

## 2026-06-23 - 2026-06-23-inference-diagnostics

Outcome: done.

Exposed component-level `paper_homoskedastic` inference diagnostics through
`AdaptiveIVResults.inference_diagnostics` and added flat variance columns to
`simulations/validate_inference.py` artifacts. The diagnostics table reports
split components `a` and `b` plus the final averaged estimator, while
point-estimate-only fits continue to raise `NotImplementedError` for inference
accessors. Full tests, lint, typing, docs build, and inference smoke validation
passed.

## 2026-06-23 - DGP3 paper-table strength audit metadata

Outcome: in progress.

Added DGP3 strength audit metadata to the paper-table validation runner. DGP3
config manifests and simulation rows now carry the fixed strength mode, seed,
nonzero count, sum of squares, and full realized strength vector. Focused
diagnostics indicate that oracle-threshold choice alone does not explain the
remaining Table 4 gap; the unresolved issue is still the paper's unpublished
realized DGP3 strength vectors and possibly other unrecovered simulation state.

## 2026-06-23 - DGP3 independent strength-seed diagnostics

Outcome: in progress.

Added `--dgp3-strength-seed-base` and `--dgp3-strength-seed` to the paper-table
validation runner so DGP3 fixed-strength vectors can vary independently from
observation-level simulation seeds. Added `--dgp3-strength-seed-map` for
per-configuration reconstruction diagnostics. A full 500-repetition Table 4 /
DGP3 / normal / G=40 run with exact strength seed `11890` passed all strict
paper-table checks, showing that the previous gap is consistent with missing
realized DGP3 strength state rather than a confirmed estimator bug. A
100-repetition six-config seed-map diagnostic passed release tolerance but still
showed large chi-square MSE deviations, so this remains candidate
reconstruction evidence, not proof of the paper's original seeds.

## 2026-06-23 - DGP3 paper-table report provenance

Outcome: in progress.

Paper-table reports now record DGP3 strength-vector provenance in the Markdown
settings block: fixed versus redraw mode, strength-seed base, exact strength
seed, and per-configuration strength-seed map when supplied. This keeps Table 4
reconstruction diagnostics auditable from the report itself, while the
underlying manifests and simulation rows continue to carry the full realized
strength-vector metadata.

## 2026-06-23 - validation tail diagnostics

Outcome: in progress.

Validation summaries and paper-comparison artifacts now carry tail-error
diagnostics: max absolute error, 95th and 99th percentile absolute error,
maximum scaled squared error, and the share of MSE from the largest
absolute-error realization. Re-aggregating the full paper-table chunks still
fails the strict numerical confirmation checks, but the report now makes clear
which MSE deviations are tail-driven and which, such as DGP3 normal rows, are
not. The source PDF text confirms the very large Table 3 / DGP2 / chi-square /
G=100 pooled MSE target is not a transcription typo.

## 2026-06-23 - best-current DGP3 seed-map reconstruction

Outcome: in progress.

Ran full-preset DGP3 reconstruction diagnostics with refined fixed-strength
seeds. Seed `39414` strictly confirms both G=200 Table 4 rows at 500
repetitions for normal and chi-square errors. The best-current six-config DGP3
map is `24=11890,25=11890,26=39414,27=40437,28=11890,29=39414`; it matches all
60 Table 4 comparison rows and passes scaled MAD under the strict 25 percent
tolerance, but still fails scaled MSE because G=40 chi-square pooled 2SLS is
dominated by a tail event. The package still cannot claim full numerical
confirmation against the paper.

## 2026-06-23 - 2026-06-23-inference-ecosystem-api

Outcome: done.

Aligned the first supported inference result with common Python statistics API
expectations without changing the covariance formula or widening inference
eligibility. Supported `paper_homoskedastic` fits now expose `std_errors`,
`cov_params()`, and `reference_distribution="normal"` alongside the existing
`bse`, `cov`, p-values, intervals, and diagnostics. Point-estimate-only fits
continue to raise explicit unavailable-inference errors through the aliases.

## 2026-06-23 - implied paper-tail diagnostics

Outcome: in progress.

Added paper-comparison diagnostics for heavy-tailed MSE blockers. For
`scaled_mse` rows where the paper target exceeds the observed run, artifacts now
report the single-tail-event absolute error, scaled squared error, ratio to the
largest observed error, and paper-implied MSE share needed to match the paper
target while leaving other replications unchanged. Rebuilt
`validation/outputs/paper_tables/full_combined_reconstructed_dgp3`; the
remaining strict all-table failure is still pooled chi-square MSE, but the
largest row now shows that matching Table 3 / DGP2 / chi-square / G=100 pooled
MSE would require an absolute error about `994.649`, roughly `89.39` times the
largest observed error.

## 2026-06-23 - pooled tail seed scan diagnostic

Outcome: in progress.

Added `simulations/diagnose_pooled_tail_seeds.py`, a fast cross-product seed
scanner for pooled 2SLS tail events in paper Section 4 configurations. The
diagnostic matches the package pooled estimator but avoids fitting every
estimator, making it practical to audit rare near-zero denominator events. A
10,000-seed scan for Table 3 / DGP2 / chi-square / G=100 found 5 seeds with
`|beta_hat| > 100`, 2 with `|beta_hat| > 500`, and 1 with
`|beta_hat| > 1000`; the largest was seed `459030378` with
`beta_hat=-2009.722` and denominator `0.597888`.

## 2026-06-23 - 2026-06-23-public-release-metadata

Outcome: done.

Hardened public package metadata for release. Added an MIT `LICENSE`,
`project.license`, PyPI classifiers, and pip-installable `docs` and `examples`
extras while keeping uv dependency groups. Packaging tests now check metadata
and sdist contents, including the license and public validation scripts. Built
sdist/wheel metadata reports `License-Expression: MIT`, `License-File:
LICENSE`, classifiers, and extras.

## 2026-06-23 - 2026-06-23-inference-request-gating

Outcome: done.

Tightened the implemented inference contract so analytic homoskedastic
inference is computed only when a supported covariance request is active.
`cov_type="none"` now stays a pure point-estimate-and-diagnostics path rather
than computing hidden inference quantities and discarding them.

## 2026-06-23 - 2026-06-23-release-readiness-audit

Outcome: done.

Added `simulations/audit_release_readiness.py`, a strict release audit that
reads qualitative replication, inference validation, full paper-table
comparison, and distribution-artifact outputs, then writes CSV/JSON/Markdown
audit artifacts. The current audit correctly reports `ready=false` with
`paper_table_validation` as the only failed gate, because
`scaled_mse_within_relative_tolerance` remains red.

## 2026-06-24 - 2026-06-24-pooled-mse-tail-targets

Outcome: done.

Added `simulations/diagnose_pooled_mse_targets.py` to read failed pooled
`scaled_mse` paper-comparison rows and rank real observation-level seeds by
closeness to the paper-implied single-tail event. The diagnostic classifies the
Table 2 / DGP1 / chi-square / G=100 pooled row as an observed overshoot, finds a
near-match for the Table 3 / DGP2 / chi-square / G=40 missing-tail row in a
1,000-seed scan, and shows that the Table 3 / DGP2 / chi-square / G=100 paper
MSE still needs a much rarer tail seed or better reconstruction of original
simulation state.

## 2026-06-24 - 2026-06-24-pooled-rng-convention-audit

Outcome: done.

Added `simulations/audit_pooled_rng_conventions.py` to compare pooled 2SLS MSE
under current, fixed-order, and same-RNG shuffled DGP1/DGP2 strength assignment
conventions. The diagnostic reproduces the current pooled row, shows that
simple convention changes can move the smaller failed pooled chi-square rows,
and rules out these conventions as an explanation for the very large Table 3 /
DGP2 / chi-square / G=100 pooled MSE target.

## 2026-06-24 - 2026-06-24-pooled-tail-splice

Outcome: done.

Added `simulations/diagnose_pooled_tail_splice.py` to test one-tail
counterfactual replacements for failed pooled chi-square MSE paper-table rows.
The diagnostic ranks real high-tail candidate seeds by spliced paper-target
error and writes a Markdown report plus CSV artifacts. Current reconstruction
evidence finds one candidate for config 18 and one for config 19 inside the
25 percent tolerance, supporting a rare-tail explanation while leaving release
readiness red because the original paper seeds remain unidentified.

## 2026-06-24 - 2026-06-24-inference-support-contract

Outcome: done.

Implemented a machine-readable inference support contract around the existing
paper-baseline inference design. `InferenceSupport`,
`AdaptiveIV.inference_support(...)`, and `AdaptiveIV.supports_inference(...)`
let downstream code check whether a proposed covariance request has validated
support before fitting. The change improves ecosystem integration without
adding new covariance estimators or changing unsupported-inference behavior.

## 2026-06-24 - 2026-06-24-pooled-error-convention-audit

Outcome: done.

Added `simulations/audit_pooled_error_conventions.py` to test whether the
remaining pooled chi-square MSE paper-table failures are explained by
centering or scaling of chi-square errors. The audit compares centered, raw,
standardized, and raw-standardized conventions. Current evidence shows raw and
centered conventions coincide under the pooled intercept, while standardized
conventions move the failed rows and the broader DGP1/DGP2 chi-square rows much
farther from the paper targets. This rules out a simple chi-square
centering/scaling mismatch; release readiness remains blocked by
`paper_table_validation`.

## 2026-06-24 - 2026-06-24-observation-seed-reconstruction

Outcome: done.

Added `simulations/reconstruct_paper_table_seeds.py` to replace selected
observation-level Monte Carlo replications and recompute all implemented
estimators for those full replication blocks. The reconstruction artifact
`validation/outputs/paper_tables/full_combined_reconstructed_observation_seeds`
uses replacements `7:229:1`, `18:40:3585863902`, and `19:369:2612616469`; it
passes all four paper-table checks under the existing 25 percent relative
tolerance with maximum absolute scaled-MSE relative error `0.236256`. This is
strong evidence that the final strict paper-table blocker is rare simulation
tail state, not an estimator formula mismatch, but it remains reconstruction
evidence rather than recovered original paper seeds.

## 2026-06-24 - 2026-06-24-release-evidence-audit

Outcome: done.

Updated the release-readiness audit to distinguish required gates from optional
supporting evidence. At that point, the unreplaced full-combined paper-table
artifact was still the numerical-confirmation gate, while
`full_combined_reconstructed_observation_seeds` is reported as non-required
supporting evidence. The refreshed audit therefore stays `Ready: False` with
failed gate `paper_table_validation`, but now also records that reconstructed
observation-seed evidence passes 4/4 checks. Packaging tests now also assert
that generated validation outputs, internal specs, built site files, and Python
cache files are absent from the source distribution.

Superseded release-scope note: as of
`2026-06-24-release-scope-replication-limits`, exact paper-table replication is
reported as non-blocking evidence rather than a required release gate.

## 2026-06-24 - 2026-06-24-paper-method-coverage

Outcome: done.

Added machine-readable coverage metadata for the nine Section 4 Monte Carlo
method labels in the paper. This initially showed that `adaptiveiv` implemented
and validated `2SLS-P`, `2SLS-INT`, `2SLS-SSINT`, `2SLS-INF`, and `2SLS-ADPT`,
but not `LIML-INT`, `2SLS-SSL`, `UJIVE`, or `IJIVE`. Later LIML work moved
`LIML-INT` into the implemented and target-transcribed set, while `2SLS-SSL`,
`UJIVE`, and `IJIVE` remain missing. At that point, the release-readiness audit
included a `paper_method_coverage` gate to avoid overstating original-paper
method coverage while competitor-method columns remained missing.

Superseded release-scope note: as of
`2026-06-24-release-scope-replication-limits`, full original-paper method
coverage is reported as a non-blocking limitation rather than a required release
gate.

## 2026-06-24 - 2026-06-24-liml-int-benchmark

Outcome: done.

Added a point-estimate-only `liml_interacted` benchmark estimator using a
self-contained k-class LIML calculation over group-interacted, groupwise
residualized instruments. `estimate_methods_once(...,
methods=["liml_interacted"])` now produces a finite validation row on
paper-style DGPs, and method coverage has moved from five to six implemented
paper methods. A later target-transcription step added LIML-INT MAD targets, so
the remaining method-coverage blockers are `2SLS-SSL`, `UJIVE`, and `IJIVE`. A
forced no-editable uv reinstall is now the verified way to smoke-test fresh
local package code when the version remains `0.1.0`: `uv run --no-editable
--reinstall-package adaptiveiv --offline`.

## 2026-06-24 - 2026-06-24-liml-paper-targets

Outcome: done.

Transcribed the paper-reported LIML-INT `N x MAD` targets from Tables 2-4 and
included the existing LIML-INT point-estimate benchmark in default paper-table
validation runs. Added a release-audit freshness gate that compares generated
`paper_targets.csv` row counts to the current in-code target table so stale
paper-table artifacts are treated as required blockers. Remaining release
blockers are the canonical paper-table numerical discrepancy and unimplemented
2SLS-SSL, UJIVE, and IJIVE comparators.

## 2026-06-24 - 2026-06-24-release-scope-replication-limits

Outcome: done.

Updated the current release scope so exact original-paper Tables 2-4
replication is a documented limitation rather than a required release gate. The
release audit still reports paper-table numerical checks, paper-target artifact
freshness, and full paper-method coverage as non-blocking evidence. The
limitation note states that the current package does not exactly replicate all
original paper tables, likely because original simulation seeds/state are
unrecovered and some external comparators remain unimplemented.
