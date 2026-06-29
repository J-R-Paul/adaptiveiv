# Validation

## Completion Standard

This design work item is complete when the inference design spec identifies the
inferential target, supported and unsupported covariance modes, result-object
contract, repeated-split restrictions, validation requirements, and open design
questions.

At design closeout, inference itself was intentionally deferred to a later
implementation stage. The first paper-baseline implementation slice was
completed later on 2026-06-22 with its own algebraic, simulation,
documentation, and packaging checks.

## Design Review Checklist

- The spec names the first public covariance mode.
- The spec does not treat repeated split components as independent draws.
- The spec distinguishes select-and-interact from invalid select-and-pool
  inference.
- The spec keeps robust, clustered, bootstrap, and absolute-sign inference as
  separately validated extensions.
- The spec says how `AdaptiveIVResults` should behave when inference is
  available and unavailable.
- The spec contains a release validation gate for finite standard errors.
- The spec is compatible with the existing Python statistics-style API.

## Final Result

Design spec completed on 2026-06-22.

No code was changed. No statistical inference was exposed.

The completed artifact is `specs/2026-06-22-inference-design/design.md`.

## Evidence Consulted

- Existing package source in `src/adaptiveiv/`.
- Current README, docs, tests, and package-level `plan.md`.
- Previous validation and repeated-split specs.
- Local paper PDF `../Abadie-Gu-Shen.pdf`, especially:
  - the warning that first-stage select-and-pool can invalidate conventional
    inference,
  - the split-sample select-and-interact definition,
  - Lemma 1's first-order asymptotic normality statement,
  - Theorem 3's adaptive threshold selection logic,
  - simulation tables reporting rejection rates as robustness checks.

## Sign-Off

Done on 2026-06-22.

Fresh checks performed at closeout:

- The spec folder contains `requirements.md`, `design.md`, `plan.md`,
  `validation.md`, and `decisions.md`.
- Placeholder scan over `specs/2026-06-22-inference-design`, `specs/log.md`,
  and `specs/backlog.md` found no common placeholder marker tokens.
- ASCII scan over the same files found no non-ASCII characters.
- The design was line-reviewed after writing and patched to avoid assuming that
  cross-fit component estimates are unconditionally independent.
