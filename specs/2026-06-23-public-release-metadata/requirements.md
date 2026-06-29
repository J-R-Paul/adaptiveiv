# Requirements

Spec: 2026-06-23-public-release-metadata
Created: 2026-06-23
Status: done
Type: packaging / public release
Summary: Harden public package metadata and distribution contents for a PyPI-style release.

## Goal

Make the package metadata credible for public distribution by declaring a
license, standard classifiers, installable optional extras, and by ensuring the
license and validation materials are included in source distributions.

## Scope

- Add explicit license metadata.
- Add public package classifiers.
- Add `docs` and `examples` optional dependencies for pip users while retaining
  uv dependency groups for development.
- Include the license file in source and wheel artifacts.
- Test that metadata and sdist contents are present.

## Out Of Scope

- Choosing a repository URL before the public remote exists.
- Publishing to PyPI.
- Changing the package version.
- Resolving the remaining pooled chi-square paper-table MSE tail-state blocker.

## Success Looks Like

- `pyproject.toml` declares MIT license metadata, classifiers, and extras.
- The built sdist includes `LICENSE`, validation docs, and public simulation
  diagnostics.
- The built wheel metadata reports the license, classifiers, and extras.
- Packaging tests, full tests, lint, typing, docs, build, and clean no-editable
  smoke checks pass.

## Outcome

Completed on 2026-06-23. The package now has MIT license metadata, classifiers,
docs/examples extras, and built artifacts include the license file.
