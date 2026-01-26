# Release Process

This document describes how Veriscope releases are versioned, cut, and stabilized.

## Versioning

Veriscope follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes to public contracts (CLI behavior, artifacts, schemas).
- **MINOR**: Backward-compatible new features or extensions.
- **PATCH**: Backward-compatible bug fixes and documentation updates.

Contract and compatibility references live in `docs/contract_v1.md` and
`docs/migration_invariants.md`.

## Cutting a release

1. Ensure `main` is green and all contract checks pass.
2. Update version references (if any) and release notes.
3. Tag the release (`vX.Y.Z`) and publish a GitHub release.
4. Attach any release artifacts if applicable.

## Stability and compatibility

- The artifact schema and validation rules are treated as stable public contracts.
- Breaking changes to artifacts, CLI exit codes, or decision semantics require a MAJOR bump.
- Backward-compatible extensions (new optional fields, additional docs) are MINOR updates.
