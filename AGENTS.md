# Repository organization

- External/user-provided reference material belongs under `references/`.
- Project-owned documentation belongs under `docs/`.
- Current normative behavior belongs under `docs/spec/`.
- Architecture and design rationale belong under `docs/design/`.
- Validation and measurement evidence belongs under `docs/verification/`.
- Temporary implementation plans and Codex instructions belong under `docs/work-notes/`.
- Historical documents belong under `docs/archive/` and are not normative.
- Automated test data belongs under `tests/data/`.
- Human-readable validation screenshots belong under `docs/verification/assets/` unless they are actual automated-test inputs.

# Documentation policy

- Look for an existing canonical document before creating another design/spec document.
- Do not treat work-notes or archive documents as the current specification.
- Do not modify external reference material; record project interpretations in `docs/`.
- When implementation behavior intentionally changes, update the relevant current specification.
- Keep permanent design decisions out of temporary Codex instruction files.
- Read only documentation relevant to the task, not the entire document tree.
- Follow the directory-specific rules in `references/AGENTS.md`, `docs/AGENTS.md`, and `tests/AGENTS.md`.

# Change policy

- Preserve existing behavior unless the task explicitly requests a behavior change.
- Structural refactors must not silently alter DSP, measurement, protocol, calibration, or RF-test behavior.
- Prefer `git mv` for repository reorganizations. Preserve unrelated working-tree changes and do not delete files based on apparent age.
- Fix links, imports, and data paths after moves; do not regenerate fixtures or change test expectations to accommodate a move.
- Run the relevant test suite after structural changes; run the full suite for broad test-tree reorganizations and distinguish pre-existing failures from regressions.
