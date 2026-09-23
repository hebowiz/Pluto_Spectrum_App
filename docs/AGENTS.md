# Documentation placement

- Current normative specifications belong in `spec/`; architecture and design rationale belong in `design/`.
- Measurement evidence, performance evaluations, and audits belong in `verification/`.
- Temporary Codex instructions, investigation notes, and implementation plans belong in `work-notes/`.
- Use `archive/` only for explicitly legacy, obsolete, or superseded material. It is not normative; age alone is insufficient.
- External/user-provided material belongs in `../references/`, not here. Keep user manuals in `user-manual/` and their images in `images/user-manual/`.
- Before creating a document, find the existing canonical document for that subject. Do not duplicate formal specifications.
- Update the relevant `spec/` when an authorized implementation change changes behavior. Do not promote temporary instructions to formal specifications.
- Follow the explicit-request-only manual policy in `../AGENTS.md`: do not update `user-manual/`, `images/user-manual/`, or generated manual PDFs alongside implementation changes unless the user explicitly requests manual work for that task. This includes the VSA analysis supplement, screenshots, and manual indexes/validation notes; specification and verification maintenance remains required where applicable.
- Consult only documents relevant to the task. Indexes should explain placement and entry points, without fixed branch names, commit hashes, or update dates.
- Classify mixed historical notes by their declared purpose; do not rewrite, merge, or delete their contents as part of a structural move.
