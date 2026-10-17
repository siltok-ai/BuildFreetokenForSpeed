# Working boundaries

Read README.md first, then docs/maintenance.md for a source or release change.

- `upstream/FreeToken/` is an unmodified official snapshot. Its AGENTS.md and CONTRIBUTING.md are upstream documents. Do not edit this browsing tree directly; update snapshots only through a reviewed upstream candidate.
- Our engine changes live in `patches/series/`; the current release is selected by `integration/speed/current-release.json` (its profile selects the ordered patch catalog). Historical series remain immutable. Apply the selected profile to a new output tree before development or testing.
- Preserve accepted archives, patch bytes and artifacts. Never silently replace an accepted wheel or reuse its version for different source. `.gitattributes` disables line-ending conversion to preserve frozen SHA-256 identities.
- Speed application adapters belong in Speed. This repository owns FreeToken customization, build inputs, materials, consumer contracts and evidence.
- Run `python -B tools/validate_repository.py` and `python -B -m unittest discover -s tests -v` after changes. These checks do not constitute GPU or packaged Speed acceptance.
- Record actual hardware/runtime validation separately. Expert alias results are invalid model outputs and must remain explicitly marked.
- Do not push or submit changes to the official FreeToken repository without explicit user authorization.
