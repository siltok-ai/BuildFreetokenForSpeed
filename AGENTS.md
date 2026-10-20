# Working boundaries

Read README.md and docs/maintenance.md before editing source or build/release tools.

- `upstream/FreeToken/` is an unmodified official snapshot. Develop in a new output tree prepared by `tools/prepare_source.py`; save engine changes as ordered patches.
- `integration/speed/current-release.json` selects the current profile, materials, model sources and preset. Maintain current usage and build instructions; keep exploration logs and run evidence outside this repository.
- Preserve current patch and accepted wheel/DLL bytes. Changed engine source requires a new version. Historical versions can be retrieved from Git.
- This repository owns FreeToken customization, build materials and model preparation. Speed owns application adapters, HTTP, services, authorization and installer behavior.
- After changes run `python -B tools/validate_repository.py` and `python -B -m unittest discover -s tests -v`. Install the tool test dependency from `tools/requirements-test.txt` first. Offline checks do not constitute GPU or installer acceptance.
- Resource package changes must preserve the documented fixed sources, relative layout and OFF/ON behavior. Check generated package content against the matching reference package.
- Keep third-party licenses and attribution. Do not submit to official FreeToken without explicit user authorization.
