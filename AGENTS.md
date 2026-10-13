# Working boundaries

Read README.md and docs/maintenance.md before a source or release change.

- `upstream/FreeToken/` is the unmodified official 0781324 snapshot. Do not edit it.
- Only `patches/qwen1-series.json` and `build/profiles/qwen1.json` are active on this branch.
- This branch serves Qwen3.6-35B-A3B and Qwen3.8-27B text inference. Do not add Flash Next, RowStore, Q2 or alias customization.
- Speed adapters belong in Speed. Use serial Windows expert loading through its adapter.
- Apply patches to a new output tree. Preserve accepted material bytes and use a new custom revision for source changes.
- Historical accepted inputs remain on main and in Git history; do not overwrite them.
- Run `python -B tools/validate_repository.py` and `python -B -m unittest discover -s tests -v` after changes.
- These checks do not constitute GPU acceptance. Bind actual model and packaged acceptance to exact source and binary identities.
- Do not push to official FreeToken without explicit user authorization.
