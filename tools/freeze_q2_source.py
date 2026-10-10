"""Freeze the isolated engine worktree as a replayable Q2 experiment profile."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if any((root / name).exists() for name in (
        "patches/series/q2exp1-windows-and-experts.patch", "patches/q2exp1-series.json",
        "build/profiles/q2exp1.json",
    )):
        raise FileExistsError("q2exp1 is already frozen; use a new custom revision and new paths")
    base = "078132480ea4a0424f677a27eb22085624d331e6"
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=args.source)
    if git("rev-parse", "HEAD").decode().strip() != base:
        raise ValueError("engine worktree does not match the frozen base")
    tracked = git("diff", "--binary", "HEAD").decode("utf-8")
    files = git("diff", "--name-only", "HEAD").decode().splitlines()
    for name in git("ls-files", "--others", "--exclude-standard").decode().splitlines():
        path = args.source / name
        text = path.read_text(encoding="utf-8")
        tracked += f"diff --git a/{name} b/{name}\nnew file mode 100644\n"
        tracked += "".join(difflib.unified_diff([], text.splitlines(keepends=True), fromfile="/dev/null", tofile=f"b/{name}"))
        files.append(name)
    patch = root / "patches/series/q2exp1-windows-and-experts.patch"
    patch.write_text(tracked, encoding="utf-8", newline="\n")
    series_path = root / "patches/q2exp1-series.json"
    write(series_path, {"format": 1, "upstream_commit": base, "patches": [{
        "path": patch.relative_to(root).as_posix(), "sha256": digest(patch), "kind": "experiment",
        "id": "FTQ2-001", "title": "Windows runtime/build with complete native Q2_0 experts",
        "status": "active", "upstream_status": "local experiment; no expert aliases", "files": sorted(files)}]})
    old = json.loads((root / "build/profiles/win1.json").read_text())
    old.update(custom_version="0.1.3+speed.q2exp1",
               wheel_filename="freetoken-0.1.3+speed.q2exp1-cp312-cp312-win_amd64.whl",
               patch_series=series_path.relative_to(root).as_posix())
    names = [old[key] for key in ("upstream_lock", "source_archive", "patch_series", "spec_file", "requirements_lock")]
    names.append(patch.relative_to(root).as_posix())
    old["input_sha256"] = {name: digest(root / name) for name in names}
    write(root / "build/profiles/q2exp1.json", old)
    print(json.dumps({"files": len(files), "patch_sha256": digest(patch), "version": old["custom_version"]}), flush=True)


if __name__ == "__main__":
    main()
