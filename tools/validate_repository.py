"""Offline validation of upstream snapshot, patch equivalence and native artifacts."""

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from prepare_source import ROOT, apply_patch, extract_archive, load_profile, prepare, repository_path, sha256


def tree_hashes(root):
    return {path.relative_to(root).as_posix(): sha256(path) for path in root.rglob("*") if path.is_file()}


def main():
    profile, series = load_profile()
    lock = json.loads(repository_path(profile["upstream_lock"]).read_text(encoding="utf-8"))
    if lock["snapshot_modified"]:
        raise ValueError("official browsing snapshot must be unmodified")
    with tempfile.TemporaryDirectory(prefix="speed-freetoken-validate-") as temporary:
        temporary = Path(temporary)
        original = temporary / "original"
        extract_archive(repository_path(profile["source_archive"]), original)
        original_hashes = tree_hashes(original)
        if original_hashes != tree_hashes(repository_path(lock["browse_tree"])):
            raise ValueError("browsable upstream snapshot differs from the official archive")
        stacked, _ = prepare(ROOT / "build/profiles/win1.json", temporary / "stacked")
        legacy = temporary / "legacy"
        extract_archive(repository_path(profile["source_archive"]), legacy)
        old_patch = repository_path("patches/legacy/windows-source.patch")
        if sha256(old_patch) != series["equivalent_legacy_patch_sha256"]:
            raise ValueError("historical patch checksum mismatch")
        apply_patch(legacy, old_patch, temporary / "legacy.log")
        stacked_hashes = tree_hashes(stacked)
        if stacked_hashes != tree_hashes(legacy):
            raise ValueError("split patch series does not reproduce historical build source")
        changed = {name for name in original_hashes.keys() | stacked_hashes.keys()
                   if original_hashes.get(name) != stacked_hashes.get(name)}
        declared = [name for patch in series["patches"] for name in patch["files"]]
        if len(declared) != len(set(declared)) or changed != set(declared):
            raise ValueError("patch file catalog does not describe the actual changes")
        with zipfile.ZipFile(ROOT / "artifacts/win1-cp312-cu130-sm120/wheels" / profile["wheel_filename"]) as wheel:
            for name in wheel.namelist():
                if name.startswith("freetoken/") and name.endswith(".py"):
                    if wheel.read(name) != (stacked / "python" / name).read_bytes():
                        raise ValueError(f"historical wheel Python source mismatch: {name}")
    artifacts = ROOT / "artifacts/win1-cp312-cu130-sm120"
    record = json.loads((artifacts / "build-record.json").read_text(encoding="utf-8"))
    for key in ("source_commit", "custom_version", "wheel_filename", "torch", "cuda", "apache_tvm_ffi", "sm"):
        if record[key] != profile[key]:
            raise ValueError(f"historical artifact contract differs from profile: {key}")
    for old, new in {"source/base.zip": profile["source_archive"],
                     "source/windows-source.patch": "patches/legacy/windows-source.patch",
                     "speed-specs.txt": profile["spec_file"]}.items():
        if record["input_sha256"][old] != profile["input_sha256"][new]:
            raise ValueError(f"historical provenance changed: {old}")
    specs = repository_path(profile["spec_file"]).read_text(encoding="utf-8").splitlines()
    expected_dlls = {f"kernel-cache/{spec}/{spec}.dll" for spec in specs}
    if expected_dlls != {name for name in record["artifacts"] if name.endswith(".dll")}:
        raise ValueError("AOT artifact inventory differs from current specifications")
    contract = json.loads((ROOT / "integration/speed/consumer-contract.json").read_text(encoding="utf-8"))
    for key, expected in {"version": profile["custom_version"], "upstream_commit": profile["source_commit"],
                          "torch": profile["torch"], "cuda": profile["cuda"], "sm": profile["sm"]}.items():
        if contract[key] != expected:
            raise ValueError(f"Speed consumer contract mismatch: {key}")
    if contract["wheel"]["sha256"] != record["wheel_sha256"] or contract["kernel_cache"]["dll_count"] != len(specs):
        raise ValueError("Speed consumer artifact identity mismatch")
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/verify_materials.py"), str(artifacts)], check=True)
    for directory in ("build", "tools", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(json.dumps({"official_files": len(original_hashes), "changed_files": len(changed),
                      "patches": len(series["patches"]), "split_equals_historical": True,
                      "wheel_python_matches_prepared_source": True, "aot_count": len(specs), "verified": True}))


if __name__ == "__main__":
    main()
