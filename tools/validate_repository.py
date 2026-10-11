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


def validate_packaged_evidence(contract):
    """核对追加证据与消费合同的身份，不将离线校验当作重新执行硬件验收。"""
    record_path = repository_path(contract["speed_source_identity"])
    evidence = record_path.parent
    index = json.loads((evidence / "evidence.json").read_text(encoding="utf-8"))
    for name, expected in index["files"].items():
        path = (evidence / name).resolve()
        if not path.is_relative_to(evidence.resolve()):
            raise ValueError("packaged evidence path escapes its directory")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"packaged evidence checksum mismatch: {name}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record["speed_commit"] != contract["speed_adapter"]["commit"]
            or record["wheel_sha256"] != contract["wheel"]["sha256"]
            or record["freetoken_materials_commit"] != contract["accepted_materials_commit"]):
        raise ValueError("packaged evidence consumer identity mismatch")
    for key in ("installer_validated", "clean_machine_validated", "normal_full_flash_next_validated"):
        if record[key] != contract[key]:
            raise ValueError(f"packaged evidence scope mismatch: {key}")
    if contract["packaged_speed_validated"] is not True or record["staged_runtime_validated"] is not True:
        raise ValueError("win1 packaged runtime acceptance is not recorded")
    runs = json.loads((evidence / "packaged-models-summary.json").read_text(encoding="utf-8"))
    expected_runs = {(profile, mode) for profile in ("qwen36", "qwen38", "flash-next") for mode in ("http", "backend")}
    if len(runs) != 6 or {(run["profile"], run["mode"]) for run in runs} != expected_runs:
        raise ValueError("packaged evidence lacks the six model runs")
    if any(run["returncode"] != 0 or run["invalid_model"] != (run["profile"] == "flash-next") for run in runs):
        raise ValueError("packaged evidence has a failed or incorrectly labelled model run")


def main():
    profile, series = load_profile(ROOT / "build/profiles/win1.json")
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
    validate_packaged_evidence(contract)
    subprocess.run([sys.executable, "-B", str(ROOT / "tools/verify_materials.py"), str(artifacts)], check=True)
    # 性能修订独立重放并验证新 wheel 的源码/二进制来源，不覆盖历史 win1 验收。
    from validate_python_revision import validate as validate_python_revision
    validate_python_revision(ROOT / "build/profiles/q2opt1.json", ROOT / "artifacts/q2opt1-cp312-cu130-sm120")
    validate_python_revision(ROOT / "build/profiles/q2opt2.json", ROOT / "artifacts/q2opt2-cp312-cu130-sm120")
    current = json.loads((ROOT / "integration/speed/consumer-contract-q2opt2.json").read_text(encoding="utf-8"))
    current_profile, _ = load_profile(ROOT / current["profile"])
    current_materials = repository_path(current["materials"])
    current_record = json.loads((current_materials / "build-record.json").read_text(encoding="utf-8"))
    if (current["version"] != current_profile["custom_version"]
            or current["wheel"]["sha256"] != current_record["wheel_sha256"]
            or current["kernel_cache"]["manifest_sha256"] != sha256(current_materials / current["kernel_cache"]["manifest"])):
        raise ValueError("current Q2 consumer contract differs from actual materials")
    review = repository_path(current["speed_source_identity"])
    evidence = json.loads((review.parent / "evidence.json").read_text(encoding="utf-8"))
    for name, entry in evidence["files"].items():
        path = (review.parent / name).resolve()
        if not path.is_relative_to(review.parent) or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Q2 review evidence identity mismatch: {name}")
    summary = json.loads(review.read_text(encoding="utf-8"))
    if summary["wheel_sha256"] != current_record["wheel_sha256"] or summary["version"] != current["version"]:
        raise ValueError("Q2 review evidence belongs to a different wheel")
    for key in ("packaged_speed_validated", "normal_full_flash_next_validated", "installer_validated", "clean_machine_validated"):
        if summary[key] != current[key]:
            raise ValueError(f"Q2 review acceptance scope differs: {key}")
    for directory in ("build", "tools", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(json.dumps({"official_files": len(original_hashes), "changed_files": len(changed),
                      "patches": len(series["patches"]), "split_equals_historical": True,
                      "wheel_python_matches_prepared_source": True, "aot_count": len(specs),
                      "packaged_acceptance_evidence_verified": True, "verified": True}))


if __name__ == "__main__":
    main()
