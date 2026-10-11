"""Verify frozen inputs, then prepare a separate patched FreeToken source tree."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "build/profiles/q2opt2.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def repository_path(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"input escapes repository: {name}")
    return path


def load_profile(path=DEFAULT_PROFILE):
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {profile[key] for key in (
        "upstream_lock", "source_archive", "patch_series", "spec_file", "requirements_lock")}
    if not required <= profile["input_sha256"].keys():
        raise ValueError("profile must freeze every declared build input")
    for name, expected in profile["input_sha256"].items():
        if sha256(repository_path(name)) != expected:
            raise ValueError(f"frozen build input changed: {name}")
    lock = json.loads(repository_path(profile["upstream_lock"]).read_text(encoding="utf-8"))
    series = json.loads(repository_path(profile["patch_series"]).read_text(encoding="utf-8"))
    if lock["commit"] != profile["source_commit"] or series["upstream_commit"] != lock["commit"]:
        raise ValueError("upstream/profile/patch series commit mismatch")
    if lock["archive"] != profile["source_archive"] or lock["archive_sha256"] != sha256(repository_path(lock["archive"])):
        raise ValueError("upstream archive identity mismatch")
    for patch in series["patches"]:
        if profile["input_sha256"].get(patch["path"]) != patch["sha256"]:
            raise ValueError(f"patch not frozen by profile: {patch['path']}")
    return profile, series


def extract_archive(archive_path, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            if not (destination / entry.filename).resolve().is_relative_to(destination):
                raise ValueError("source archive path escapes output directory")
        destination.mkdir(parents=True, exist_ok=False)
        archive.extractall(destination)


def apply_patch(source, patch, log):
    # Prevent an output nested inside this repository from inheriting its Git root.
    environment = os.environ.copy()
    environment["GIT_CEILING_DIRECTORIES"] = str(Path(source).resolve().parent)
    with Path(log).open("w", encoding="utf-8") as output:
        for flags in (["--check"], []):
            # 冻结归档与已验收 wheel 使用 CRLF；重放不能依赖调用者的 Git 行尾配置。
            subprocess.run(["git", "-c", "core.autocrlf=true", "apply", *flags, str(Path(patch).resolve())],
                           cwd=source, env=environment, stdout=output,
                           stderr=subprocess.STDOUT, check=True)


def prepare(profile_path, output):
    profile, series = load_profile(profile_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("use a new output directory; existing outputs are never overwritten")
    output.mkdir(parents=True)
    source = output / "build-source"
    extract_archive(repository_path(profile["source_archive"]), source)
    logs = output / "logs"
    logs.mkdir()
    for number, patch in enumerate(series["patches"], 1):
        apply_patch(source, repository_path(patch["path"]), logs / f"patch-{number:04d}.log")
    record = {"source_commit": profile["source_commit"], "custom_version": profile["custom_version"],
              "input_sha256": profile["input_sha256"], "patches": series["patches"]}
    (output / "preparation-record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return source, profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, profile = prepare(args.profile, args.output)
    print(json.dumps({"source": str(source), "version": profile["custom_version"], "verified": True}))


if __name__ == "__main__":
    main()
