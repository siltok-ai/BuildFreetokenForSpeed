"""从冻结的源码及 Windows 补丁生成 Speed 使用的 FreeToken wheel 和 AOT DLL。"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from importlib.metadata import version
from email.parser import BytesParser
from packaging.requirements import Requirement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_source import DEFAULT_PROFILE, load_profile, prepare, repository_path


def sha256(path):
    """记录实际制品字节身份，不以目录名或版本号代替校验。"""
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run(command, cwd, log):
    """每阶段独立日志；编译失败立即停止并保留已产生的诊断。"""
    with log.open("w", encoding="utf-8") as output:
        subprocess.run(command, cwd=cwd, stdout=output, stderr=subprocess.STDOUT, check=True)


def main():
    """使用可复制的冻结输入，在新输出目录内完成构建和库存生成。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    args = parser.parse_args()
    kit = Path(__file__).resolve().parents[1]
    inputs, series = load_profile(args.profile)
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if sys.platform != "win32" or sys.version_info[:2] != (3, 12) or sys.maxsize <= 2**32:
        raise RuntimeError("BuildPython must be CPython 3.12 Windows x64")
    dependencies = {}
    for line in repository_path(inputs["requirements_lock"]).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        installed = version(requirement.name)
        if not requirement.specifier.contains(installed, prereleases=True):
            raise RuntimeError(f"BuildPython dependency mismatch: {line}; installed {installed}")
        dependencies[requirement.name] = installed
    compiler = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True).stdout
    if "release 13.0," not in compiler:
        raise RuntimeError("CUDA Toolkit compiler must be 13.0")
    msvc = subprocess.run(["cl"], capture_output=True, text=True)
    compiler_msvc = msvc.stdout + msvc.stderr
    if "19.44." not in compiler_msvc:
        raise RuntimeError("the win1 profile requires MSVC 19.44 (VS2022 toolset 14.44)")
    import torch
    if torch.__version__ != inputs["torch"] or torch.version.cuda != inputs["cuda"]:
        raise RuntimeError("BuildPython Torch/CUDA differs from the frozen build contract")
    if version("apache-tvm-ffi") != inputs["apache_tvm_ffi"]:
        raise RuntimeError("BuildPython TVM FFI differs from the frozen build contract")
    if args.preflight_only:
        print(json.dumps({"verified_inputs": inputs["input_sha256"], "dependencies": dependencies,
                          "cuda_compiler": compiler, "msvc_compiler": compiler_msvc,
                          "status": "preflight_passed"}, indent=2))
        return
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError("use a new output directory; existing artifacts are never overwritten")
    source, inputs = prepare(args.profile, out)
    logs = out / "logs"
    wheels = out / "wheels"
    wheels.mkdir()
    print("building customized FreeToken wheel", flush=True)
    run([sys.executable, str(source / "tools/build_windows_extensions.py"), "bdist_wheel",
         "--dist-dir", str(wheels)], source, logs / "wheel.log")
    wheel = wheels / inputs["wheel_filename"]
    if not wheel.is_file():
        raise FileNotFoundError("custom wheel version/ABI differs from the frozen contract")
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1 or BytesParser().parsebytes(archive.read(metadata[0]))["Version"] != inputs["custom_version"]:
            raise ValueError("wheel metadata differs from the custom version contract")
        native = sorted(name for name in archive.namelist() if name.endswith(".pyd"))
        for module in ("_pinned_tensor", "_cpu_moe", "_row_store"):
            if not any(Path(name).name.startswith(module + ".") for name in native):
                raise ValueError(f"custom wheel lacks native module {module}")
    cache = out / "kernel-cache"
    print("building matching AOT kernel cache", flush=True)
    run([sys.executable, str(source / "tools/build_windows_kernel_cache.py"), "--output", str(cache),
         "--spec-file", str(repository_path(inputs["spec_file"])), "--arch", "12.0", "--jobs", str(args.jobs)],
        source, logs / "aot.log")
    specs = repository_path(inputs["spec_file"]).read_text(encoding="utf-8").splitlines()
    dlls = [cache / spec / (spec + ".dll") for spec in specs]
    if any(not path.is_file() for path in dlls):
        raise ValueError("matching AOT inventory is incomplete")
    metadata_path = cache / "build-metadata.json"
    kernel_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    kernel_metadata["source_commit"] = inputs["source_commit"]
    kernel_metadata["source_patch_series_sha256"] = inputs["input_sha256"][inputs["patch_series"]]
    metadata_path.write_text(json.dumps(kernel_metadata, indent=2) + "\n", encoding="utf-8")
    manifest = {"schema_version": 1, "freetoken_version": inputs["custom_version"],
                "torch_version": inputs["torch"], "torch_cuda": inputs["cuda"],
                "python_minor": [3, 12], "cuda_capabilities": [[12, 0]], "spec_names": specs,
                "artifacts": [{"path": path.relative_to(cache).as_posix(), "size": path.stat().st_size}
                              for path in dlls]}
    (cache / "speed-kernel-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    inventory = {str(path.relative_to(out)).replace("\\", "/"): {"bytes": path.stat().st_size,
                 "sha256": sha256(path)} for path in [wheel, *dlls]}
    record = {"format": 1, "status": "built_requires_runtime_validation", **inputs,
              "python": sys.version, "build_python": sys.executable, "native_modules": native,
              "dependencies": dependencies, "cuda_compiler": compiler, "msvc_compiler": compiler_msvc,
              "producer_sha256": {name: sha256(kit / name) for name in (
                  "build/build_materials.py", "build/build-materials.ps1", "tools/prepare_source.py",
                  "build/requirements.lock.txt")},
              "patches": series["patches"],
              "artifacts": inventory, "wheel_sha256": sha256(wheel), "aot_count": len(dlls)}
    (out / "build-record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wheel": str(wheel), "wheel_sha256": record["wheel_sha256"],
                      "aot_count": len(dlls), "status": record["status"]}), flush=True)


if __name__ == "__main__":
    main()
