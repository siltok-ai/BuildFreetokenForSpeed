"""Serially validate all three models using bundled Python and a clean PATH."""
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
APP = Path("C:/workspace/build/ftwin1-review-001/app")
SOURCE = Path("C:/workspace/worktrees/speed-freetoken-win1")
CATALOG = Path("C:/workspace/artifacts/speed-freetoken-win1/model-catalog/weights")
CACHE = APP / "runtime/kernel-cache"
environment = os.environ.copy()
for name in list(environment):
    if name in {"CUDA_HOME", "CUDA_PATH", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV",
                "FREETOKEN_ALLOW_INVALID_MODEL", "FREETOKEN_INVALID_EXPERT_KEEP_LAYERS",
                "FREETOKEN_KERNEL_CACHE_DIR", "SILITOK_FREETOKEN_KERNEL_CACHE_DIR",
                "SILITOK_FREETOKEN_RUNTIME_DIR"} or name.startswith("CUDA_PATH_V"):
        environment.pop(name)
system = Path(os.environ["SystemRoot"])
environment["PATH"] = os.pathsep.join(str(p) for p in (
    APP / "python", APP / "python/Scripts", system / "System32", system,
    system / "System32/WindowsPowerShell/v1.0"))
environment.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", SPEED_REVIEW_APP=str(APP))
records = []
for profile, weights in (("qwen36", "qwen36"), ("qwen38", "qwen38"), ("flash-next", "flash-next")):
    for mode in ("http", "backend"):
        output = HERE / f"packaged-{profile}-{mode}.json"
        script = SOURCE / ("tools/smoke/freetoken_qwen36.py" if mode == "http" else "tools/bench/benchmark_freetoken_qwen36.py")
        arguments = ["--profile", profile, "--model", str(CATALOG / weights), "--quant", "nvfp4",
                     "--kernel-cache", str(CACHE), "--startup-timeout", "600", "--output", str(output)]
        if mode == "http":
            arguments += ["--vram-safety-gib", "0.75" if profile == "qwen38" else "2",
                          "--commit-safety-gib", "1" if profile == "flash-next" else "8"]
        else:
            arguments += ["--repeats", "2", "--long-repeats", "1", "--tokens", "32", "--long-prompt-tokens", "1024"]
        if profile == "flash-next":
            arguments += ["--invalid-expert-keep-layers", "20", "--vram-estimate-gib", "16", "--host-commit-estimate-gib", "32"]
        command = [str(APP / "python/python.exe"), "-I", "-B", str(HERE / "packaged_child.py"), str(script), *arguments]
        environment["SPEED_REVIEW_IDENTITY_PATH"] = str(HERE / f"packaged-{profile}-{mode}-identity.json")
        print(f"START {profile} {mode}", flush=True)
        started = time.perf_counter()
        with (HERE / f"packaged-{profile}-{mode}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=APP, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        record = {"profile": profile, "mode": mode, "returncode": result.returncode,
                  "wall_seconds": time.perf_counter() - started, "command": command,
                  "output": output.name, "invalid_model": profile == "flash-next"}
        records.append(record)
        (HERE / "packaged-models-summary.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"DONE {profile} {mode}: exit={result.returncode}, seconds={record['wall_seconds']:.2f}", flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
