"""Serial real-model acceptance with embedded Python and a cleaned PATH.

Uses the reviewed Speed smoke/benchmark scripts, but preimports the compiled
application so their source-path insertion cannot substitute source modules.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def check_report(report, mode):
    """Check retained answers and lifecycle evidence, not only exit status."""
    if mode == "http":
        assert report["worker_pids_gone"]
        assert len(report["quality_probes"]) == 4
        assert all(case["passed"] for case in report["quality_probes"])
        assert report["sync"]["status_code"] == 200
        if report["quant"] == "q2_0":
            # The full storage contract is also validated by the Speed backend.
            assert report["expert_storage"]
    else:
        assert report["worker_pid_gone"] and report["cancel"]["worker_reused"]
        assert report["post_cancel"]["text"].strip() == "worker reused"
        cases = {case["id"]: case["result"] for case in report["quality_cases"]}
        assert cases["arithmetic_no_thinking"]["text"].strip() == "703"
        thinking = cases["reasoning_with_thinking"]
        assert thinking["reasoning_content"] and thinking["finish_reason"] == "stop"
        assert re.search(r"FINAL:\s*76\s*$", thinking["text"])
        structured = cases["long_structured_no_thinking"]
        assert structured["finish_reason"] == "stop" and structured["text"]
        assert len(report["long_runs"]) == 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", type=Path, required=True)
    parser.add_argument("--speed-source", type=Path, required=True)
    parser.add_argument("--flash-next-model", type=Path, required=True)
    parser.add_argument("--qwen36-model", type=Path, required=True)
    parser.add_argument("--qwen38-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs='+', choices=('flash-next', 'qwen36', 'qwen38'),
                        default=['flash-next', 'qwen36', 'qwen38'],
                        help='Select independent model regressions; only all three give full acceptance')
    args = parser.parse_args()
    app, source = args.app.resolve(strict=True), args.speed_source.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=True)
    output_root = args.output.resolve()
    environment = os.environ.copy()
    for name in list(environment):
        if name in {"CUDA_HOME", "CUDA_PATH", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV",
                    "FREETOKEN_ALLOW_INVALID_MODEL", "FREETOKEN_INVALID_EXPERT_KEEP_LAYERS",
                    "FREETOKEN_KERNEL_CACHE_DIR", "SILITOK_FREETOKEN_KERNEL_CACHE_DIR",
                    "SILITOK_FREETOKEN_RUNTIME_DIR"} or name.startswith("CUDA_PATH_V"):
            environment.pop(name)
    system = Path(os.environ["SystemRoot"])
    environment["PATH"] = os.pathsep.join(str(path) for path in (
        app / "python", app / "python/Scripts", system / "System32", system,
        system / "System32/WindowsPowerShell/v1.0"))
    environment.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", SPEED_REVIEW_APP=str(app))
    records = []
    for profile, model in (("flash-next", args.flash_next_model),
                           ("qwen36", args.qwen36_model), ("qwen38", args.qwen38_model)):
        if profile not in args.profiles:
            continue
        for mode in ("http", "backend"):
            report_path = output_root / f"packaged-{profile}-{mode}.json"
            identity_path = output_root / f"packaged-{profile}-{mode}-identity.json"
            script = source / ("tools/smoke/freetoken_qwen36.py" if mode == "http"
                               else "tools/bench/benchmark_freetoken_qwen36.py")
            arguments = ["--profile", profile, "--model", str(model.resolve(strict=True)),
                         "--quant", "q2_0" if profile == "flash-next" else "nvfp4",
                         "--kernel-cache", str(app / "runtime/kernel-cache"),
                         "--startup-timeout", "600", "--output", str(report_path)]
            if mode == "http":
                arguments += ["--vram-safety-gib", "0.75" if profile == "qwen38" else "2",
                              "--commit-safety-gib", "1" if profile == "flash-next" else "8"]
            else:
                arguments += ["--repeats", "2", "--long-repeats", "1", "--tokens", "32",
                              "--long-prompt-tokens", "1024" if profile == "qwen38" else "4096",
                              "--quality"]
            command = [str(app / "python/python.exe"), "-X", "utf8", "-I", "-B",
                       str(Path(__file__).with_name("packaged_speed_child.py")), str(script), *arguments]
            environment["SPEED_REVIEW_IDENTITY_PATH"] = str(identity_path)
            print(f"START {profile} {mode}", flush=True)
            started = time.perf_counter()
            with (output_root / f"packaged-{profile}-{mode}.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=app, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=1800)
            record = {"profile": profile, "mode": mode, "returncode": result.returncode,
                      "wall_seconds": time.perf_counter() - started,
                      "command": command, "output": report_path.name, "identity": identity_path.name}
            records.append(record)
            (output_root / "packaged-models-summary.json").write_text(
                json.dumps(records, indent=2) + "\n", encoding="utf-8")
            print(f"DONE {profile} {mode}: exit={result.returncode}, seconds={record['wall_seconds']:.2f}", flush=True)
            if result.returncode:
                raise RuntimeError(f"failed; see {output_root / f'packaged-{profile}-{mode}.log'}")
            check_report(json.loads(report_path.read_text(encoding="utf-8")), mode)
            identity = json.loads(identity_path.read_text(encoding="utf-8"))
            manifest = identity["build_manifest"]
            modules = manifest["modules"]
            mismatches = []
            for relative, entry in modules.items():
                current = hashlib.sha256((source / "silitok_speed" / relative).read_bytes()).hexdigest()
                if current != entry["source_sha256"]:
                    mismatches.append(relative)
            if mismatches:
                raise RuntimeError(f"compiled source does not match checkout: {mismatches}")
            proof = {"module_count": len(modules), "mismatches": mismatches,
                     "build_manifest": manifest,
                     "scope": "staged application on original machine; no installer or clean-machine acceptance"}
            (output_root / "compiled-source-proof.json").write_text(
                json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: packaged profiles {','.join(args.profiles)}, HTTP logic and backend reasoning/lifecycle", flush=True)


if __name__ == "__main__":
    main()
