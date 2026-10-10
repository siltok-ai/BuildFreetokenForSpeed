"""Record machine commit throughout an isolated validation command without importing Torch."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import subprocess
import time


class PerformanceInfo(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
        "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize",
    )] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


def snapshot() -> dict[str, int]:
    """使用 Windows 原生接口采样当前整机 commit 和物理可用内存。"""
    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not ctypes.WinDLL("psapi").GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise ctypes.WinError()
    return {"commit_used": info.CommitTotal * info.PageSize,
            "commit_limit": info.CommitLimit * info.PageSize,
            "physical_available": info.PhysicalAvailable * info.PageSize}


def main() -> int:
    """执行真实验证子进程并保留成功或失败期间的容量与命令记录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a validation command is required after --")
    baseline = snapshot()
    observations = [baseline]
    started = time.perf_counter()
    child = subprocess.Popen(command)
    while child.poll() is None:
        observations.append(snapshot())
        time.sleep(.1)
    observations.append(snapshot())
    result = {"command": command, "returncode": child.returncode,
              "sampling_interval_seconds": .1, "samples": len(observations), "baseline": baseline,
              "peak_commit_used": max(item["commit_used"] for item in observations),
              "min_commit_available": min(item["commit_limit"] - item["commit_used"] for item in observations),
              "min_physical_available": min(item["physical_available"] for item in observations),
              "wall_seconds": time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "capacity_monitor", **result}), flush=True)
    return child.returncode


if __name__ == "__main__":
    raise SystemExit(main())
