"""Measure full native Q2_0 host-bank residency/pinning, without claiming model accuracy."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time

import psutil
import torch


def snapshot():
    class PerformanceInfo(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable", "SystemCache",
            "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [(name, wintypes.DWORD) for name in (
                "HandleCount", "ProcessCount", "ThreadCount")]
    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not ctypes.WinDLL("psapi").GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise ctypes.WinError()
    process = psutil.Process().memory_info()
    return {"rss": process.rss, "private": process.private, "peak_rss": process.peak_wset,
            "commit_used": info.CommitTotal * info.PageSize, "commit_limit": info.CommitLimit * info.PageSize,
            "physical_available": info.PhysicalAvailable * info.PageSize,
            "vram_reserved": torch.cuda.memory_reserved()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from freetoken.layers.quantization.moe.base import MoEConfig
    from freetoken.layers.quantization.moe.q2_0 import Q20MoEMethod
    from freetoken.moe.expert_banks import build_expert_banks

    torch.cuda.init()
    baseline = snapshot()
    print(json.dumps({"stage": "baseline", **baseline}), flush=True)
    started = time.perf_counter()
    banks = build_expert_banks(Q20MoEMethod(MoEConfig(num_experts=512, hidden=2560, intermediate=640,
                               top_k=10, strategy="offload")), 48, None,
                               device=torch.device("cuda"), dummy=True)
    tensors = [tensor for role in banks.sources.values() for tensor in role]
    bank_bytes = sum(tensor.nbytes for tensor in tensors)
    after_banks = snapshot()
    if bank_bytes != 33_973_862_400 or not all(tensor.is_pinned() for tensor in tensors):
        raise AssertionError("full Q2_0 banks were not resident and pinned as required")
    # 实际 H2D 复制覆盖多个 layer/expert 索引；容量探测的人工字节不用于模型正确性声明。
    for layer, expert in ((0, 0), (17, 311), (47, 511)):
        for role in ("gate_up", "down"):
            copied = banks.sources[role][layer][expert].to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            torch.testing.assert_close(copied.cpu(), banks.sources[role][layer][expert], rtol=0, atol=0)
    result = {"status": "passed_capacity_only", "bank_bytes": bank_bytes, "bank_GiB": bank_bytes / 2**30,
              "layers": 48, "experts_per_layer": 512, "all_pinned": True, "baseline": baseline,
              "after_banks": after_banks, "rss_delta": after_banks["rss"] - baseline["rss"],
              "private_delta": after_banks["private"] - baseline["private"],
              "elapsed_s": time.perf_counter() - started, "h2d_samples": "passed"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
