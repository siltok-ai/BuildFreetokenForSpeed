"""Measure the complete engine with real HF trunk/SSD PLE and explicitly artificial Q2 banks."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import threading
import time

from probe_q2_bank_memory import snapshot
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kernel-cache", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["FREETOKEN_KERNEL_CACHE_DIR"] = args.kernel_cache
    os.environ["FREETOKEN_DISABLE_JIT"] = "1"
    from silitok_speed.qwen36.freetoken_kernel_cache import activate_runtime
    activate_runtime()
    backend = importlib.import_module("freetoken.kernel.backend")
    backend.is_flashinfer_installed = lambda: False
    engine_module = importlib.import_module("freetoken.engine.engine")
    original = engine_module.load_expert_banks
    def artificial_banks(*arguments, **keywords):
        # 只在这个容量探测进程替换专家 bank 填充，不改引擎源码或真实权重目录。
        keywords["dummy"] = True
        return original(*arguments, **keywords)
    engine_module.load_expert_banks = artificial_banks
    from freetoken.llm.llm import LLM
    from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
    baseline = snapshot()
    observed = []
    stopped = threading.Event()
    def observe():
        while not stopped.wait(.5):
            observed.append(snapshot())
    monitor = threading.Thread(target=observe, daemon=True)
    monitor.start()
    started = time.perf_counter()
    try:
        llm = LLM(args.model, dtype=torch.bfloat16, max_running_req=1, attention_backend="qsa_sparse",
            moe_strategy="offload", ple_backend="disk", expert_load="serial", moe_cache_size=1024,
            moe_cache_auto=False, num_token_override=8192, kv_reserve_tokens=8192, memory_ratio=.80,
            max_seq_len_override=262144, max_extend_tokens=1024, cuda_graph_max_bs=1,
            mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)))
        cache = llm.engine.moe_offload_cache
        tensors = [tensor for role in cache.bank_sources.values() for tensor in role]
        assert sum(tensor.nbytes for tensor in tensors) == 33_973_862_400
        assert all(tensor.is_pinned() for tensor in tensors)
        after = snapshot()
        result = {"status": "passed_capacity_only_artificial_experts", "baseline": baseline, "after_engine": after,
                  "expert_bytes": sum(tensor.nbytes for tensor in tensors), "all_pinned": True,
                  "peak_rss": max([after["rss"], *[value["rss"] for value in observed]]),
                  "peak_commit": max([after["commit_used"], *[value["commit_used"] for value in observed]]),
                  "elapsed_s": time.perf_counter() - started, "snapshot_count": len(observed)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result), flush=True)
        llm.shutdown()
    finally:
        stopped.set()
        monitor.join()


if __name__ == "__main__":
    main()
