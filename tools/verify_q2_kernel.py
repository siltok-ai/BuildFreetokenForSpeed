"""Compare the GPU Q2_0 MoE kernel with independently decoded real expert bytes."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import requests
import torch
import torch.nn.functional as F

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
FILE = "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"


def sample(role, directory):
    offsets = {"down": 757141792, "gate": 994197792, "up": 1233624352}
    path = directory / f"q2-layer0-{role}-sample.bin"
    length = 3 * 460800
    if path.exists() and path.stat().st_size == length:
        return path
    start = offsets[role]
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}?sample={role}"
    response = requests.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=(15, 60))
    response.raise_for_status()
    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{start + length - 1}/37623740192":
        raise ValueError("server did not honor the exact sample range")
    if len(response.content) != length:
        raise ValueError("short Q2_0 sample")
    path.write_bytes(response.content)
    return path


def decode(raw, rows, columns):
    # 独立 CPU 参考：逐块恢复 FP16 scale 和四个相邻 2-bit code。
    blocks = np.frombuffer(raw, dtype=np.uint8).reshape(3, rows, columns // 64, 18)
    scale = blocks[..., :2].copy().view("<f2").astype(np.float32)[..., 0]
    packed = blocks[..., 2:]
    codes = np.stack([(packed >> shift) & 3 for shift in (0, 2, 4, 6)], axis=-1)
    values = (codes.reshape(3, rows, columns // 64, 64).astype(np.float32) - 1) * scale[..., None]
    return torch.from_numpy(values.reshape(3, rows, columns).copy()).to(device="cuda", dtype=torch.bfloat16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from freetoken.moe.fused_q2_0 import fused_experts_q2_0

    with ThreadPoolExecutor(3) as pool:
        paths = dict(zip(("gate", "up", "down"), pool.map(lambda role: sample(role, args.output), ("gate", "up", "down"))))
    data = {role: path.read_bytes() for role, path in paths.items()}
    raw = {role: torch.from_numpy(np.frombuffer(value, np.uint8).copy()).reshape(3, 2560 if role == "down" else 640, 180 if role == "down" else 720).cuda() for role, value in data.items()}
    gate_up = torch.cat((raw["gate"], raw["up"]), 1).contiguous()
    dense = {role: decode(value, 2560 if role == "down" else 640, 640 if role == "down" else 2560) for role, value in data.items()}
    records = []
    torch.manual_seed(3108)
    for tokens in (1, 3, 16):
        x = torch.randn(tokens, 2560, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, 3, (tokens, 3), device="cuda", dtype=torch.int32)
        ids[0] = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
        weights = torch.softmax(torch.randn(tokens, 3, device="cuda"), -1).contiguous()
        expected = torch.zeros(tokens, 2560, dtype=torch.float32, device="cuda")
        for token in range(tokens):
            for route in range(3):
                expert = ids[token, route].item()
                g = F.linear(x[token], dense["gate"][expert]).float()
                u = F.linear(x[token], dense["up"][expert]).float()
                activated = (F.silu(g) * u).to(torch.bfloat16)
                expected[token] += F.linear(activated, dense["down"][expert]).float() * weights[token, route]
        expected = expected.to(torch.bfloat16)
        actual = fused_experts_q2_0(x, gate_up, raw["down"], weights, ids)
        # 模拟 decode cache 的 slot 重排，验证不能把 expert id 与 cache slot 混用。
        permutation = torch.tensor([2, 0, 1], device="cuda")
        inverse = torch.argsort(permutation)
        remapped = fused_experts_q2_0(x, gate_up[permutation].contiguous(), raw["down"][permutation].contiguous(), weights, inverse[ids.long()].int().contiguous())
        torch.testing.assert_close(remapped, actual, rtol=0, atol=0)
        error = actual.float() - expected.float()
        nrmse = (error.square().mean() / expected.float().square().mean()).sqrt().item()
        if not torch.isfinite(actual).all() or nrmse > 0.008:
            raise AssertionError(f"Q2_0 reference mismatch, tokens={tokens}, NRMSE={nrmse}")
        for _ in range(3):
            fused_experts_q2_0(x, gate_up, raw["down"], weights, ids)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            fused_experts_q2_0(x, gate_up, raw["down"], weights, ids)
        torch.cuda.synchronize()
        record = {"tokens": tokens, "nrmse": nrmse, "max_abs_error": error.abs().max().item(),
                  "ms": (time.perf_counter() - started) * 1000 / 20}
        print(json.dumps(record), flush=True)
        records.append(record)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fused_experts_q2_0(x, gate_up, raw["down"], weights, ids)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
    result = {"status": "passed", "repo": REPO, "revision": REVISION, "file": FILE,
              "samples": {role: hashlib.sha256(value).hexdigest() for role, value in data.items()},
              "precision": "Q2_0 weights rounded to BF16, BF16 activations/intermediates, FP32 accumulation/routing sum",
              "records": records, "cache_remapping": "passed", "cuda_graph": "passed"}
    (args.output / "q2-kernel-validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
