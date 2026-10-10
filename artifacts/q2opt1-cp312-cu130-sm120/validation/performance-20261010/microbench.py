"""真实 GPU 上对比分组 Q2 算法、原向量算法和独立 PyTorch 数值参考。"""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'runtime-candidate'))
import torch
from freetoken.moe.fused_q2_0 import fused_experts_q2_0, grouped_experts_q2_0


def packed_weights(experts, n, k):
    """生成原生块及独立解码参考，包含负值和非零 2-bit code。"""
    scale = (torch.rand((experts, n, k // 64), device='cuda') * .025).half()
    codes = torch.randint(0, 4, (experts, n, k), device='cuda', dtype=torch.uint8)
    packed = torch.empty((experts, n, k // 64, 18), dtype=torch.uint8, device='cuda')
    packed[..., :2] = scale.contiguous().view(torch.uint8).reshape(experts, n, k // 64, 2)
    fields = codes.reshape(experts, n, k // 64, 16, 4)
    packed[..., 2:] = fields[..., 0] | (fields[..., 1] << 2) | (fields[..., 2] << 4) | (fields[..., 3] << 6)
    decoded = ((codes.float() - 1) * scale.float().repeat_interleave(64, dim=-1)).bfloat16()
    return packed.reshape(experts, n, k // 64 * 18), decoded


def reference(x, gu, down, weights, ids):
    """按路由分别执行 FP32 矩阵乘法，保留既有 BF16 舍入边界。"""
    rows = []
    intermediate = gu.shape[1] // 2
    for token in range(x.shape[0]):
        routed = []
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            values = (gu[expert].float() @ x[token].float()).bfloat16().float()
            gate, up = values[:intermediate], values[intermediate:]
            active = (torch.nn.functional.silu(gate) * up).bfloat16().float()
            routed.append((down[expert].float() @ active).bfloat16().float() * weights[token, route])
        rows.append(torch.stack(routed).sum(0).bfloat16())
    return torch.stack(rows)


def timed(fn):
    """预热后用 CUDA events 计量，避免把首编计入算法吞吐。"""
    for _ in range(3):
        fn()
    times = []
    for _ in range(5):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def main():
    """先验证小形状，再测 Flash Next 的真实 H/I/E/top-k 几何。"""
    torch.manual_seed(142)
    torch.backends.cuda.matmul.allow_tf32 = False
    gu, gu_ref = packed_weights(8, 128, 128)
    down, down_ref = packed_weights(8, 128, 64)
    x = torch.randn((19, 128), device='cuda', dtype=torch.bfloat16)
    ids = torch.randint(0, 8, (19, 3), device='cuda', dtype=torch.int32)
    weights = torch.softmax(torch.randn((19, 3), device='cuda'), dim=-1)
    expected = reference(x, gu_ref, down_ref, weights, ids)
    grouped = grouped_experts_q2_0(x, gu, down, weights, ids)
    torch.testing.assert_close(grouped, expected, atol=.01, rtol=.02)
    report = {'small_reference_max_abs': (grouped.float() - expected.float()).abs().max().item(), 'runs': []}
    del gu, gu_ref, down, down_ref
    gu, gu_ref = packed_weights(512, 1280, 2560)
    del gu_ref
    down, down_ref = packed_weights(512, 2560, 640)
    del down_ref
    for tokens in (70, 128, 256, 1024, 4096):
        x = torch.randn((tokens, 2560), device='cuda', dtype=torch.bfloat16)
        ids = torch.topk(torch.rand((tokens, 512), device='cuda'), 10, dim=-1).indices.int()
        weights = torch.softmax(torch.randn((tokens, 10), device='cuda'), dim=-1)
        os.environ['FREETOKEN_Q2_GROUPED_PREFILL'] = '0'
        vector = fused_experts_q2_0(x, gu, down, weights, ids)
        grouped = grouped_experts_q2_0(x, gu, down, weights, ids)
        relative_l2 = ((grouped.float() - vector.float()).norm() / vector.float().norm()).item()
        if relative_l2 > .015:
            raise AssertionError(f'Q2 grouped relative L2 {relative_l2}')
        vector_ms = timed(lambda: fused_experts_q2_0(x, gu, down, weights, ids))
        grouped_ms = timed(lambda: grouped_experts_q2_0(x, gu, down, weights, ids))
        item = dict(tokens=tokens, vector_ms=vector_ms, grouped_ms=grouped_ms,
                    speedup=vector_ms / grouped_ms, relative_l2=relative_l2)
        report['runs'].append(item)
        (ROOT / 'microbench.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(item), flush=True)


if __name__ == '__main__':
    main()
