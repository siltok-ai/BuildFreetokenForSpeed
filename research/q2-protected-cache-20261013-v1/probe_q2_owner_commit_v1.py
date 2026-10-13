"""Fresh-process commit accounting for exact pinned banks and a fixed GPU pool.

Synthetic bytes exercise allocation/residency only, not inference or quality.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import time

import capture_freetoken_reference
import torch


class PerformanceInfo(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong)] + [(n, ctypes.c_size_t) for n in (
        'CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
        'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')]
    _fields_ += [(n, ctypes.c_ulong) for n in ('HandleCount', 'ProcessCount', 'ThreadCount')]


def memory():
    p = PerformanceInfo()
    p.cb = ctypes.sizeof(p)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(p), p.cb):
        raise ctypes.WinError()
    free, total = torch.cuda.mem_get_info()
    return dict(commit_used=p.CommitTotal * p.PageSize, commit_limit=p.CommitLimit * p.PageSize,
                physical_available=p.PhysicalAvailable * p.PageSize,
                gpu_allocated=torch.cuda.memory_allocated(), gpu_reserved=torch.cuda.memory_reserved(),
                gpu_free=free, gpu_total=total)


def main():
    from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('full_host', 'complement'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    experts, protected, dynamic, prefill = 2048, 1536, 256, 1024
    sizes = (921600, 460800)
    warm = torch.ones((1,), device='cuda')
    torch.cuda.synchronize()
    initial = memory()
    slots = protected + dynamic + prefill
    gpu = [torch.empty((slots, size), device='cuda', dtype=torch.uint8) for size in sizes]
    for tensor in gpu:
        tensor.fill_(0x5A)
    torch.cuda.synchronize()
    after_gpu = memory()
    host_rows = experts if args.mode == 'full_host' else experts - protected
    host = [alloc_pinned_tensor(host_rows, size, dtype=torch.uint8) for size in sizes]
    for tensor in host:
        tensor.fill_(0xA5)
    aliases = [dict(cpu=t.data_ptr(), device=device_ptr(t)) for t in host]
    # A small actual mapped read proves the host banks are GPU-visible.
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
    dst_ptrs = torch.tensor([t.data_ptr() for t in gpu], device='cuda', dtype=torch.int64)
    src_ptrs = torch.tensor([device_ptr(t) for t in host], device='cuda', dtype=torch.int64)
    features = torch.tensor(sizes, device='cuda', dtype=torch.int64)
    indices = torch.tensor([0], device='cuda', dtype=torch.int32)
    count = torch.ones((1,), device='cuda', dtype=torch.int64)
    fast_index_copy_multi_jit(dst_ptrs, src_ptrs, features, indices, indices, count)
    torch.cuda.synchronize()
    correct = all(bool((t[0] == 0xA5).all()) for t in gpu)
    samples = []
    for _ in range(10):
        samples.append(memory())
        time.sleep(.1)
    report = dict(passed=correct, mode=args.mode, experts=experts, protected=protected,
                  dynamic=dynamic, prefill=prefill, gpu_slots=slots,
                  host_bytes=host_rows * sum(sizes), gpu_bank_bytes=slots * sum(sizes),
                  aliases=aliases, initial=initial, after_gpu=after_gpu, samples=samples,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Fresh-process synthetic exact-size Q2 bank allocations; fixed GPU slots; commit accounting and mapped-host copy only, no model performance/quality acceptance.',
                  full_engine_acceptance=False)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2), flush=True)
    if not correct:
        raise RuntimeError('Mapped allocation copy failed')


if __name__ == '__main__':
    with torch.cuda.stream(torch.cuda.Stream()):
        main()
