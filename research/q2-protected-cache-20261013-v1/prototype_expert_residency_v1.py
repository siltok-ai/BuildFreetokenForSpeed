"""Mixed mapped-host/GPU Q2 owner gather, independent of engine cache wiring.

This probe copies real GGUF bytes; it does not execute a different expert format,
change installed FreeToken, or claim an end-to-end throughput improvement.
"""
import hashlib
import json
import statistics
from pathlib import Path

import capture_freetoken_reference
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent


@triton.jit
def _owner_gather(DST, SOURCES, FEATS, DI, SI, COUNT,
                  TABLE_SIZE: tl.constexpr, BLOCKS: tl.constexpr,
                  UNITS: tl.constexpr):
    bank = tl.program_id(0)
    block = tl.program_id(1)
    count = tl.load(COUNT)
    units_per_row = tl.load(FEATS + bank) // 16
    destination = tl.load(DST + bank)
    total = count * units_per_row
    lanes = tl.arange(0, UNITS)
    words = tl.arange(0, 4)
    for start in range(0, tl.cdiv(total, BLOCKS * UNITS)):
        unit = start * (BLOCKS * UNITS) + block * UNITS + lanes
        valid = unit < total
        route = unit // units_per_row
        src_id = tl.load(SI + route, valid, 0).to(tl.int64)
        dst_id = tl.load(DI + route, valid, 0).to(tl.int64)
        source = tl.load(SOURCES + bank * TABLE_SIZE + src_id, valid, 0)
        col = (unit % units_per_row) * 16
        src_addr = (source + col).to(tl.pointer_type(tl.uint32))
        dst_addr = (destination + dst_id * units_per_row * 16 + col).to(tl.pointer_type(tl.uint32))
        value = tl.load(src_addr[:, None] + words[None, :], valid[:, None], 0)
        tl.store(dst_addr[:, None] + words[None, :], value, valid[:, None])


def gather(descriptor, dst_ids, src_ids, count, blocks=8):
    dst, sources, feats = descriptor
    _owner_gather[(len(dst), blocks)](
        dst, sources, feats, dst_ids, src_ids, count,
        sources.shape[1], blocks, 1024, num_warps=8)


def timing(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(16):
            fn()
    samples = []
    for _ in range(7):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(4):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / 64)
    return dict(median_us=statistics.median(samples), samples_us=samples,
                operations_per_sample=64, scope='GPU events, captured copy only; no model or CPU launch timing')


def main():
    from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
    from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
    from freetoken.models.qwen4_exp.gguf_source import GGUFModelSource

    target = ROOT / 'prototype-expert-residency-v1.json'
    if target.exists():
        raise FileExistsError(target)
    inventory = json.loads((ROOT / 'complete-gguf-inventory.json').read_text(encoding='utf-8'))
    source = GGUFModelSource([s['path'] for s in inventory['shards']])
    h, intermediate, rows = 2560, 640, 10
    sizes = (2 * intermediate * (h // 64 * 18), h * (intermediate // 64 * 18))
    logical_ids = [7, 4095, 1027, 1901, 29, 3028, 811, 2458, 90, 3800]
    # A full host allocation is retained only for the independent baseline probe.
    baseline = [alloc_pinned_tensor(rows, size, dtype=torch.uint8) for size in sizes]
    source_hashes = []
    for expert in range(rows):
        gate = source.tensors['blk.0.ffn_gate_exps.weight'].read_rows(expert * intermediate, intermediate)
        up = source.tensors['blk.0.ffn_up_exps.weight'].read_rows(expert * intermediate, intermediate)
        down = source.tensors['blk.0.ffn_down_exps.weight'].read_rows(expert * h, h)
        for bank, raw in enumerate((gate + up, down)):
            baseline[bank][expert].copy_(torch.frombuffer(bytearray(raw), dtype=torch.uint8))
            source_hashes.append(dict(expert=expert, bank=bank, sha256=hashlib.sha256(raw).hexdigest()))
    destination = [torch.empty((16, size), device='cuda', dtype=torch.uint8) for size in sizes]
    dst_ptrs = torch.tensor([t.data_ptr() for t in destination], device='cuda', dtype=torch.int64)
    feats = torch.tensor(sizes, device='cuda', dtype=torch.int64)
    base_ptrs = torch.tensor([device_ptr(t) for t in baseline], device='cuda', dtype=torch.int64)
    checks, allocations, timings = [], [], {}
    ids = torch.zeros((12,), device='cuda', dtype=torch.int32)
    dst_ids = torch.arange(12, device='cuda', dtype=torch.int32)
    count = torch.zeros((1,), device='cuda', dtype=torch.int64)
    table = torch.zeros((2, 4096), device='cuda', dtype=torch.int64)
    descriptor = dst_ptrs, table, feats

    def owners(protected):
        protected = tuple(protected)
        host_experts = [e for e in range(rows) if e not in protected]
        host = [alloc_pinned_tensor(len(host_experts), size, dtype=torch.uint8)
                if host_experts else None for size in sizes]
        gpu = [torch.empty((len(protected), size), device='cuda', dtype=torch.uint8)
               if protected else None for size in sizes]
        pointer_values = torch.zeros((2, 4096), dtype=torch.int64)
        for bank in range(2):
            for compact, expert in enumerate(host_experts):
                host[bank][compact].copy_(baseline[bank][expert])
                pointer_values[bank, logical_ids[expert]] = device_ptr(host[bank]) + compact * sizes[bank]
            for slot, expert in enumerate(protected):
                gpu[bank][slot].copy_(baseline[bank][expert])
                pointer_values[bank, logical_ids[expert]] = gpu[bank].data_ptr() + slot * sizes[bank]
        table.copy_(pointer_values)
        torch.cuda.synchronize()
        allocations.append(dict(protected_experts=list(protected), host_experts=host_experts,
                                compact_host_bytes=len(host_experts) * sum(sizes),
                                protected_gpu_bytes=len(protected) * sum(sizes),
                                full_host_bytes=rows * sum(sizes),
                                actual_host_aliases=[dict(cpu=t.data_ptr(), device=device_ptr(t))
                                                     for t in host if t is not None]))
        # These references must survive all captured pointer dereferences.
        return host, gpu, pointer_values

    def set_route(experts):
        ids.fill_(-1)
        ids[:len(experts)].copy_(torch.tensor([logical_ids[e] for e in experts], device='cuda', dtype=torch.int32))
        count.fill_(len(experts))

    def check(label, experts, protected):
        torch.cuda.synchronize()
        n = len(experts)
        correct = all(torch.equal(t[:n].cpu(), baseline[b][experts]) for b, t in enumerate(destination))
        untouched = all(bool((t[n:] == 0xA5).all()) for t in destination)
        retained = all(torch.equal(t.cpu(), baseline[b][list(protected)])
                       for b, t in enumerate(owner_refs[1]) if t is not None)
        row = dict(name=label, routes=experts, exact_bytes=correct,
                   untouched_destination_tail=untouched, protected_owners_unchanged=retained)
        row['passed'] = correct and untouched and retained
        checks.append(row)
        if not row['passed']:
            raise RuntimeError(row)

    for label, protected in [('mixed', (0, 3, 6, 9)), ('host_only', ()), ('gpu_only_zero_host', tuple(range(rows)))]:
        owner_refs = owners(protected)
        for name, experts in [('ten', list(range(rows))), ('repeated', [9, 9, 3, 0, 7]), ('zero', [])]:
            for t in destination:
                t.fill_(0xA5)
            set_route(experts)
            gather(descriptor, dst_ids, ids, count)
            check(label + '_' + name, experts, protected)
        set_route(list(range(rows)))
        for blocks in (8, 16, 32):
            timings[label + '_blocks' + str(blocks)] = timing(lambda: gather(descriptor, dst_ids, ids, count, blocks))
        del owner_refs
        torch.cuda.synchronize()

    owner_refs = owners((0, 3, 6, 9))
    set_route(list(range(rows)))
    gather(descriptor, dst_ids, ids, count)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gather(descriptor, dst_ids, ids, count)
    for experts in ([9, 0, 9, 7], [], [1, 2, 4, 5, 7, 8]):
        for t in destination:
            t.fill_(0xA5)
        set_route(experts)
        graph.replay()
        check('graph_changed_route_count', experts, (0, 3, 6, 9))
    old_refs = owner_refs
    owner_refs = owners((1, 2, 4, 5, 7, 8))
    for t in destination:
        t.fill_(0xA5)
    set_route([0, 1, 2, 3, 9, 8, 8])
    graph.replay()
    check('same_graph_changed_pointer_contents', [0, 1, 2, 3, 9, 8, 8], (1, 2, 4, 5, 7, 8))
    torch.cuda.synchronize()
    del old_refs

    base_ids = torch.arange(rows, device='cuda', dtype=torch.int32)
    base_dst_ids = torch.arange(rows, device='cuda', dtype=torch.int32)
    base_count = torch.full((1,), rows, device='cuda', dtype=torch.int64)
    timings['native_full_host_baseline'] = timing(lambda: fast_index_copy_multi_jit(
        dst_ptrs, base_ptrs, feats, base_dst_ids, base_ids, base_count))
    torch.cuda.synchronize()
    report = dict(passed=all(c['passed'] for c in checks), checks=checks,
                  allocations=allocations, timings=timings, expert_row_bytes=sum(sizes),
                  logical_source_ids=logical_ids, source_hashes=source_hashes,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  gguf_revision=inventory['revision'],
                  prototype_only=True, engine_modified=False, full_engine_acceptance=False,
                  scope='Ten real layer-0 GGUF experts, two original Q2 banks; sparse 4095 logical IDs, compact mapped host / GPU owners, graph pointer updates. No cache/LRU/prefill lifecycle integration.',
                  gpu_max_allocated=torch.cuda.max_memory_allocated())
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in ('source_hashes', 'allocations')}, indent=2), flush=True)


if __name__ == '__main__':
    with torch.cuda.stream(torch.cuda.Stream()):
        main()
