"""Device-only Q2 protected-owner/LRU partition probe, not engine integration."""
import hashlib
import json
from pathlib import Path
import random

import capture_freetoken_reference
import torch
import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parent


@triton.jit
def _ensure(IDS, MAP, DYNAMIC_IDS, USAGE, STEP, COPY_SRC, COPY_DST, COUNT,
            STATS, BASE: tl.constexpr, DYNAMIC_BEGIN: tl.constexpr,
            PROTECTED_BEGIN: tl.constexpr, DYNAMIC_SIZE: tl.constexpr,
            K: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr):
    step = tl.load(STEP) + 1
    tl.store(STEP, step)
    k = tl.arange(0, BK)
    valid = k < K
    ids = tl.load(IDS + k, valid, 0).to(tl.int64)
    global_ids = ids + BASE
    slots = tl.load(MAP + global_ids, valid, -1)
    hits = valid & (slots >= 0)
    dynamic_hits = hits & (slots >= DYNAMIC_BEGIN) & (slots < PROTECTED_BEGIN)
    tl.store(USAGE + slots - DYNAMIC_BEGIN, step, dynamic_hits)
    same = (global_ids[:, None] == global_ids[None, :]) & (k[:, None] > k[None, :]) & valid[:, None] & valid[None, :]
    first = valid & (tl.sum(same.to(tl.int32), 1) == 0)
    first_miss = first & (slots < 0)
    miss = valid & (slots < 0)
    rank = tl.sum(((global_ids[None, :] < global_ids[:, None]) & first_miss[None, :]).to(tl.int32), 1)
    n = tl.sum(first_miss.to(tl.int32))
    tl.store(COUNT, n.to(tl.int64))
    out = slots
    if n > 0:
        tl.debug_barrier()
        c = tl.arange(0, BC)
        u = tl.load(USAGE + c, c < DYNAMIC_SIZE, 0x3FFFFFFFFFFFFFFF)
        u = tl.where((c >= DYNAMIC_SIZE) | (u == step), 0x3FFFFFFFFFFFFFFF, u)
        for i in range(n):
            victim = tl.argmin(u, 0).to(tl.int32)
            old = tl.load(DYNAMIC_IDS + victim)
            if old >= 0:
                tl.store(MAP + old, -1)
            incoming = tl.sum(tl.where(first_miss & (rank == i), global_ids, 0), 0)
            physical = victim + DYNAMIC_BEGIN
            tl.store(DYNAMIC_IDS + victim, incoming.to(tl.int32))
            tl.store(MAP + incoming, physical)
            tl.store(USAGE + victim, step)
            tl.store(COPY_SRC + i, (incoming - BASE).to(tl.int32))
            tl.store(COPY_DST + i, physical)
            out = tl.where(miss & (rank == i), physical, out)
            u = tl.where(c == victim, 0x3FFFFFFFFFFFFFFF, u)
    tl.store(IDS + k, out, valid)
    s = tl.arange(0, 4)
    values = tl.where(s == 0, tl.sum(first.to(tl.int32), 0), tl.where(s == 1, n, 1))
    tl.atomic_add(STATS + s, values.to(tl.int64), s < 3)


def main():
    target = ROOT / 'prototype-protected-admission-v1.json'
    if target.exists():
        raise FileExistsError(target)
    layers, experts, dynamic, routes = 3, 32, 8, 8
    begin = 2 * experts
    protected_begin = begin + dynamic
    protected = [0, 3, 17, 31, 32 + 9, 32 + 22, 64, 64 + 29, 64 + 31]
    original = [-1] * (layers * experts)
    for ordinal, expert in enumerate(protected):
        original[expert] = protected_begin + ordinal
    slot_map = torch.tensor(original, device='cuda', dtype=torch.int32)
    dyn_ids = torch.full((dynamic,), -1, device='cuda', dtype=torch.int32)
    usage = torch.zeros((dynamic,), device='cuda', dtype=torch.int64)
    step = torch.zeros((1,), device='cuda', dtype=torch.int64)
    src = torch.full((routes,), -1, device='cuda', dtype=torch.int32)
    dst = torch.full((routes,), -1, device='cuda', dtype=torch.int32)
    count = torch.zeros((1,), device='cuda', dtype=torch.int64)
    stats = torch.zeros((3,), device='cuda', dtype=torch.int64)
    ids = torch.zeros((routes,), device='cuda', dtype=torch.int32)
    reference = dict(mapping=original.copy(), dynamic_ids=[-1] * dynamic,
                     usage=[0] * dynamic, step=0, stats=[0, 0, 0])

    def reset():
        # Restoring the immutable map preserves all permanent GPU owners.
        slot_map.copy_(torch.tensor(original, device='cuda', dtype=torch.int32))
        dyn_ids.fill_(-1)
        usage.zero_()
        step.zero_()
        stats.zero_()
        reference.update(mapping=original.copy(), dynamic_ids=[-1] * dynamic,
                         usage=[0] * dynamic, step=0, stats=[0, 0, 0])

    def oracle(layer, query):
        r = reference
        r['step'] += 1
        global_ids = [layer * experts + e for e in query]
        active = set(global_ids)
        missing = sorted(e for e in active if r['mapping'][e] < 0)
        for e in active:
            physical = r['mapping'][e]
            if begin <= physical < protected_begin:
                r['usage'][physical - begin] = r['step']
        incoming, destinations = [], []
        for e in missing:
            candidates = [i for i in range(dynamic) if r['usage'][i] != r['step']]
            if not candidates:
                raise RuntimeError('Probe exceeds dynamic admission capacity')
            victim = min(candidates, key=lambda i: (r['usage'][i], i))
            old = r['dynamic_ids'][victim]
            if old >= 0:
                r['mapping'][old] = -1
            r['dynamic_ids'][victim] = e
            r['mapping'][e] = begin + victim
            r['usage'][victim] = r['step']
            incoming.append(e - layer * experts)
            destinations.append(begin + victim)
        r['stats'] = [r['stats'][0] + len(active), r['stats'][1] + len(missing), r['stats'][2] + 1]
        return [r['mapping'][e] for e in global_ids], incoming, destinations

    def launch(layer):
        _ensure[(1,)](ids, slot_map, dyn_ids, usage, step, src, dst, count, stats,
                      layer * experts, begin, protected_begin, dynamic, routes,
                      triton.next_power_of_2(routes), triton.next_power_of_2(dynamic), num_warps=4)

    randomizer = random.Random(20261013)
    checks = []
    graphs = {}
    for layer in range(layers):
        ids.fill_(0)
        launch(layer)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(layer)
        graphs[layer] = graph
    reset()
    for index in range(330):
        layer = index % layers
        if index in (95, 191, 287):
            reset()
        if index % 11 == 0:
            local = [e % experts for e in protected if e // experts == layer]
            query = [local[i % len(local)] for i in range(routes)]
        elif index % 13 == 0:
            query = list(range(routes))
        else:
            query = [randomizer.randrange(experts) for _ in range(routes)]
        expected, pending_src, pending_dst = oracle(layer, query)
        ids.copy_(torch.tensor(query, device='cuda', dtype=torch.int32))
        if index % 2:
            graphs[layer].replay()
        else:
            launch(layer)
        torch.cuda.synchronize()
        actual_count = int(count.item())
        result = dict(index=index, layer=layer, graph=bool(index % 2),
                      query=query, slots=ids.tolist(), expected=expected,
                      pending_src=src[:actual_count].tolist(), pending_dst=dst[:actual_count].tolist())
        result['passed'] = (result['slots'] == expected and actual_count == len(pending_src)
                            and result['pending_src'] == pending_src and result['pending_dst'] == pending_dst
                            and slot_map.tolist() == reference['mapping']
                            and dyn_ids.tolist() == reference['dynamic_ids']
                            and usage.tolist() == reference['usage'] and stats.tolist() == reference['stats']
                            and all(slot_map[e].item() == original[e] for e in protected)
                            and all(begin <= s < protected_begin for s in result['pending_dst']))
        checks.append(result)
        if not result['passed']:
            raise RuntimeError(result)
    report = dict(passed=all(c['passed'] for c in checks), checks=checks,
                  total_checks=len(checks), graph_replays=sum(c['graph'] for c in checks),
                  protected_global_ids=protected, prefill_slots=[0, begin],
                  dynamic_slots=[begin, protected_begin], protected_slots=[protected_begin, protected_begin + len(protected)],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Independent Python LRU oracle versus device admission, 330 multi-layer calls and three resets, fixed GPU owners, duplicate queries, full-hit/miss/churn; no payload/prefill engine integration.',
                  prototype_only=True, full_engine_acceptance=False)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2), flush=True)


if __name__ == '__main__':
    with torch.cuda.stream(torch.cuda.Stream()):
        main()
