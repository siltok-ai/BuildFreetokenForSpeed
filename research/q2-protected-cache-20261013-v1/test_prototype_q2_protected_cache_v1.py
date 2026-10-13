"""Real-weight compact loader, protected cache, prefill and graph lifecycle probe."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import capture_freetoken_reference
import torch
import prototype_q2_protected_cache_v1 as candidate

ROOT = Path(__file__).resolve().parent


def main():
    from freetoken.models.qwen4_exp.gguf_source import GGUFModelSource
    from freetoken.layers.quantization import QuantKind
    from freetoken.layers.quantization.moe.base import BankSpec
    from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
    target = ROOT / 'prototype-q2-protected-cache-v1-component.json'
    if target.exists():
        raise FileExistsError(target)
    inventory = json.loads((ROOT / 'complete-gguf-inventory.json').read_text(encoding='utf-8'))
    gguf = GGUFModelSource([s['path'] for s in inventory['shards']])
    h, intermediate, experts, layers = 2560, 640, 10, 2
    layout = {'gate_up': BankSpec((2 * intermediate, h // 64 * 18), torch.uint8),
              'down': BankSpec((h, intermediate // 64 * 18), torch.uint8)}
    original = {role: [] for role in layout}
    pieces = []
    for layer in range(layers):
        roles = {}
        for name, count, width in [('gate', intermediate, h // 64 * 18), ('up', intermediate, h // 64 * 18), ('down', h, intermediate // 64 * 18)]:
            raw = gguf.tensors[f'blk.{layer}.ffn_{name}_exps.weight'].read_rows(0, experts * count)
            roles[name] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(experts, count, width)
        original['gate_up'].append(torch.cat((roles['gate'], roles['up']), dim=1))
        original['down'].append(roles['down'].clone())
        for start in range(0, experts, 3):
            end = min(experts, start + 3)
            pieces.append((layer, start, end, {role: value[start:end] for role, value in roles.items()}))

    def pack(piece, out):
        out['gate_up'][:, :intermediate].copy_(piece['gate'])
        out['gate_up'][:, intermediate:].copy_(piece['up'])
        out['down'].copy_(piece['down'])
        return {}

    protected = [(0, expert) for expert in range(experts)] + [(1, expert) for expert in (0, 3, 6, 9)]
    candidate.SETTINGS = dict(protected=protected, dynamic=experts)
    method = SimpleNamespace(kind=QuantKind.Q2_0, kernel=SimpleNamespace(name='triton'), layout=lambda: layout, pack=pack)
    config = SimpleNamespace(num_moe_layers=layers, num_experts=experts)
    with patch('freetoken.moe.expert_pieces.iter_expert_pieces', return_value=iter(pieces)):
        banks = candidate.load_compact_banks('real-GGUF-component', config, method=method,
                                             device=torch.device('cuda'), dtype=torch.bfloat16)
    payload = candidate.PAYLOAD
    cache = candidate.ProtectedQ2Cache(layers, experts, payload['slots'], torch.device('cuda'),
                                      prefill_overlap=True, quant_format='q2_0', layout=layout)
    cache.set_bank_sources(banks.sources)
    checks = []
    saved = {role: value[payload['protected_begin']:].clone() for role, value in cache.bank_caches.items()}

    def owners_intact():
        return all(torch.equal(cache.bank_caches[role][payload['protected_begin']:], reference)
                   for role, reference in saved.items())

    def check(name, condition):
        result = dict(name=name, passed=bool(condition and owners_intact()))
        checks.append(result)
        if not result['passed']:
            raise RuntimeError(result)

    check('true_compact_storage_zero_host_layer', all(banks.sources[r][0].size(0) == 0 and banks.sources[r][1].size(0) == 6 for r in layout))
    ids = torch.empty((1, 10), device='cuda', dtype=torch.int32)
    for repeat in range(6):
        cache.begin_prefill()
        for layer in range(layers):
            cache.prefetch_prefill_layer(layer)
            views = cache.wait_prefill_layer(layer)
            for role, view in zip(layout, views):
                check(f'prefill_{repeat}_{layer}_{role}', torch.equal(view.cpu(), original[role][layer]))
            cache.release_prefill_layer(layer)
        cache.reset()
        for layer in range(layers):
            route = list(range(experts)) if repeat % 2 == 0 else [9, 3, 7, 0, 7, 1, 8, 8, 4, 2]
            ids.copy_(torch.tensor([route], device='cuda', dtype=torch.int32))
            cache.ensure_experts(layer, ids)
            cache.copy_missing()
            torch.cuda.synchronize()
            slots = ids[0].long()
            for role in layout:
                check(f'decode_{repeat}_{layer}_{role}', torch.equal(cache.bank_caches[role][slots].cpu(), original[role][layer][route]))
    raw_ids = torch.arange(experts, device='cuda', dtype=torch.int32)[None]
    ids.copy_(raw_ids)
    cache.ensure_experts(1, ids)
    cache.copy_missing()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cache.ensure_experts(1, ids)
        cache.copy_missing()
    for repetition in range(9):
        cache.reset()
        route = [(i * 3 + repetition) % experts for i in range(experts)]
        ids.copy_(torch.tensor([route], device='cuda', dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        for role in layout:
            check(f'graph_reset_{repetition}_{role}', torch.equal(cache.bank_caches[role][ids[0].long()].cpu(), original[role][1][route]))
    x = torch.randn((1, h), device='cuda', dtype=torch.float32) * .1
    weights = torch.full((1, experts), .1, device='cuda', dtype=torch.float32)
    reference_banks = [original[r][1].cuda() for r in layout]
    expected = fused_experts_q2_q8(x, *reference_banks, weights, raw_ids)
    result = fused_experts_q2_q8(x, *cache.bank_views(), weights, ids)
    # The current route is a permutation; identical routing weights make the sum
    # mathematically equal, while summation order is explicitly bounded.
    error = float((result - expected).norm() / expected.norm().clamp_min(1e-10))
    check('real_q2_projection_from_relocated_rows', bool(torch.isfinite(result).all()) and error < 2e-6)
    cache.materialize_layer(1)
    cache.copy_missing()
    for role, view in zip(layout, cache.bank_views(experts)):
        check('nonoverlap_materialize_' + role, torch.equal(view.cpu(), original[role][1]))
    report = dict(passed=all(c['passed'] for c in checks), checks=checks, total_checks=len(checks),
                  projection_relative_l2=error, evidence=candidate.evidence(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='First ten real experts from layers 0 and 1, exact-byte compact load, zero-host layer, prefill double buffers, route admission, graph replay after reset and real Q2 projection; bounded prototype, no full-model acceptance.',
                  full_engine_acceptance=False)
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2), flush=True)


if __name__ == '__main__':
    with torch.cuda.stream(torch.cuda.Stream()):
        main()
