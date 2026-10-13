"""Port frozen validated primitives into an explicit optional FreeToken feature."""
import ast
import hashlib
import json
from pathlib import Path
import re
import struct

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'candidate-source-dev6-working/build-source'


def write(relative, text):
    path = SOURCE / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8', newline='\n')


def definition(filename, name):
    source = (ROOT / filename).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if getattr(n, 'name', None) == name)
    start = min([node.lineno, *(n.lineno for n in getattr(node, 'decorator_list', []))])
    return '\n'.join(source.splitlines()[start - 1:node.end_lineno]) + '\n'


def main():
    target = SOURCE / 'python/freetoken/moe/q2_residency.py'
    if target.exists():
        raise FileExistsError(target)
    raw = (ROOT / 'strata-official-expert-profile.bin').read_bytes()
    origin = json.loads((ROOT / 'strata-official-expert-profile-source.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == origin['sha256']
    version, layers, experts, slots, ranked = struct.unpack_from('<5I', raw, 4)
    assert raw[:4] == b'STRP' and (version, layers, experts, slots, ranked) == (1, 48, 512, 24576, 24576)
    pairs = [struct.unpack_from('<HH', raw, 24 + 4 * i) for i in range(ranked)]
    inverse = [-1] * ranked
    for i, (layer, expert) in enumerate(pairs):
        inverse[layer * experts + expert] = i
    assert inverse == list(struct.unpack_from('<24576i', raw, 24 + 4 * ranked))
    profile = dict(format=1, layers=layers, experts=experts, source=origin, ranked=pairs)
    data = json.dumps(profile, separators=(',', ':')) + '\n'
    profile_sha = hashlib.sha256(data.encode()).hexdigest()
    write('python/freetoken/moe/configs/q2_residency/strata_flash_next.json', data)

    gather = definition('prototype_expert_residency_v1.py', '_owner_gather')
    gather += '\n' + definition('prototype_expert_residency_v1.py', 'gather')
    ensure = definition('prototype_protected_admission_v1.py', '_ensure')
    ensure = ensure.replace('K: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr):',
                            'K: tl.constexpr, BK: tl.constexpr, BC: tl.constexpr,\n            COLLECT_STATS: tl.constexpr):')
    start = ensure.index('    s = tl.arange(0, 4)')
    ensure = ensure[:start] + '    if COLLECT_STATS:\n' + ''.join('    ' + line + '\n' for line in ensure[start:].splitlines())

    loader = definition('prototype_q2_protected_cache_v1.py', 'load_compact_banks')
    loader = loader.replace('*, method, device, dtype, dummy=False,', '*, protected, dynamic, method, device, dtype, dummy=False,')
    loader = loader.replace('    global PAYLOAD\n', '')
    loader = loader.replace("if PAYLOAD is not None or method.kind", "if method.kind")
    loader = loader.replace("Protected research loader only accepts one real GPU Q2 model", "Protected residency requires real GPU Q2 expert banks")
    loader = loader.replace("    protected = tuple(SETTINGS['protected'])", "    protected = tuple(protected)")
    loader = loader.replace("    dynamic = SETTINGS['dynamic']\n", '')
    loader = loader.replace('    begin, protected_begin = 2 * experts, 2 * experts + dynamic', '    protected_begin = 2 * experts + dynamic')
    loader = loader.replace('    PAYLOAD = dict(', '    ownership = Q2Ownership(')
    loader = loader.replace("return ExpertBanks('q2_0', sources, kind=method.kind, kernel=method.kernel.name, layout=layout)",
                            "return ExpertBanks('q2_0', sources, kind=method.kind, kernel=method.kernel.name, layout=layout), ownership")
    loader = loader.replace("            print('protected-Q2 loaded layer', layer, 'compact host rows', host_counts[layer], flush=True)",
                            "            logger.info_rank0(f'Protected Q2 layer {layer}: {host_counts[layer]} compact host rows')")

    cache = definition('prototype_q2_protected_cache_v1.py', 'ProtectedQ2Cache')
    cache = cache.replace('    def set_bank_sources(self, sources, layer_residency=None):', '''    def __init__(self, *args, ownership: Q2Ownership, **kwargs):
        self.protected_payload = ownership
        super().__init__(*args, **kwargs)

    def set_bank_sources(self, sources, layer_residency=None):''')
    cache = cache.replace('        payload = PAYLOAD', '        payload = self.protected_payload')
    cache = cache.replace("payload is None or sources", 'sources')
    cache = cache.replace('        self.protected_payload = payload\n', '')
    cache = cache.replace('Prototype cache is restricted', 'Protected cache is restricted')
    cache = cache.replace("0 < n <= min(p['dynamic'], 128)", '0 < n <= 128')
    cache = cache.replace('Prototype decode route geometry is unsupported', 'Protected decode supports at most 128 routes per layer')
    cache = cache.replace("triton.next_power_of_2(p['dynamic']), num_warps=4)",
                          "triton.next_power_of_2(p['dynamic']), self.collect_stats, num_warps=4)")
    cache = cache.replace('        begin, end = 2 * self.num_experts,', '''        if self.collect_decode_freq:
            self.decode_freq[layer_id].scatter_add_(0, expert_ids.view(-1).to(torch.int64),
                torch.ones(n, dtype=torch.int64, device=self.device))
        begin, end = 2 * self.num_experts,''')
    cache = cache.replace("    def rebuild(self, new_cache_size):\n        raise RuntimeError('Protected-cache research prototype cannot rebuild; restart required')", '''    def validate_rebuild(self, cache_size):
        """Reject an owner-pool resize before the engine destroys live graphs."""
        super().validate_rebuild(cache_size)
        if cache_size != self.protected_payload.slots:
            raise ValueError('Protected Q2 expert capacity is fixed; reload the model to change it')

    def rebuild(self, cache_size):
        """Same-capacity rebuild resets dynamic copies and retains all owner addresses."""
        self.validate_rebuild(cache_size)
        self.reset()
        self.reset_decode_miss_stats()
        self.decode_freq.zero_()
        self.prefill_hit_rows = self.prefill_total_rows = 0''')
    cache = cache.replace('        self.expert_recency.fill_(-1)', '''        self.expert_recency.fill_(-1)
        self.num_indices.zero_()
        self._pending_src_layer = None
        self._pending_whole_layer = False
        # All users of the borrowed prefill views finished before these tags reset.
        self._prefill_buffer_layer = [None, None]
        self._prefill_buffer_released = [True, True]
        self._prefill_buffer_has_release_event = [False, False]''')
    # The ownership bundle is explicit and belongs to this cache, not the process.
    for name in ['sources', 'caches', 'host_indices', 'protected_slots', 'initial_map', 'slots', 'dynamic',
                 'protected_begin', 'layers', 'experts', 'layout', 'host_bytes', 'gpu_bank_bytes']:
        cache = cache.replace(f"payload['{name}']", f'payload.{name}').replace(f"p['{name}']", f'p.{name}')
    cache = cache.replace("payload['protected_slots'][layer, e]", 'payload.protected_slots[layer, e]')
    module = '''"""Compressed Q2 experts with disjoint protected, dynamic and prefill regions.

Owners are permanent GPU rows or compact pinned host rows. Dynamic cache entries
are disposable copies. Keeping those roles separate prevents a GPU-only owner
from being evicted or overwritten by prefill, including across graph replay.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
import triton
import triton.language as tl
from freetoken.utils import init_logger
from .offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass(frozen=True)
class Q2Ownership:
    sources: dict
    caches: dict
    host_indices: list
    protected_slots: dict
    initial_map: torch.Tensor
    slots: int
    dynamic: int
    protected_begin: int
    layers: int
    experts: int
    layout: dict
    host_bytes: int
    gpu_bank_bytes: int


def protected_ranking(layers, experts, count):
    """Use the frozen Strata ranking, never routes from evaluation requests."""
    path = Path(__file__).parent / 'configs/q2_residency/strata_flash_next.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != '__PROFILE_SHA__':
        raise ValueError('Packaged Q2 expert ranking has changed')
    profile = json.loads(raw)
    if (profile['format'], profile['layers'], profile['experts']) != (1, layers, experts):
        raise ValueError('Protected Q2 ranking does not match this model geometry')
    ranked = [tuple(pair) for pair in profile['ranked']]
    if not 0 < count <= len(ranked):
        raise ValueError('Protected Q2 count exceeds the frozen expert ranking')
    return ranked[:count]


'''.replace('__PROFILE_SHA__', profile_sha)
    write('python/freetoken/moe/q2_residency.py', module + gather + '\n\n' + ensure + '\n\n' + loader + '\n\n' + cache)

    path = SOURCE / 'python/freetoken/engine/config.py'
    text = path.read_text(encoding='utf-8')
    text = text.replace('    moe_cache_size: int = 0\n', '''    moe_cache_size: int = 0
    # Explicit opt-in: GPU-only owners plus compact host complements, fixed for this load.
    moe_q2_resident_experts: int = 0
    moe_q2_dynamic_slots: int = 1024
''')
    text = text.replace('    def __post_init__(self):\n', '''    def __post_init__(self):
        if self.moe_q2_resident_experts < 0:
            raise ValueError('moe_q2_resident_experts must be nonnegative')
        if self.moe_q2_resident_experts:
            if (self.moe_strategy != 'offload' or self.moe_backend is not None
                    or self.moe_cpu_layers or self.use_dummy_weight
                    or self.moe_cache_auto or self.moe_cache_rate is not None
                    or self.moe_cache_policy != 'lru' or self.max_running_req != 1):
                raise ValueError('Protected Q2 requires explicit GPU offload, fixed LRU capacity and max_running_req=1')
            if self.moe_q2_dynamic_slots < 512:
                raise ValueError('Protected Q2 requires at least 512 dynamic slots')
            expected = 1024 + self.moe_q2_dynamic_slots + self.moe_q2_resident_experts
            if self.moe_cache_size != expected:
                raise ValueError(f'Protected Q2 moe_cache_size must be {expected}, got {self.moe_cache_size}')
''')
    write('python/freetoken/engine/config.py', text)

    path = SOURCE / 'python/freetoken/engine/engine.py'
    text = path.read_text(encoding='utf-8')
    text = text.replace('        method = shared_offload_method(self.model)\n        num_moe_layers', '''        method = shared_offload_method(self.model)
        ownership = None
        if config.moe_q2_resident_experts:
            from freetoken.moe.q2_residency import load_compact_banks, protected_ranking, ProtectedQ2Cache
            if (method is None or method.cfg.scheme.weight.elem != 'q2_0_q8'
                    or (config.model_config.num_moe_layers, config.model_config.num_experts) != (48, 512)):
                raise ValueError('Protected Q2 requires the complete Flash-Next GSQ-RCO expert path')
            protected = protected_ranking(48, 512, config.moe_q2_resident_experts)
        num_moe_layers''', 1)
    text = text.replace('                banks = load_expert_banks(\n', '''                loader = load_compact_banks if config.moe_q2_resident_experts else load_expert_banks
                owner_args = dict(protected=protected, dynamic=config.moe_q2_dynamic_slots) if config.moe_q2_resident_experts else {}
                loaded = loader(
                    **owner_args,
''', 1)
    # Keyword expansion cannot precede positional arguments.
    text = text.replace('                    **owner_args,\n                    config.model_path,\n                    config.model_config,',
                        '                    config.model_path,\n                    config.model_config,\n                    **owner_args,', 1)
    text = text.replace('        except PinFailed as exc:\n', '''                if config.moe_q2_resident_experts:
                    banks, ownership = loaded
                else:
                    banks = loaded
        except PinFailed as exc:
''', 1)
    text = text.replace('        cache = OffloadMoeCache(\n', '''        cache_type = ProtectedQ2Cache if ownership is not None else OffloadMoeCache
        cache = cache_type(
            **({'ownership': ownership} if ownership is not None else {}),
''', 1)
    write('python/freetoken/engine/engine.py', text)

    path = SOURCE / 'python/freetoken/server/args.py'
    text = path.read_text(encoding='utf-8').replace('    moe_cache_group = parser.add_mutually_exclusive_group()', '''    parser.add_argument('--moe-q2-resident-experts', type=int, default=ServerArgs.moe_q2_resident_experts,
        help='Permanent GPU Q2 expert owners; requires fixed cache capacity and single-request offload.')
    parser.add_argument('--moe-q2-dynamic-slots', type=int, default=ServerArgs.moe_q2_dynamic_slots,
        help='Dynamic LRU slots alongside permanent Q2 owners (default: 1024).')
    moe_cache_group = parser.add_mutually_exclusive_group()''')
    write('python/freetoken/server/args.py', text)

    path = SOURCE / 'tests/models/qwen4_exp/test_gguf_execution.py'
    text = path.read_text(encoding='utf-8').replace('        torch.testing.assert_close(full[-1:],decode,rtol=0,atol=0)', '''        # Parallel decode reduces K chunks in a different FP32 order from prefill.
        torch.testing.assert_close(full[-1:],decode,rtol=3e-5,atol=3e-7)
        relative_l2 = (full[-1:] - decode).norm() / full[-1:].norm().clamp_min(1e-10)
        self.assertLess(float(relative_l2), 2e-6)''')
    write('tests/models/qwen4_exp/test_gguf_execution.py', text)
    path = SOURCE / 'python/freetoken/version.py'
    write('python/freetoken/version.py', path.read_text().replace('strataq2dev5', 'strataq2dev6'))
    for file in SOURCE.rglob('*.py'):
        ast.parse(file.read_text(encoding='utf-8'), filename=str(file))
    report = dict(version='0.1.3+speed.strataq2dev6', packaged_profile_sha256=profile_sha,
                  source=str(SOURCE), status='source_port_requires_runtime_validation',
                  prototype_origin_sha256=hashlib.sha256((ROOT / 'prototype_q2_protected_cache_v1.py').read_bytes()).hexdigest())
    (ROOT / 'dev6-source-port-v1.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
