"""Combine validated Q2 ownership and admission probes for bounded engine research.

Installed/freezing inputs are never edited. Rebuild and CPU/hybrid execution are
intentionally rejected by this prototype; it is not a release candidate.
"""
import hashlib
import json
import math
from pathlib import Path
import struct

import capture_freetoken_reference
import torch
import triton

from prototype_expert_residency_v1 import gather
from prototype_protected_admission_v1 import _ensure
from freetoken.moe.offload_cache import OffloadMoeCache

ROOT = Path(__file__).resolve().parent
SETTINGS = {}
PAYLOAD = None


def read_profile(path, layers, experts):
    raw = Path(path).read_bytes()
    if len(raw) < 24 or raw[:4] != b'STRP':
        raise ValueError('Not a Strata v1 profile')
    version, nl, ne, slots, ranked = struct.unpack_from('<5I', raw, 4)
    if (version, nl, ne) != (1, layers, experts) or not (ranked <= slots <= layers * experts):
        raise ValueError('Profile geometry/version mismatch')
    if len(raw) != 24 + 4 * ranked + 4 * layers * experts:
        raise ValueError('Profile must contain the complete ranked list and mapping')
    pairs = [struct.unpack_from('<HH', raw, 24 + 4 * i) for i in range(ranked)]
    if len(set(pairs)) != len(pairs) or any(l >= layers or e >= experts for l, e in pairs):
        raise ValueError('Invalid or duplicated expert ranking')
    expected = [-1] * (layers * experts)
    for ordinal, (layer, expert) in enumerate(pairs):
        expected[layer * experts + expert] = ordinal
    actual = list(struct.unpack_from('<' + str(layers * experts) + 'i', raw, 24 + 4 * ranked))
    if expected != actual:
        raise ValueError('Profile mapping disagrees with its ranking')
    return pairs, hashlib.sha256(raw).hexdigest()


def load_compact_banks(model_path, model_config, *, method, device, dtype, dummy=False,
                       parallel=None, decode_target='gpu', layer_residency=None, **kwargs):
    global PAYLOAD
    from freetoken.layers.quantization import QuantKind
    from freetoken.kernel.pinned import alloc_pinned_tensor
    from freetoken.moe.expert_banks import ExpertBanks
    from freetoken.moe.expert_pieces import iter_expert_pieces

    if PAYLOAD is not None or method.kind is not QuantKind.Q2_0 or dummy or decode_target != 'gpu' or layer_residency:
        raise ValueError('Protected research loader only accepts one real GPU Q2 model')
    layers, experts = model_config.num_moe_layers, model_config.num_experts
    protected = tuple(SETTINGS['protected'])
    if len(set(protected)) != len(protected) or any(not (0 <= l < layers and 0 <= e < experts) for l, e in protected):
        raise ValueError('Invalid protected ownership set')
    dynamic = SETTINGS['dynamic']
    if dynamic < experts:
        raise ValueError('Dynamic region must fit a complete expert layer')
    begin, protected_begin = 2 * experts, 2 * experts + dynamic
    slots = protected_begin + len(protected)
    layout = method.layout()
    if tuple(layout) != ('gate_up', 'down') or any(s.resident or s.dtype is not torch.uint8 for s in layout.values()):
        raise ValueError('Unexpected native Q2 bank layout')
    protected_slots = {pair: protected_begin + i for i, pair in enumerate(protected)}
    host_indices = [[-1] * experts for _ in range(layers)]
    host_counts = []
    for layer in range(layers):
        n = 0
        for expert in range(experts):
            if (layer, expert) not in protected_slots:
                host_indices[layer][expert] = n
                n += 1
        host_counts.append(n)
    # GPU owners are uploaded straight into the final pool, not a second staging bank.
    caches = {role: torch.empty((slots, *spec.shape), dtype=spec.dtype, device=device)
              for role, spec in layout.items()}
    sources = {role: [alloc_pinned_tensor(n, *spec.shape, dtype=spec.dtype) if n
                      else torch.empty((0, *spec.shape), dtype=spec.dtype)
                      for n in host_counts] for role, spec in layout.items()}
    written = torch.zeros((layers, experts), dtype=torch.bool)
    for layer, e0, e1, pieces in iter_expert_pieces(model_path, model_config, method.kind, parallel=False):
        if not (0 <= layer < layers and 0 <= e0 < e1 <= experts) or bool(written[layer, e0:e1].any()):
            raise ValueError('Missing/out-of-range/duplicated source piece')
        packed = {role: torch.empty((e1 - e0, *spec.shape), dtype=spec.dtype) for role, spec in layout.items()}
        if method.pack(pieces, packed):
            raise ValueError('Native Q2 banks should not return separate alpha tensors')
        host_rows = [e - e0 for e in range(e0, e1) if host_indices[layer][e] >= 0]
        gpu_rows = [e - e0 for e in range(e0, e1) if host_indices[layer][e] < 0]
        if host_rows:
            host_dst = torch.tensor([host_indices[layer][e0 + row] for row in host_rows], dtype=torch.int64)
            host_src = torch.tensor(host_rows, dtype=torch.int64)
        if gpu_rows:
            gpu_dst = torch.tensor([protected_slots[layer, e0 + row] for row in gpu_rows], device=device, dtype=torch.int64)
            gpu_src = torch.tensor(gpu_rows, dtype=torch.int64)
        for role in layout:
            if host_rows:
                sources[role][layer].index_copy_(0, host_dst, packed[role].index_select(0, host_src))
            if gpu_rows:
                caches[role].index_copy_(0, gpu_dst, packed[role].index_select(0, gpu_src).to(device))
        written[layer, e0:e1] = True
        if e1 == experts:
            print('protected-Q2 loaded layer', layer, 'compact host rows', host_counts[layer], flush=True)
    if not bool(written.all()):
        raise ValueError('Incomplete Q2 ownership load')
    torch.cuda.synchronize(device)
    initial_map = torch.full((layers, experts), -1, dtype=torch.int32)
    for (layer, expert), slot in protected_slots.items():
        initial_map[layer, expert] = slot
    PAYLOAD = dict(sources=sources, caches=caches, host_indices=host_indices,
                   protected_slots=protected_slots, initial_map=initial_map,
                   slots=slots, dynamic=dynamic, protected_begin=protected_begin,
                   layers=layers, experts=experts, layout=layout,
                   host_bytes=sum(sum(t.numel() * t.element_size() for t in per_layer) for per_layer in sources.values()),
                   gpu_bank_bytes=sum(t.numel() * t.element_size() for t in caches.values()))
    return ExpertBanks('q2_0', sources, kind=method.kind, kernel=method.kernel.name, layout=layout)


class ProtectedQ2Cache(OffloadMoeCache):
    def set_bank_sources(self, sources, layer_residency=None):
        from freetoken.kernel.pinned import device_ptr
        payload = PAYLOAD
        if payload is None or sources is not payload['sources'] or self.cache_size != payload['slots']:
            raise ValueError('Protected cache must attach the exact loaded owner payload')
        if self.quant_format != 'q2_0' or self.decode_target != 'gpu' or self.cpu_layer_ids:
            raise ValueError('Prototype cache is restricted to pure GPU Q2')
        self.protected_payload = payload
        self.bank_sources = sources
        self.bank_caches = payload['caches']
        self.banks = [(sources[role], self.bank_caches[role]) for role in self.bank_schema]
        self.layer_residency = ['pinned'] * self.num_layers
        self.prefill_hit_d2d = False
        self._copy_fused_ok = True
        self._copy_dst_ptrs = torch.tensor([self.bank_caches[r].data_ptr() for r in self.bank_schema], device=self.device, dtype=torch.int64)
        row_bytes = [math.prod(self.bank_caches[r].shape[1:]) * self.bank_caches[r].element_size() for r in self.bank_schema]
        self._copy_feat_bytes = torch.tensor(row_bytes, device=self.device, dtype=torch.int64)
        self._owner_tables = []
        for layer in range(self.num_layers):
            pointers = []
            for bank, role in enumerate(self.bank_schema):
                host = sources[role][layer]
                host_base = device_ptr(host) if host.size(0) else 0
                gpu_base = self.bank_caches[role].data_ptr()
                pointers.append([host_base + payload['host_indices'][layer][e] * row_bytes[bank]
                                 if payload['host_indices'][layer][e] >= 0
                                 else gpu_base + payload['protected_slots'][layer, e] * row_bytes[bank]
                                 for e in range(self.num_experts)])
            self._owner_tables.append(torch.tensor(pointers, device=self.device, dtype=torch.int64))
        self._protected_initial_map = payload['initial_map'].to(self.device)
        reverse = torch.full((self.cache_size,), -1, dtype=torch.int32)
        for (layer, expert), slot in payload['protected_slots'].items():
            reverse[slot] = layer * self.num_experts + expert
        self._protected_reverse_map = reverse.to(self.device)
        self._whole_ids = torch.arange(self.num_experts, device=self.device, dtype=torch.int32)
        self._whole_count = torch.full((1,), self.num_experts, device=self.device, dtype=torch.int64)
        self._whole_dst = [self._whole_ids + b * self.num_experts for b in (0, 1)]
        self.reset()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def _gather(self, layer, dst, src, count, blocks):
        gather((self._copy_dst_ptrs, self._owner_tables[layer], self._copy_feat_bytes), dst, src, count, blocks)

    def ensure_experts(self, layer_id, expert_ids):
        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        n = expert_ids.numel()
        p = self.protected_payload
        if not (0 <= layer_id < self.num_layers) or not (0 < n <= min(p['dynamic'], 128)):
            raise ValueError('Prototype decode route geometry is unsupported')
        begin, end = 2 * self.num_experts, p['protected_begin']
        _ensure[(1,)](expert_ids, self.slot_for_id.view(-1), self.id_of_slot[begin:end],
                      self.usage[begin:end], self.step, self.src_indices, self.evict_slots,
                      self.num_indices, self.lru_stats[layer_id], layer_id * self.num_experts,
                      begin, end, p['dynamic'], n, triton.next_power_of_2(n),
                      triton.next_power_of_2(p['dynamic']), num_warps=4)

    def copy_missing(self):
        layer = self._pending_src_layer
        if layer is None:
            raise RuntimeError('No pending Q2 source layer')
        if self._pending_whole_layer:
            self._gather(layer, self._whole_dst[0], self._whole_ids, self._whole_count, 64)
        else:
            self._gather(layer, self.evict_slots, self.src_indices, self.num_indices, 32)

    def materialize_layer(self, layer_id):
        self._pending_src_layer = layer_id
        self._pending_whole_layer = True

    def _invalidate_prefill_buffer(self, buffer_id):
        # Admission never installs an expert mapping in either prefill-only buffer.
        pass

    def prefetch_prefill_layer(self, layer_id):
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError('Invalid source layer')
        buffer = layer_id % 2
        if self._prefill_buffer_layer[buffer] == layer_id:
            return
        if self._prefill_buffer_layer[buffer] is not None and not self._prefill_buffer_released[buffer]:
            raise RuntimeError('Prefill buffer reused before compute released it')
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer])
            self._gather(layer_id, self._whole_dst[buffer], self._whole_ids, self._whole_count, 64)
            self.prefill_ready_events[buffer].record(self.prefill_copy_stream)
        self._prefill_buffer_layer[buffer] = layer_id
        self._prefill_buffer_released[buffer] = False

    def reset(self):
        if self.prefill_copy_stream is not None:
            self.prefill_copy_stream.synchronize()
        torch.cuda.current_stream(self.device).synchronize()
        self.slot_for_id.copy_(self._protected_initial_map)
        self.id_of_slot.copy_(self._protected_reverse_map)
        self.usage.zero_()
        self.step.zero_()
        self.expert_recency.fill_(-1)

    def rebuild(self, new_cache_size):
        raise RuntimeError('Protected-cache research prototype cannot rebuild; restart required')


def install(protected=6144, dynamic=1024):
    global SETTINGS
    from freetoken.engine import engine
    for filename in ('prototype-expert-residency-v1.json', 'prototype-protected-admission-v1.json'):
        if not json.loads((ROOT / filename).read_text(encoding='utf-8'))['passed']:
            raise RuntimeError('Ownership/admission component gate is incomplete')
    component = json.loads((ROOT / 'prototype-q2-protected-cache-v2-component.json').read_text(encoding='utf-8'))
    if (not component['passed'] or component['evidence']['source_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise RuntimeError('Exact combined-cache prototype is not component validated')
    ranking, digest = read_profile(ROOT / 'strata-official-expert-profile.bin', 48, 512)
    if not (0 < protected <= len(ranking)):
        raise ValueError('Not enough verified ranked experts')
    SETTINGS = dict(protected=ranking[:protected], dynamic=dynamic, profile_sha256=digest,
                    scope='Frozen official Strata ranking; no evaluation prompt route training')
    engine.load_expert_banks = load_compact_banks
    engine.OffloadMoeCache = ProtectedQ2Cache


def evidence():
    p = PAYLOAD
    return dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                profile_sha256=SETTINGS.get('profile_sha256'), protected_count=len(SETTINGS['protected']),
                dynamic_count=SETTINGS['dynamic'], expert_cache_slots=p['slots'] if p else None,
                compact_host_bytes=p['host_bytes'] if p else None,
                gpu_bank_bytes=p['gpu_bank_bytes'] if p else None,
                process_engine_override=True, full_engine_acceptance=False,
                restrictions=['one real Q2 GPU model', 'fixed protected set', 'rebuild requires restart', 'decode routes <=128'])
