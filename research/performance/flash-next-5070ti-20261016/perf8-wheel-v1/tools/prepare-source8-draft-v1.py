"""Stage indexed draft heads and a paired whole-cycle policy on a fresh source tree."""
from pathlib import Path
import base64
import hashlib
import shutil
import zlib

R=Path(__file__).parent
S=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate8\build-source')
ST=R/'source8-patch-stages-v1'

def edit(stage,name,change):
    """Retain exact before/after files for immutable patch generation later."""
    p=S/name
    saved=ST/stage/'before'/name
    before=saved.read_bytes() if saved.exists() else (p.read_bytes() if p.exists() else b'')
    after=change(before.decode('utf-8').replace('\r\n','\n')).replace('\r\n','\n').encode()
    assert before.replace(b'\r\n',b'\n')!=after,name
    for side,data in [('before',before),('after',after)]:
        target=ST/stage/side/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(after)

def replace(s,a,b):
    assert s.count(a)==1,(a[:80],s.count(a))
    return s.replace(a,b)

def main():
    if Path(r'C:\workspace\wt\freetoken-mats\flash-perf\build\profiles\strataq2perf7.json').exists():
        raise RuntimeError('Profile7 already frozen; never modify its source with this preparation script')
    proto=(R/'mapped_q5_proto4_r1.cuh').read_text().replace('mapped_q5_candidate','freetoken_draft_q5').replace('DraftMappedQ5','GGUFIndexedDraftQ5')
    edit('44-indexed-draft-head','python/freetoken/kernel/csrc/jit/gguf_mmvq.cuh',lambda s:s.replace(
        '// 小批多列投影复用已移植的Strata算术，保持每列归约顺序。',
        '// Small batches reuse the Strata arithmetic and each column reduction order.')+'\n\n'+proto)
    edit('44-indexed-draft-head','python/freetoken/kernel/aot.py',lambda s:s.replace('gguf_mmvq_r2','gguf_mmvq_r3'))
    def native(s):
        s=replace(s,'"gguf_mmvq_r2"','"gguf_mmvq_r3"')
        s=replace(s,'("project_interleaved", "GGUFInterleavedMMVQ::project")]',
            '("project_interleaved", "GGUFInterleavedMMVQ::project"),\n                       ("project_indexed_q5", "GGUFIndexedDraftQ5::project")]')
        return s+'''

def indexed_q5(packed, x, ids):
    """Project indexed original Q5 rows; the target head remains complete."""
    if x.shape != (1, 2560) or ids.ndim != 1 or ids.dtype != torch.int64:
        raise ValueError('Indexed draft Q5 requires one2560-wide row and int64 row indices')
    q = quantize_q8_1(x)
    out = torch.empty((1, len(ids)), device=x.device, dtype=torch.float32)
    _module().project_indexed_q5(packed, q, ids, out, torch.cuda.current_stream(x.device).cuda_stream)
    return out
'''
    edit('44-indexed-draft-head','python/freetoken/kernel/gguf_native.py',native)
    data=Path(r'C:\workspace\models\strata\flash-next-mtp-reference\draft_vocab.bin').read_bytes()
    digest=hashlib.sha256(data).hexdigest()
    assert digest=='b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91'
    encoded=base64.b85encode(zlib.compress(data,9)).decode()
    chunks='\n'.join('    '+repr(encoded[i:i+100]) for i in range(0,len(encoded),100))
    vocab='''"""Fixed Strata CJK draft rows; only the speculative head uses this map.

Source: local Strata reference draft_vocab.bin,106299 little-endian int32 IDs.
The original target vocabulary and all model weights remain unchanged.
"""
import base64
import hashlib
import zlib
import torch

SHA256 = %r
DATA = (
%s
)

def load(device):
    """Verify the fixed row map before creating its small owned CUDA index."""
    raw=zlib.decompress(base64.b85decode(DATA))
    if len(raw)!=106299*4 or hashlib.sha256(raw).hexdigest()!=SHA256:
        raise ValueError('Draft vocabulary identity changed')
    ids=torch.frombuffer(bytearray(raw),dtype=torch.int32).long().sort().values
    return ids.to(device)
'''%(digest,chunks)
    edit('45-paired-draft-policy','python/freetoken/models/qwen4_exp/mtp_draft_vocab.py',lambda s:vocab)
    policy=(R/'mtp_paired_head_policy_v8.py').read_text()
    policy=policy[:policy.index('\n\ndef enable(')].replace('from functools import partial\n','')
    policy=policy.replace("def __init__(self,arms=('full-3','cjk-3')):","def __init__(self,arms=('full-3','cjk-2')):")
    edit('45-paired-draft-policy','python/freetoken/models/qwen4_exp/mtp_draft_policy.py',lambda s:policy)
    def config(s):
        s=replace(s,'    mtp_compact_state: bool = False','    mtp_compact_state: bool = False\n    mtp_draft_policy: str = "full"')
        return replace(s,'    def __post_init__(self):','    def __post_init__(self):\n        if self.mtp_draft_policy not in ("full", "paired_cjk"):\n            raise ValueError("MTP draft policy must be full or paired_cjk")')
    edit('45-paired-draft-policy','python/freetoken/engine/config.py',config)
    edit('45-paired-draft-policy','python/freetoken/engine/engine.py',lambda s:replace(s,
        'self.model.prepare_mtp_runtime(config.mtp_runtime_path, config.mtp_max_context)',
        'self.model.prepare_mtp_runtime(config.mtp_runtime_path, config.mtp_max_context, draft_policy=config.mtp_draft_policy)'))
    def model(s):
        s=replace(s,'def prepare_mtp_runtime(self, path: str, max_context: int) -> None:',
            'def prepare_mtp_runtime(self, path: str, max_context: int, *, draft_policy="full") -> None:')
        return replace(s,'MtpDrafter(Path(path), self, max_context=max_context)',
            'MtpDrafter(Path(path), self, max_context=max_context, draft_policy=draft_policy)')
    edit('45-paired-draft-policy','python/freetoken/models/qwen4_exp/model.py',model)
    def drafter(s):
        s=replace(s,'from freetoken.kernel.gguf_native import bf16_gemv','from freetoken.kernel.gguf_native import bf16_gemv, indexed_q5')
        s=replace(s,'def __init__(self, runtime: Path, main_model, *, max_context=8192):',
            'def __init__(self, runtime: Path, main_model, *, max_context=8192, draft_policy="full"):')
        point='        self.owned_bytes = sum((t.numel() * t.element_size() for t in self.weights.values())) + 2 * self.keys.numel() * self.keys.element_size()'
        s=replace(s,point,point+'''
        self.draft_policy = draft_policy
        self.draft_vocab = None
        if draft_policy == 'paired_cjk':
            from .mtp_draft_vocab import load
            if self.head.packed_0.numel() != 248320 * 1760:
                raise ValueError('CJK draft policy requires the original248320x2560 Q5_K head')
            self.draft_vocab = load(self.device)
            self.owned_bytes += self.draft_vocab.numel() * self.draft_vocab.element_size()
''')
        s=replace(s,'def step(self, residual, token, cell, *, positions=None, attend=None):',
            'def step(self, residual, token, cell, *, positions=None, attend=None, vocab_ids=None):')
        s=replace(s,'        logits = self.head.quant_method.apply(self.head, mixed)\n        return (logits.argmax(-1).to(torch.int32), r)',
            '''        if vocab_ids is None:
            logits = self.head.quant_method.apply(self.head, mixed)
            token = logits.argmax(-1).to(torch.int32)
        else:
            logits = indexed_q5(self.head.packed_0, mixed, vocab_ids)
            token = vocab_ids[logits.argmax(-1)].to(torch.int32)
        return token, r''')
        return s
    edit('45-paired-draft-policy','python/freetoken/models/qwen4_exp/mtp_drafter.py',drafter)
    def graph(s):
        s=replace(s,'def __init__(self, draft, stream, *, attention_width):','def __init__(self, draft, stream, *, attention_width, vocab_ids=None):')
        return replace(s,'draft.step(self.r, self.token, 0, positions=self.cell, attend=self.attend)',
            'draft.step(self.r, self.token, 0, positions=self.cell, attend=self.attend, vocab_ids=vocab_ids)')
    edit('45-paired-draft-policy','python/freetoken/models/qwen4_exp/mtp_draft_graph.py',graph)
    def runtime(s):
        s=replace(s,'import copy\n','import copy\nimport time\n')
        s=replace(s,'        self.graphs = {}\n        self.draft_graphs = None\n        self.active_owner = None',
            '        self.graphs = {}\n        self.draft_graphs = None\n        self.full_draft_graphs = None\n        self.draft_policy = None\n        self._policy_req = None\n        self._policy_previous = None\n        self.active_owner = None')
        s=replace(s,'        self.draft_graphs = DraftGraphs(self.draft, self.engine.stream, attention_width=self.draft.max_context)',
            '''        self.draft_graphs = DraftGraphs(self.draft, self.engine.stream,
            attention_width=self.draft.max_context, vocab_ids=self.draft.draft_vocab)
        if self.draft.draft_policy == 'paired_cjk':
            self.full_draft_graphs = DraftGraphs(self.draft, self.engine.stream, attention_width=self.draft.max_context)
        self._policy_req = self._policy_previous = None''')
        s=replace(s,'            self.draft_graphs = None\n        self.active_owner = None',
            '''            self.draft_graphs = None
        if self.full_draft_graphs is not None:
            self.full_draft_graphs.close()
            self.full_draft_graphs = None
        self._policy_req = self._policy_previous = None
        self.active_owner = None''')
        s=replace(s,'    def forward(self, batch, manager, token_pool, eos_ids, clip, anchor_id):',
            '''    def forward(self, batch, manager, token_pool, eos_ids, clip, anchor_id):
        started = time.perf_counter() if self.draft.draft_policy == 'paired_cjk' else None''')
        s=replace(s,'        depth = window_depth(req, manager, self.draft.max_context)\n        manager.allocate_paged',
            '''        depth = window_depth(req, manager, self.draft.max_context)
        action = 'full-3'
        if started is not None:
            from .mtp_draft_policy import PairedPolicy
            if self._policy_req is not req:
                self._policy_req = req
                self.draft_policy = PairedPolicy()
                self._policy_previous = None
            if self._policy_previous is not None:
                stamp, previous_position, arm, actual, accepted = self._policy_previous
                if previous_position == req.cached_len and actual == int(arm[-1]):
                    self.draft_policy.observe(arm, accepted, started - stamp)
            action = self.draft_policy.choose()
            depth = min(depth, int(action[-1]))
        manager.allocate_paged''')
        s=replace(s,'            candidates = self.draft_graphs.chain(residual, pending, position - 1, depth)',
            '''            selected = self.full_draft_graphs if started is not None and action == 'full-3' else self.draft_graphs
            candidates = selected.chain(residual, pending, position - 1, depth)''')
        s=replace(s,'        return ForwardOutput(tokens[-1:], bundle[-1:], event)',
            '''        if started is not None:
            self._policy_previous = (started, req.cached_len, action, depth, accepted)
        return ForwardOutput(tokens[-1:], bundle[-1:], event)''')
        return s
    edit('45-paired-draft-policy','python/freetoken/models/qwen4_exp/mtp_runtime.py',runtime)
    # CPU extension bytes are accepted inputs; this source-stage copy is not a new build.
    for p in Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand6\Lib\site-packages\freetoken').rglob('*.pyd'):
        target=S/'python/freetoken'/p.relative_to(Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand6\Lib\site-packages\freetoken'))
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(p,target)
    print('Source8 staged; version remains experimental6 until explicit profile freeze, not accepted materials.')

if __name__=='__main__':main()
