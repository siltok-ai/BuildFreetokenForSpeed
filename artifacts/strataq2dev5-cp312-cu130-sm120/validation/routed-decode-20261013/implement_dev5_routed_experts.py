"""Apply the validated prototype design to a fresh replayed FreeToken tree."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'candidate-source-dev5-working/build-source'
path=SOURCE/'python/freetoken/moe/fused_q2_q8.py'
base=ROOT/'dev4-built-native-materials-v2/build-source/python/freetoken/moe/fused_q2_q8.py'
if path.read_bytes()!=base.read_bytes():raise RuntimeError('Fresh dev4 replay expected')
gate=json.loads((ROOT/'prototype-expert-dp4a-v13-component.json').read_text())
if not gate['passed']:raise RuntimeError('DP4A component failed')
proto=(ROOT/'prototype_expert_dp4a_v13.py').read_text(encoding='utf-8')
tree=ast.parse(proto)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_dp4a_vector','vector_project')]
parts=[]
for n in nodes:
    start=min([n.lineno]+[d.lineno for d in n.decorator_list])-1
    parts.append('\n'.join(proto.splitlines()[start:n.end_lineno]))
kernel='\n\n'.join(parts).replace('_dp4a_vector','_routed_vector').replace('vector_project','_routed_project')
kernel=kernel.replace('    from freetoken.moe.fused_q2_q8 import quantize_expert_activation\n','')
text=path.read_text(encoding='utf-8')
oldtree=ast.parse(text);old=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef) and n.name=='_vector')
start=min([old.lineno]+[d.lineno for d in old.decorator_list])-1
lines=text.splitlines()
text='\n'.join(lines[:start])+ '\n'+kernel+'\n'+'\n'.join(lines[old.end_lineno:])+'\n'
text=text.replace('    sorted_ids, experts, padded = moe_align_block_size(topk_ids, 16, gate_up.shape[0])',
    '    vector_decode = not is_prefill and tokens <= 8\n    if not vector_decode:\n        sorted_ids, experts, padded = moe_align_block_size(topk_ids, 16, gate_up.shape[0])')
text=text.replace('    def project(value, bank, result, input_topk):\n        q, scales = quantize_expert_activation(value)',
    '    def project(value, bank, result, input_topk):\n        if vector_decode:\n            return _routed_project(value, bank, result, topk_ids, input_topk)\n        q, scales = quantize_expert_activation(value)')
text=text.replace('    # Batch-dependent reduction orders can change subsequent Q8 codes and routing.\n    # Use the same integer-dot / FP32 accumulation for prefill, cache tails and decode.',
    '    # Decode is sparse in route space, independent of total cache capacity.\n    # Integer chunk dots and FP32 activation scales match prefill; float reduction\n    # trees may differ within the validated bound. Large prefill reuses weights.')
text=text.replace('    if x.dtype != torch.float32 or gate_up.dtype != torch.uint8 or down.dtype != torch.uint8:',
    '    if x.ndim != 2 or gate_up.ndim != 3 or down.ndim != 3 or topk_ids.ndim != 2:\n        raise ValueError("GGUF Q2xQ8 expects matrix activations/routes and 3D banks")\n    if topk_weights.dtype != torch.float32 or topk_ids.dtype not in (torch.int32, torch.int64):\n        raise ValueError("GGUF Q2xQ8 expects FP32 routing weights and integer indices")\n    if any(t.device != x.device for t in (gate_up, down, topk_weights, topk_ids)):\n        raise ValueError("GGUF Q2xQ8 operands must share one CUDA device")\n    if x.dtype != torch.float32 or gate_up.dtype != torch.uint8 or down.dtype != torch.uint8:')
text=text.replace('    if (hidden % 64 or intermediate % 64 or gate_up.shape[2] != hidden // 64 * 18',
    '    if (hidden <= 0 or intermediate <= 0 or topk <= 0 or gate_up.shape[0] <= 0\n            or gate_up.shape[1] % 2 or hidden % 64 or intermediate % 64\n            or gate_up.shape[2] != hidden // 64 * 18')
text=text.replace('    gate_values = torch.empty((routes, 2 * intermediate)',
    '    if not tokens:\n        return torch.empty_like(x)\n    gate_values = torch.empty((routes, 2 * intermediate)')
text=text.replace('    low=tl.load(W+base,valid,0).to(tl.uint16)',
    '    # Q2 blocks are 18 bytes: load headers as bytes, never assume int32 alignment.\n    low=tl.load(W+base,valid,0).to(tl.uint16)')
text=text.replace('    products=tl.inline_asm_elementwise',
    '    # The unsigned two-bit codes need the exact integer sum(x) correction.\n    products=tl.inline_asm_elementwise')
ast.parse(text)
path.write_text(text,encoding='utf-8')
print(json.dumps(dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),prototype_sha256=gate['source_sha256'])))
