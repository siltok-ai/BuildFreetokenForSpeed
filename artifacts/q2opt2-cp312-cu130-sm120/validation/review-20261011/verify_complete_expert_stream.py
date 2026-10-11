"""流式读取真实完整专家，不同时加载主干 / GPU，也不把它记作推理验收。"""
import json
from pathlib import Path
import time
from types import SimpleNamespace

from freetoken.models.qwen4_exp.q2_source import iter_q2_expert_pieces

root = Path(__file__).parent / 'checkpoint-finalize'
config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
source = config['freetoken_expert_source']
model = SimpleNamespace(expert_source=source, num_layers=source['layers'],
                        num_experts=source['experts_per_layer'], hidden_size=source['hidden'],
                        moe_intermediate_size=source['intermediate'])
started = time.perf_counter()
total, covered = 0, {}
for layer, first, last, pieces in iter_q2_expert_pieces(root, model):
    if first != covered.get(layer, 0):
        raise RuntimeError('incomplete expert sequence')
    covered[layer] = last
    total += sum(piece.numel() for piece in pieces.values())
if covered != {layer: model.num_experts for layer in range(model.num_layers)} or total != source['expert_bytes']:
    raise RuntimeError('incomplete expert payload')
result = {'scope': 'complete_real_expert_source_read_only; no inference or bank allocation',
          'layers': len(covered), 'experts_per_layer': model.num_experts, 'bytes': total,
          'seconds': time.perf_counter() - started, 'passed': True}
(Path(__file__).parent / 'complete-expert-stream.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result), flush=True)
