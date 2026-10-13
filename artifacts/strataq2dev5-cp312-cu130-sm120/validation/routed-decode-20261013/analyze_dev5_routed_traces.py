"""Actual installed routed-expert traces; kernel sums are not wall latency."""
import json
from pathlib import Path
from analyze_freetoken_decode_traces import analyze
ROOT = Path(__file__).resolve().parent
target = ROOT / 'dev5-routed-trace-analysis-v1.json'
if target.exists():
    raise FileExistsError(target)
rows = []
for path in sorted(ROOT.glob('dev5-native-diagnostic-4096-v1.*.trace.json')):
    row = analyze(path)
    families = {k: 0. for k in ['gguf_main', 'expert_routed', 'expert_grouped', 'expert_alignment', 'expert_copy', 'lru', 'other']}
    for kernel in row['kernels']:
        name = kernel['name']
        if name == '_bf16_rows' or 'gguf_native' in name:
            key = 'gguf_main'
        elif name == '_routed_vector':
            key = 'expert_routed'
        elif name == '_grouped':
            key = 'expert_grouped'
        elif name.startswith('_moe_align'):
            key = 'expert_alignment'
        elif 'fast_index_copy_multi' in name:
            key = 'expert_copy'
        elif 'lru_ensure' in name:
            key = 'lru'
        else:
            key = 'other'
        families[key] += kernel['gpu_or_cpu_ms_per_decode']
    row['family_ms_per_decode'] = families
    rows.append(row)
    print(json.dumps(dict(path=path.name, decode_forwards=row['decode_forwards'], gpu_sum_ms=row['gpu_kernel_ms_per_decode'], gpu_union_ms=row['gpu_kernel_union_ms_per_decode'], families=families, top=row['kernels'][:5])))
if len(rows) != 4:
    raise RuntimeError('Four actual diagnostic traces required')
target.write_text(json.dumps(dict(rows=rows, scope='32-token diagnostic GPU decode work only; separate 12 x 512 actual-delivery measurements are throughput evidence. CPU blocking waits must not be added to GPU durations.', full_runtime_acceptance=False), indent=2), encoding='utf-8')
