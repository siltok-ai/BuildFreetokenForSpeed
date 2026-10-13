"""Separate owner gather/admission GPU work from unchanged arithmetic families."""
import json
from pathlib import Path
from analyze_freetoken_decode_traces import analyze

ROOT = Path(__file__).resolve().parent
target = ROOT / 'prototype-protected-cache-trace-analysis-11264-v1.json'
if target.exists():
    raise FileExistsError(target)
rows = []
for path in sorted(ROOT.glob('prototype-protected-cache-decode-only-11264-v1.*.trace.json')):
    row = analyze(path)
    families = {k: 0.0 for k in ('gguf_main', 'expert_routed', 'expert_grouped', 'expert_alignment', 'expert_copy', 'admission', 'other')}
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
        elif name == '_owner_gather' or 'fast_index_copy_multi' in name:
            key = 'expert_copy'
        elif name == '_ensure' or 'lru_ensure' in name:
            key = 'admission'
        else:
            key = 'other'
        families[key] += kernel['gpu_or_cpu_ms_per_decode']
    row['family_ms_per_decode'] = families
    rows.append(row)
    print(json.dumps(dict(trace=path.name, decode_forwards=row['decode_forwards'],
                         gpu_sum_ms=row['gpu_kernel_ms_per_decode'],
                         gpu_union_ms=row['gpu_kernel_union_ms_per_decode'], families=families)))
if len(rows) != 4:
    raise RuntimeError('Expected four completed diagnostic traces')
target.write_text(json.dumps(dict(rows=rows, full_runtime_acceptance=False,
    scope='32-token prototype graph diagnostics; actual throughput comes from separate long-output measurements. CPU blocking waits overlap GPU work and are not additive.'), indent=2), encoding='utf-8')
