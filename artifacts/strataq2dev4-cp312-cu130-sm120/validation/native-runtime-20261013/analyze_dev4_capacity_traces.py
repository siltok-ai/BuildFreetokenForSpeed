"""Attribute realized decode device work for two actual cache capacities."""
import json
from pathlib import Path
from analyze_freetoken_decode_traces import analyze
ROOT=Path(__file__).resolve().parent
target=ROOT/'dev4-capacity-trace-analysis-v1.json'
if target.exists():raise FileExistsError(target)
rows=[]
for capacity in [1024,4096]:
    for p in sorted(ROOT.glob(f'dev4-native-diagnostic-{capacity}-v1.*.trace.json')):
        row=analyze(p);row['cache_capacity']=capacity
        families=dict(gguf_main=0.,expert_grouped=0.,expert_copy=0.,lru=0.,other=0.)
        for k in row['kernels']:
            name=k['name']
            if name=='_bf16_rows' or 'gguf_native' in name:family='gguf_main'
            elif name=='_grouped':family='expert_grouped'
            elif 'fast_index_copy_multi' in name:family='expert_copy'
            elif 'lru_ensure' in name:family='lru'
            else:family='other'
            families[family]+=k['gpu_or_cpu_ms_per_decode']
        row['family_ms_per_decode']=families
        rows.append(row)
report=dict(scope='Short 32-token diagnostic traces, not sustained 512-token throughput. GPU decode markers only; cold misses change through the trace.',
    rows=rows,no_cpu_waits_added=True)
target.write_text(json.dumps(report,indent=2),encoding='utf-8')
for row in rows:print(json.dumps(dict(path=Path(row['path']).name,capacity=row['cache_capacity'],gpu_ms=row['gpu_kernel_ms_per_decode'],families=row['family_ms_per_decode'])))
