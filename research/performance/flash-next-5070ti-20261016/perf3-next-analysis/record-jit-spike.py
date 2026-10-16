"""保留冷形状JIT尖峰的时间相关证据；尚未修改冻结perf3算法。"""
from pathlib import Path
import datetime as dt
import hashlib
import json

RUN=Path(__file__).parent
CACHE=Path(r'C:\workspace\cache\triton\flash-perf-mtp3')
source=Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-03\candidate-source\build-source\python\freetoken\moe\q2_residency.py')
start=dt.datetime(2026,10,16,16,11,50).astimezone().timestamp()
end=dt.datetime(2026,10,16,16,12,18).astimezone().timestamp()
rows=[]
for path in CACHE.rglob('_ensure.json'):
    modified=path.stat().st_mtime
    if start<=modified<=end:
        data=json.loads(path.read_text())
        rows.append(dict(path=path.relative_to(CACHE).as_posix(),kernel=data['name'],
            modified_utc=dt.datetime.fromtimestamp(modified,dt.timezone.utc).isoformat(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len(rows)==48
probe=RUN/'candidate3-wheel-performance-off-v1.json'
data=json.loads(probe.read_text(encoding='utf-8'))
row=next(row for row in data['runs'] if row['name']=='code-1')
assert row['ttft_ms']>12000
result=dict(scope='post_checkpoint_cache_timestamp_and_frozen_source_analysis_not_gpu_trace',
    probe=str(probe),probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
    request=dict(name=row['name'],ttft_ms=row['ttft_ms'],elapsed_seconds=row['elapsed_seconds']),
    source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    compiled_variants=rows,compiled_count=len(rows),
    observation='48 _ensure variants appeared during12-second TTFT of first6-token cached-prefix prefill.',
    source_mechanism='BASE is tl.constexpr and passed layer_id*num_experts; K is also constexpr, so a new route count specializes48 layers.',
    conclusion='Strong evidence for per-layer cold JIT as main cause; not a captured compile-stack trace.',
    next_step='Prototype runtime layer offset with graph/numerical/owner tests and fresh-cache compile counts; retain perf3 unchanged.')
target=RUN/'jit-spike-analysis-v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(variants=len(rows),ttft_ms=row['ttft_ms'])))
