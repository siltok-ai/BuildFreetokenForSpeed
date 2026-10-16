"""Preserve completed source experiments before selecting the next material profile."""
from pathlib import Path
import hashlib
import json
import shutil

RUN=Path(__file__).parent
REPO=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
DEST=REPO/'research/performance/flash-next-5070ti-20261016/q2-execution-v1'
if DEST.exists():raise FileExistsError(DEST)
DEST.mkdir(parents=True)
files=[]


def copy(source,relative):
    target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    files.append(dict(path=relative,bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest()))


stems=[
    'native-grouped-q2-components-v2','native-grouped-q2-components-v3',
    'native-grouped-q2-quality-on-v1','native-grouped-q2-performance-on-v1',
    'native-gu-only-quality-on-v1','native-gu-only-performance-on-v1',
    'hybrid-q2-components-v2','hybrid-q2-components-v3','hybrid-q2-components-v4',
    'hybrid-q2-quality-on-v1','hybrid-q2-performance-on-v1','hybrid-q2-gpu-mask-three-language-v1',
    'candidate5-source-components-v1','candidate5-source-quality-on-v1',
    'candidate5-source-quality-off-v1','candidate5-source-performance-on-v1','candidate5-source-performance-off-v1',
    'q2-lut-exhaustive-v1','q2-lut-components-v1','q2-lut-quality-on-v1','q2-lut-quality-off-v1',
    'q2-lut-performance-on-v1','q2-lut-performance-off-v1',
    'cached-adaptation-components-v1','cached-lut-quality-on-v1','cached-lut-quality-off-v1',
    'cached-lut-performance-on-v1','cached-lut-performance-off-v1',
    'cached-lut-d0-e8-m48-quality-on-v1','cached-lut-d0-e8-m48-performance-on-v1',
    'q2-lut-tiles-grouped-components-v1']
missing=[]
for stem in stems:
    resource_path=RUN/(stem+'.resource.json')
    if not resource_path.exists():missing.append(stem);continue
    resource=json.loads(resource_path.read_text())
    assert resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants'],stem
    for suffix in ('.json','.resource.json','.stdout.log','.stderr.log','.samples.jsonl'):
        p=RUN/(stem+suffix)
        if p.exists():copy(p,'runtime/'+p.name)
for pattern in ('native_grouped_q2_*.*','native_grouped_q2_gu_variant*','hybrid_q2_*.*',
    'q2_lut_variant_v1.py','q2_cached_*v1.*','q2_grouped_lut_variant_v1.py',
    'bench-*q2*py','test-q2-lut*py','test-cached-adaptation-v1.py',
    'probe3[0-5]*.py','prepare-cached-adaptation-v1.py','run-cached-lut-combined-v1.py',
    'bench-q2-reduction-order-v1.json'):
    for p in sorted(RUN.glob(pattern)):
        relative='tools/'+p.name
        if any(x['path']==relative for x in files):continue
        if p.suffix in ('.py','.cuh','.cpp','.h','.json'):copy(p,relative)
for folder in ('candidate5-working-source-v1','strata-cpu-reference-v1'):
    for p in sorted((RUN/folder).rglob('*')):
        if p.is_file():copy(p,'snapshots/'+folder+'/'+p.relative_to(RUN/folder).as_posix())
copy(RUN/'consumer-tools/supervise-v2.py','tools/supervise-v2.py')
copy(Path(__file__),'tools/'+Path(__file__).name)
index=dict(scope='Research sources and raw completed experiments, not a delivery profile or Speed acceptance',
    base_frozen_material='0.1.3+speed.strataq2perf4',source5_snapshot='snapshots/candidate5-working-source-v1/index.json',
    decision='Keep signed-Q2 LUT and GPU owner metadata publication as next-profile candidates; cached incoming reuse modest. CPU hybrid and native grouped variants are not adopted. D0/e8/max48 not globally better. Source5 version label stillperf4; external prototypes explicitly identified.',
    numerical_boundary='Original Q2 bytes/all experts/FP32 activation scales/Q8 codes retained; LUT changes FP32 reduction layout and some output IDs. Both modes retain15/20thinking3/3, no added failures, grow/shrink pass.',
    native_median_tps=dict(keys=['english','chinese','code','short','4k'],
        off=[96.921,97.454,93.879,96.271,92.267],on=[104.824,98.210,110.539,95.611,111.279]),
    absent_optional_stems=missing,files=files)
(DEST/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in files:assert hashlib.sha256((DEST/row['path']).read_bytes()).hexdigest()==row['sha256']
print(json.dumps(dict(directory=str(DEST),files=len(files),missing=missing)))
