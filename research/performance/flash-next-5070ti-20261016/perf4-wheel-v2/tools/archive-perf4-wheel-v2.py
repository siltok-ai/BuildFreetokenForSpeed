"""封存已完成的perf4普通wheel证据，性能/采样/源码边界分别标记。"""
from pathlib import Path
import hashlib
import json
import shutil

RUN=Path(__file__).parent
REPO=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
DEST=REPO/'research/performance/flash-next-5070ti-20261016/perf4-wheel-v2'
if DEST.exists():raise FileExistsError(DEST)
DEST.mkdir(parents=True)
files=[]


def copy(source,folder):
    """逐字节复制完成的证据，生成可独立复核的SHA索引。"""
    target=DEST/folder/source.name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    files.append(dict(path=target.relative_to(DEST).as_posix(),bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest()))


for stem in ('candidate4-source-components-v1','candidate4-wheel-components-v1',
    'candidate4-source-quality-off-v1','candidate4-source-quality-on-r4608-v1',
    'candidate4-wheel-quality-off-v1','candidate4-wheel-quality-on-v1',
    'candidate4-wheel-performance-off-v1','candidate4-wheel-performance-on-v1',
    'candidate4-early-r5376-off-fivekeys-v1','host-embedding-decay07-off-fivekeys-v1',
    'host-embedding-decay07-off-quality-v1'):
    resource=json.loads((RUN/(stem+'.resource.json')).read_text())
    assert resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants']
    for suffix in (('.resource.json','.stdout.log','.stderr.log','.samples.jsonl') if 'components' in stem else ('.json','.resource.json','.stdout.log','.stderr.log','.samples.jsonl')):
        copy(RUN/(stem+suffix),'runtime')
for name in ('perf4-native-summary-v1.json','summarize-perf4-v1.py',
    'off-cache-timeline-comparison-v1.json','analyze-off-cache-timelines-v1.py',
    'nsys-off-cache-comparison-v1.json','analyze-off-cache-nsys-v1.py',
    'nsys-off-late-comparison-v2.json','analyze-off-late-nsys-v2.py'):
    copy(RUN/name,'analysis')
for name in ('probe27-wheel4.py','probe25-source4-timing.py','test-candidate4-wheel-v1.py',
    'test-candidate4-source-v1.py','prepare-cand4-env.py','freeze_candidate4.py',
    'run-perf4-measurements-v1.py','dxgi_memory_budget.py','supervise-v2.py'):
    copy(RUN/('consumer-tools' if name=='supervise-v2.py' else '')/name,'tools')
for name in ('build-perf4-validator-v1.log','build-perf4-tools-tests-v1.log'):
    copy(RUN/name,'checks')
copy(Path(__file__),'tools')
large=[]
for pattern in ('nsys-off-host-r*.nsys-rep','nsys-off-late-r*.nsys-rep'):
    for path in sorted(RUN.glob(pattern)):
        large.append(dict(local_path=str(path),bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
summary=json.loads((RUN/'perf4-native-summary-v1.json').read_text(encoding='utf-8'))
index=dict(scope='Frozen perf4 ordinary wheel, not Speed HTTP or installed acceptance',
    version=summary['version'],wheel_sha256='3b83b9fc79e8427663b775f27516d9acb13b55403b6b9fce0c202b90b363c5bc',
    decision='Keep early mapped original embedding, 32-round heat decay0.7, runtime BASE reuse. OFF7424 slots reaches same-Q2 Strata in EN/CN/code native tests; ON6656 improves but remains15-23% below Strata. Continue MTP exploration.',
    files=files,local_large_diagnostics=large)
(DEST/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for row in files:assert hashlib.sha256((DEST/row['path']).read_bytes()).hexdigest()==row['sha256']
print(json.dumps(dict(directory=str(DEST),verified_files=len(files))))
