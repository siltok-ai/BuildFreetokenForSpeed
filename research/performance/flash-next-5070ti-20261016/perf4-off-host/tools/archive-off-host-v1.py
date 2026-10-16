"""Archive the first-objective no-MTP cache comparison separately from MTP measurements."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).resolve().parent
DEST=Path(r'C:\workspace\wt\freetoken-mats\flash-perf\research\performance\flash-next-5070ti-20261016\perf4-off-host')
if DEST.exists():raise FileExistsError(DEST)
DEST.mkdir(parents=True)
files=[]


def copy(name,folder):
    target=DEST/folder/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/name,target)
    files.append(dict(path=target.relative_to(DEST).as_posix(),bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),source=str(ROOT/name)))


for stem in ('host-embedding-r5376-fivekeys-off-v1','host-embedding-r5120-fivekeys-off-v1'):
    report=json.loads((ROOT/(stem+'.json')).read_text(encoding='utf-8-sig'))
    resource=json.loads((ROOT/(stem+'.resource.json')).read_text(encoding='utf-8-sig'))
    assert report['status']=='passed' and report['shutdown_complete']
    assert resource['returncode']==0 and not resource['stop_reason'] and not resource['remaining_descendants']
    for suffix in ('.json','.resource.json','.stdout.log','.stderr.log','.samples.jsonl'):
        copy(stem+suffix,'native')
for name in ('summarize-off-host-v1.py','off-host-comparison-v1.json'):
    copy(name,'analysis')
copy('archive-off-host-v1.py','tools')
index=dict(scope='No-MTP first objective; process-only host mapping/cache comparison, not a new release',
    source_checkpoint='18f3f31e20306e2122715482d92cba70a2280dd4',
    decision='Keep mapped embedding direction; do not adopt7424-slot preset with late workload slowdown. Root cause not confirmed by passing resource floor.',
    priority='First approach Strata no-MTP performance, second obtain clear MTP uplift; neither objective removed.',
    files=files)
(DEST/'index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
for row in files:assert hashlib.sha256((DEST/row['path']).read_bytes()).hexdigest()==row['sha256']
print(json.dumps(dict(directory=str(DEST),verified_files=len(files))))
