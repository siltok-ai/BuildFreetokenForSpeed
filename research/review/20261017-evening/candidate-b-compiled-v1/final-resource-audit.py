"""Audit every final source/performance/embedded run, including non-primary ABBA observations."""
from pathlib import Path
import json
R=Path(__file__).parent
paths=[]
for pattern in ('candidate-b-source-*-v2.resource.json','*-http-*-abba*-v3.resource.json'):
    paths.extend(R.glob(pattern))
paths.extend((R/'staged-review2b/results').glob('*.resource.json'))
assert len(paths)==20, len(paths)
rows=[]
for path in sorted(paths):
    data=json.loads(path.read_text(encoding='utf-8'))
    assert data['returncode']==0 and data['stop_reason'] is None and not data['remaining_descendants'],path
    assert data['min_gpu_free']>=256*2**20 and data['min_ram_available']>=2*2**30 and data['min_commit_available']>=2**30,path
    rows.append(dict(report=str(path),min_gpu_free=data['min_gpu_free'],
                     min_ram_available=data['min_ram_available'],min_commit_available=data['min_commit_available']))
result=dict(status='passed',scope='all20 final source/ABBA/compiled process runs; sampled floors, not instantaneous guarantee',
    runs=len(rows),minimum_gpu_free=min(row['min_gpu_free'] for row in rows),
    minimum_ram_available=min(row['min_ram_available'] for row in rows),
    minimum_commit_available=min(row['min_commit_available'] for row in rows),reports=rows)
target=R/'final-resource-summary-v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:value for key,value in result.items() if key!='reports'}))
