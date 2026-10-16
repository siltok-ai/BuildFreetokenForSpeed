"""Compare unprofiled token intervals and exact cache-miss counters at both capacities."""
from pathlib import Path
from statistics import median,quantiles
import json

ROOT=Path(__file__).resolve().parent
STEMS=('off-cache-r5120-timing-v1','off-cache-r5376-timing-v1')
KEYS=('english','chinese','code','70-0','diverse-4096-0')


def analyze(stem):
    r=json.loads((ROOT/(stem+'.json')).read_text(encoding='utf-8'))
    resource=json.loads((ROOT/(stem+'.resource.json')).read_text(encoding='utf-8'))
    assert r['status']=='passed' and r['args']['stats'] and not r['args']['trace']
    assert resource['returncode']==0 and not resource['stop_reason'] and not resource['remaining_descendants']
    rows=[]
    for row in r['runs']:
        if row['name'].rsplit('-',1)[0] not in KEYS:continue
        offsets=row['arrival_offsets_s']
        assert len(offsets)==row['output_tokens']==512
        intervals=[1000*(b-a) for a,b in zip(offsets,offsets[1:])]
        qs=quantiles(intervals,n=100,method='inclusive')
        rows.append(dict(name=row['name'],decode_tps=row['decode_tps'],
            interval_ms=dict(mean=sum(intervals)/len(intervals),p50=median(intervals),p95=qs[94],p99=qs[98],
                max=max(intervals),over25=sum(v>25 for v in intervals),over50=sum(v>50 for v in intervals),
                over100=sum(v>100 for v in intervals)),miss_stats=row['miss_stats'],
            chunks64=[dict(first_interval=i,mean_ms=sum(intervals[i:i+64])/len(intervals[i:i+64])) for i in range(0,len(intervals),64)]))
    return dict(args=r['args'],medians={key:median(row['decode_tps'] for row in rows if row['name'].rsplit('-',1)[0]==key) for key in KEYS},
        rows=rows,min_gpu_free=resource['min_gpu_free'],input_ids=r['input_ids'],
        original_outputs={row['name']:row['output']['token_ids'] for row in r['runs']})


data={stem:analyze(stem) for stem in STEMS}
left,right=(data[stem] for stem in STEMS)
assert left['input_ids']==right['input_ids']
assert left['original_outputs']==right['original_outputs']
for r in data.values():r.pop('original_outputs');r.pop('input_ids')
result=dict(scope='No external profiler; same stats-enabled admission on both, normal arithmetic; final throughput needs stats-off acceptance',
    exact_output_trajectories=True,reports=data)
target=ROOT/'off-cache-timeline-comparison-v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
for stem,r in data.items():
    print(json.dumps(dict(stem=stem,medians=r['medians'],selected=[row for row in r['rows'] if row['name'] in ('code-1','70-0-1','diverse-4096-0-1')])))
