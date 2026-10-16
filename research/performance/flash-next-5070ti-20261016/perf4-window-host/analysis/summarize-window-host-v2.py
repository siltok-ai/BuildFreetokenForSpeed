"""Compare complete measurements, preserving each cache recipe and actual output IDs."""
from pathlib import Path
from statistics import median
import json

ROOT=Path(__file__).resolve().parent
KEYS=('english','chinese','code','70-0','diverse-4096-0')


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))


def summarize(stem):
    r=read(stem+'.json'); resource=read(stem+'.resource.json')
    assert r['status']=='passed' and r['shutdown_complete']
    assert resource['returncode']==0 and not resource['stop_reason'] and not resource['remaining_descendants']
    groups={key:[row for row in r['runs'] if row['name'].rsplit('-',1)[0]==key] for key in KEYS}
    assert all(len(rows)==3 and all(row['output_tokens']==512 for row in rows) for rows in groups.values())
    return dict(report=stem+'.json',args=r['args'],
        medians={key:median(row['decode_tps'] for row in rows) for key,rows in groups.items()},
        first_ttft_ms={key:rows[0]['ttft_ms'] for key,rows in groups.items()},
        first_total_s={key:rows[0]['elapsed_seconds'] for key,rows in groups.items()},
        min_gpu_free=resource['min_gpu_free'],min_commit_available=resource['min_commit_available'],
        host_embedding=r.get('host_embedding'),resource_exit=resource['returncode'])


def compare(a,b):
    left={row['name']:row for row in read(a+'.json')['runs']}
    right={row['name']:row for row in read(b+'.json')['runs']}
    shared=sorted(left.keys()&right.keys())
    differences=[name for name in shared if left[name]['output']['token_ids']!=right[name]['output']['token_ids']]
    return dict(left=a,right=b,shared=len(shared),exact=len(shared)-len(differences),different=differences)


def main():
    stems=('candidate3-wheel-performance-on-v1','bf16-four-down-fivekeys-v1',
        'host-embedding-r4352-fivekeys-v1','host-embedding-r4608-fivekeys-v1')
    result=dict(scope='Process-only experiments on frozen perf3 ordinary wheel; no new release or Speed HTTP claim',
        order=list(stems),reports={stem:summarize(stem) for stem in stems},
        comparisons=[compare('candidate3-wheel-performance-on-v1',stem) for stem in stems[1:]]+
            [compare('admission-runtime-base-quality-on-v1','host-embedding-quality-on-v1')])
    p=ROOT/'window-host-screen-summary-v2.json'
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({stem:result['reports'][stem]['medians'] for stem in stems}))
    print(json.dumps(result['comparisons']))


if __name__=='__main__':main()
