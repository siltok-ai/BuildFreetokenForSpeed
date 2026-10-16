"""Keep no-MTP and MTP objectives separate; compare full no-MTP trajectories and resources."""
from pathlib import Path
from statistics import median
import json

ROOT=Path(__file__).resolve().parent
KEYS=('english','chinese','code','70-0','diverse-4096-0')
STEMS=('candidate3-wheel-performance-off-v1','host-embedding-r5120-fivekeys-off-v1',
       'host-embedding-r5376-fivekeys-off-v1')


def read(stem):return json.loads((ROOT/(stem+'.json')).read_text(encoding='utf-8-sig'))


def main():
    base={row['name']:row for row in read(STEMS[0])['runs']}
    records={}
    for stem in STEMS:
        r=read(stem)
        resource=read(stem+'.resource')
        assert r['status']=='passed' and r['shutdown_complete']
        assert resource['returncode']==0 and not resource['stop_reason'] and not resource['remaining_descendants']
        groups={key:[row for row in r['runs'] if row['name'].rsplit('-',1)[0]==key] for key in KEYS}
        assert all(len(rows)==3 and all(row['output_tokens']==512 for row in rows) for rows in groups.values())
        shared=[row for row in r['runs'] if row['name'] in base]
        diff=[row['name'] for row in shared if row['output']['token_ids']!=base[row['name']]['output']['token_ids']]
        records[stem]=dict(args=r['args'],
            medians={key:median(row['decode_tps'] for row in rows) for key,rows in groups.items()},
            first_ttft_ms={key:rows[0]['ttft_ms'] for key,rows in groups.items()},
            first_total_s={key:rows[0]['elapsed_seconds'] for key,rows in groups.items()},
            min_gpu_free=resource['min_gpu_free'],min_commit_available=resource['min_commit_available'],
            exact_ids=len(shared)-len(diff),shared_outputs=len(shared),different_outputs=diff,
            adaptation_ms=sum(row['wall_ms'] for row in r.get('adaptive_rounds',[])))
    q=read('host-embedding-r5376-quality-off-v1')
    result=dict(scope='MTP OFF process-only prototype; preserve frozen perf3. Do not assume larger cache universally faster.',
        reports=records,off_quality=dict(score=q['score'],thinking=q['thinking_score'],regressions=q['regressions']))
    target=ROOT/'off-host-comparison-v1.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({stem:record['medians'] for stem,record in records.items()}))
    print(json.dumps({stem:{k:record[k] for k in ('exact_ids','shared_outputs','min_gpu_free','adaptation_ms')} for stem,record in records.items()}))


if __name__=='__main__':main()
