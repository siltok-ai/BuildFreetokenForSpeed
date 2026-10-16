"""Verify completed native screens and retain their scope before archiving."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).parent
STRATA = Path(r'C:\workspace\runs\strata\flash-perf-20261016')
KEYS = ('english','chinese','code','70-0','diverse-4096-0')
REPORTS = {
    'cap2_full_head': 'mtp-cap2-fivekeys-v1.json',
    'cap3_copied_subset': 'draft-subset-three-fivekeys-v1.json',
}


def read(path):
    """Require terminal successful evidence, not a partial log."""
    report = json.loads(path.read_text(encoding='utf-8-sig'))
    if report['status']!='passed':raise RuntimeError(f'Unfinished evidence: {path}')
    return report


def medians(report,keys):
    """Only use three complete measured outputs per workload."""
    output={}
    for key in keys:
        rows=[r for r in report['runs'] if r['name'] in [f'{key}-{i}' for i in range(3)]]
        if len(rows)!=3 or any(r['output_tokens']!=512 for r in rows):
            raise RuntimeError(f'Incomplete 3x512 workload: {key}')
        output[key]=dict(median_tps=statistics.median(r['decode_tps'] for r in rows),
                         individual_tps=[r['decode_tps'] for r in rows],
                         input_tokens=[r['input_tokens'] for r in rows],
                         mtp=[r.get('mtp',r.get('done')) for r in rows])
    return output


result=dict(scope='external_process_prototypes_on_frozen_perf3; not a release; numerical trajectories can differ',
            performance={},quality={},strata={},components={})
for label,name in REPORTS.items():
    report=read(ROOT/name)
    resource=json.loads((ROOT/name.replace('.json','.resource.json')).read_text(encoding='utf-8-sig'))
    if resource['returncode']!=0 or resource['stop_reason'] is not None or resource['remaining_descendants']:
        raise RuntimeError(f'Unsafe resource result: {name}')
    result['performance'][label]=dict(report=name,results=medians(report,KEYS),
        resource=dict(min_gpu_free=resource['min_gpu_free'],min_commit_available=resource['min_commit_available']))
for label,name in [('cap2','mtp-cap2-quality-on-v1.json'),('copied_subset','draft-subset-quality-on-v1.json'),('mapped_subset','draft-mapped-quality-on-v2.json')]:
    report=read(ROOT/name)
    if report['score']!=15 or report['thinking_score']!=3 or report['regressions'] or len(report['rebuild_steps'])!=2:
        raise RuntimeError(f'Unexpected quality gate: {name}')
    result['quality'][label]=dict(report=name,score=report['score'],thinking_score=report['thinking_score'],
        regressions=report['regressions'],rebuild=report['rebuild_steps'],draft_subset_check=report.get('draft_subset_check'))
for label,name in [('full_head','reference-mtp-fullhead-3x-v1.json')]:
    report=read(STRATA/name)
    if report['binary_sha256']!='9832fcaffd5fc389963bf391ad2df5700d43399da7a85f9fcf42d65d82c58f76':
        raise RuntimeError('Strata binary changed')
    result['strata'][label]=dict(report=name,results=medians(report,KEYS[:3]),command=report['command'],
        draft_full_head_env=report['draft_full_head_env'])
for name in ('gather-aligned-components-v1.json','mapped-head-components-v1.json'):
    result['components'][name]=read(ROOT/name)
output=ROOT/'perf4-screening-summary-v1.json'
if output.exists():raise FileExistsError(output)
output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:{k:v['median_tps'] for k,v in value['results'].items()}
                  for key,value in result['performance'].items()},indent=2))

