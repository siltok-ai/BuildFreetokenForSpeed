"""Summarize immutable process-only experiments without mixing recipes or protocols."""
from pathlib import Path
from collections import defaultdict
from statistics import median
import json

ROOT=Path(__file__).resolve().parent
NAMES=('english','chinese','code','70-0','diverse-4096-0')


def performance(name):
    report=json.loads((ROOT/name).read_text(encoding='utf-8'))
    groups=defaultdict(list)
    for row in report['runs']:
        for key in NAMES:
            if row['name'].rsplit('-',1)[0]==key:
                groups[key].append(row)
    return dict(report=name,status=report['status'],shutdown_complete=report.get('shutdown_complete'),
        args=report['args'],medians={key:median(r['decode_tps'] for r in rows) for key,rows in groups.items()},
        measured_counts={key:len(rows) for key,rows in groups.items()},
        actual_output_lengths={key:[r['output_tokens'] for r in rows] for key,rows in groups.items()},
        resource=json.loads((ROOT/name.replace('.json','.resource.json')).read_text()))


def main():
    result=dict(scope='Process-only prototypes; no new frozen release or Speed HTTP claim',
        reports=[performance(name) for name in (
            'candidate3-wheel-performance-on-v1.json','bf16-four-down-fivekeys-v1.json')])
    for name in ('host-embedding-quality-on-v1.json','prefill2048-r4096-quality-on-v1.json'):
        p=ROOT/name
        if p.exists():
            r=json.loads(p.read_text(encoding='utf-8'))
            result.setdefault('quality',[]).append(dict(report=name,status=r['status'],
                score=r.get('score'),thinking=r.get('thinking_score'),regressions=r.get('regressions'),
                mapped_records=r.get('host_embedding'),rebuild=r.get('rebuild_steps')))
    out=ROOT/'window-host-screen-summary-v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({r['report']:r['medians'] for r in result['reports']}))


if __name__=='__main__':main()
