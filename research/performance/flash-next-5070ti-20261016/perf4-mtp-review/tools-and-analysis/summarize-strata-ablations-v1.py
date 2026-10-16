"""Cross-check same-binary, fixed-cache reverse ablations and actual token delivery."""
from pathlib import Path
import hashlib
import json
import statistics

ROOT = Path(__file__).parent
RUN = Path(r'C:\workspace\runs\strata\flash-perf-20261016')
NAMES = {'cjk':'reference-mtp-cjk-fixed-3x-v1','english_subset':'reference-mtp-en-fixed-3x-v1',
         'full_head':'reference-mtp-fullhead-3x-v1','synchronous_commit':'reference-mtp-commitsync-fixed-3x-v1'}
reports={}
summaries={}
for label,name in NAMES.items():
    path=RUN/(name+'.json')
    report=json.loads(path.read_text(encoding='utf-8-sig'))
    resource=json.loads((RUN/(name+'.resource.json')).read_text(encoding='utf-8-sig'))
    if report['status']!='passed' or report['returncode']!=0 or not report['shutdown_complete']:
        raise RuntimeError(f'Incomplete ablation: {name}')
    if resource['returncode']!=0 or resource['stop_reason'] or resource['remaining_descendants']:
        raise RuntimeError(f'Unsafe resource result: {name}')
    command=report['command']
    if command[command.index('--expert-cache')+1]!='7019' or command[command.index('--suffix-draft')+1]!='0':
        raise RuntimeError('Cache or lookup confounds the ablation')
    if report['binary_sha256']!='9832fcaffd5fc389963bf391ad2df5700d43399da7a85f9fcf42d65d82c58f76':
        raise RuntimeError('Reference executable changed')
    results={}
    for key in ('english','chinese','code'):
        rows=[r for r in report['runs'] if r['name'] in [f'{key}-{i}' for i in range(3)]]
        if len(rows)!=3 or any(r['output_tokens']!=512 or len(r['arrival_offsets'])!=512 for r in rows):
            raise RuntimeError('Actual delivery incomplete')
        for row in rows:
            tps=511/(row['arrival_offsets'][-1]-row['arrival_offsets'][0])
            if abs(tps-row['decode_tps'])>1e-8:raise RuntimeError('TPS does not match actual timestamps')
        results[key]=dict(median_tps=statistics.median(r['decode_tps'] for r in rows),
            individual_tps=[r['decode_tps'] for r in rows],
            accepted_and_offered=[dict(accepted=int(r['done'][7]),offered=int(r['done'][8])) for r in rows])
    reports[label]=report
    summaries[label]=dict(report=name+'.json',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        results=results,min_gpu_free=resource['min_gpu_free'],min_ram_available=resource['min_ram_available'])
base=reports['cjk']
for label,report in reports.items():
    if report['input_ids']!=base['input_ids']:raise RuntimeError('Prompts changed')
    for key,values in summaries[label]['results'].items():
        values['percent_vs_cjk']=(values['median_tps']/summaries['cjk']['results'][key]['median_tps']-1)*100
    summaries[label]['identical_output_runs_to_cjk']=sum(
        next(b for b in base['runs'] if b['name']==row['name'])['token_ids']==row['token_ids']
        for row in report['runs'] if row['name']!='warmup')
if reports['full_head']['draft_full_head_env']!='1':raise RuntimeError('Full-head switch absent')
if reports['synchronous_commit']['commit_sync_env']!='1':raise RuntimeError('Synchronous-commit switch absent')
if reports['english_subset']['draft_vocab_sha256']!='369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25':
    raise RuntimeError('English vocabulary changed')
result=dict(scope='fixed unmodified Strata; full original Q2; 7019 slots; MTP only; 3x512 each; fresh engine per mode',
    caveat='sequential fresh processes; draft choices change acceptance and expert routing; no universal speedup claim',
    ablations=summaries)
out=ROOT/'strata-reverse-ablations-summary-v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summaries,indent=2))
