"""Preserve ordinary wheel acceptance and exact source/consumer boundary evidence."""
from pathlib import Path
import hashlib,json,shutil,statistics
R=Path(__file__).parent
D=Path(r'C:\workspace\wt\freetoken-mats\flash-perf\research\performance\flash-next-5070ti-20261016\perf6-wheel-v1')
if D.exists():raise FileExistsError(D)
D.mkdir(parents=True)
files=[]
def copy(p,folder):
    t=D/folder/p.name;t.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,t)
    files.append(dict(path=t.relative_to(D).as_posix(),bytes=t.stat().st_size,sha256=hashlib.sha256(t.read_bytes()).hexdigest()))
summary={'scope':'Ordinary installed FreeToken wheel; Speed remains separate acceptance',
 'version':'0.1.3+speed.strataq2perf6','wheel_sha256':'fcd0d54792f9a4541cdb9ffcd7a2ca3f4a686fe3dfb1a78baac295ae1038f634','modes':{}}
for mode in ('off','on'):
    facts={}
    for suite in ('quality','performance'):
        stem=f'perf6-wheel-{suite}-{mode}-v1'
        resource=json.loads((R/(stem+'.resource.json')).read_text())
        assert resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants']
        current=json.loads((R/(stem+'.json')).read_text())
        assert current['status']=='passed' and current['shutdown_complete']
        old=json.loads((R/f'source7-{suite}-{mode}-v1.json').read_text())
        previous={row['name']:row['output']['token_ids'] for row in old['runs']}
        ids=sum(row['output']['token_ids']==previous[row['name']] for row in current['runs'])
        assert ids==len(current['runs'])
        facts[suite]=dict(ids_equal_source7=ids,outputs=len(current['runs']),min_gpu_free=resource['min_gpu_free'])
        if suite=='quality':
            assert (current['score'],current['thinking_score'],current['regressions'])==(15,3,[])
            facts[suite].update(score=15,thinking=3,rebuild_steps=current['rebuild_steps'])
        else:
            facts[suite]['medians']={key:statistics.median(row['decode_tps'] for row in current['runs'] if row['name'].startswith(key+'-')) for key in ['english','chinese','code','70-0','diverse-4096-0']}
        for suffix in ('.json','.resource.json','.stdout.log','.stderr.log','.samples.jsonl'):copy(R/(stem+suffix),'runtime')
    summary['modes'][mode]=facts
for stem in ('perf6-wheel-components-v1',):
    for suffix in ('.resource.json','.stdout.log','.stderr.log','.samples.jsonl'):copy(R/(stem+suffix),'runtime')
for name in ('probe63-wheel6.py','test-wheel6-components-v1.py','run-wheel6-v1.py','prepare-cand6-env.py',
 'freeze_candidate6.py','probe62-source7.py','run-source7-v1.py','run-freeze6-materials-v1.py','memory_inventory_v2.py'):
    copy(R/name,'tools')
copy(R/'consumer-tools/supervise-v2.py','tools')
for name in ('build-perf6-validator-v1.log','build-perf6-tools-v1.log','build-perf6-offline-ok-v1.json'):copy(R/name,'checks')
for mode in ('on','off'):
    for suite in ('quality','performance'):
        for suffix in ('.json','.resource.json','.stdout.log','.stderr.log','.samples.jsonl'):
            copy(R/f'source7-{suite}-{mode}-v1{suffix}','source-runtime')
for suffix in ('.resource.json','.stdout.log','.stderr.log','.samples.jsonl'):
    copy(R/('source7-components-v2'+suffix),'source-runtime')
for phase in (R/'source7-patch-stages-v1').rglob('*'):
    if phase.is_file():copy(phase,Path('patch-stages')/phase.relative_to(R/'source7-patch-stages-v1').parent)
for name in ('nsys-lut-ple-comparison-v1.json','analyze-lut-ple-nsys-v1.py'):
    if (R/name).exists():copy(R/name,'analysis')
(D/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
copy(D/'summary.json','analysis')
copy(Path(__file__),'tools')
(D/'index.json').write_text(json.dumps({'scope':summary['scope'],'files':files},indent=2)+'\n',encoding='utf-8')
for row in files:assert hashlib.sha256((D/row['path']).read_bytes()).hexdigest()==row['sha256']
print(json.dumps(summary,indent=2));print('Archived',len(files),'files')
