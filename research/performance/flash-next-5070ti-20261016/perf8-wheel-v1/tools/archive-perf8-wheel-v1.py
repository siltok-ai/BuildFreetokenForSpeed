"""Freeze actual wheel8 results after every supervised run completes successfully."""
from pathlib import Path
import hashlib,json,shutil,statistics
R=Path(__file__).parent
B=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
D=B/'research/performance/flash-next-5070ti-20261016/perf8-wheel-v1'

def main():
    summaries={}
    stems=[]
    for mode in ('off','on','on-full'):
        summaries[mode]={}
        for suite in ('quality','performance'):
            stem=f'perf8-wheel-{suite}-{mode}-v1';stems.append(stem)
            data=json.loads((R/(stem+'.json')).read_text())
            resource=json.loads((R/(stem+'.resource.json')).read_text())
            assert data['status']=='passed' and data['shutdown_complete']
            assert resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants']
            assert resource['min_gpu_free']>=256*2**20 and resource['min_ram_available']>=2*2**30 and resource['min_commit_available']>=2**30
            if suite=='quality':
                assert (data['score'],data['thinking_score'],data['regressions'])==(15,3,[])
                assert len(data['rebuild_steps'])==2
                facts=dict(score=15,thinking=3,rebuild_steps=data['rebuild_steps'])
            else:
                facts={k:statistics.median(x['decode_tps'] for x in data['runs'] if x['name'].startswith(k+'-')) for k in ('english','chinese','code','70-0','diverse-4096-0')}
            summaries[mode][suite]=facts
    for stem in ('perf8-wheel-components-v1','perf8-wheel-mtp-components-v1'):
        stems.append(stem)
        d=json.loads((R/(stem+'.resource.json')).read_text())
        assert d['returncode']==0 and d['stop_reason'] is None and not d['remaining_descendants']
    assert not D.exists(),D
    D.mkdir(parents=True)
    def copy(p,rel):
        t=D/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    for stem in stems:
        for suffix in ('.json','.resource.json','.stdout.log','.stderr.log','.samples.jsonl'):
            p=R/(stem+suffix)
            if p.exists():copy(p,'runtime/'+p.name)
    for name in ('probe66-wheel8.py','test-wheel8-components-v1.py','test-wheel8-mtp-v1.py','run-wheel8-v1.py','freeze_candidate8.py','freeze_candidate7.py','prepare-source8-draft-v1.py','archive-perf8-wheel-v1.py'):
        p=R/name
        if p.exists():copy(p,'tools/'+name)
    copy(R/'consumer-tools/supervise-v2.py','tools/supervise-v2.py')
    for name in ('build-perf8-validator-v1.log','build-perf8-tools-v1.log','build-perf8-cpu-ok-v1.json'):
        copy(R/name,'checks/'+name)
    for name in ('perf7-poison-before-v1.log','source8-policy-cpu-tests-v1.log','source8-quality-on-v3.json','source8-quality-on-v3.resource.json','source8-performance-on-v2.json','source8-performance-on-v2.resource.json'):
        copy(R/name,'development/'+name)
    (D/'summary.json').write_text(json.dumps(dict(scope='Ordinary installed FreeToken wheel; Speed remains separate acceptance',version='0.1.3+speed.strataq2perf8',modes=summaries,development_note='FT7 real poison-entry regression is preserved as failed diagnostic, not accepted runtime.'),indent=2)+'\n')
    files={p.relative_to(D).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(D.rglob('*')) if p.is_file()}
    (D/'index.json').write_text(json.dumps(dict(scope='Ordinary installed FreeToken wheel; Speed remains separate acceptance',files=files),indent=2)+'\n')
    print(json.dumps(dict(archived=len(files),medians=summaries),indent=2))

if __name__=='__main__':main()
