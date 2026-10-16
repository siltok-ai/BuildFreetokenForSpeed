from pathlib import Path
import json,runpy,subprocess,shutil
q=runpy.run_path(str(Path(__file__).with_name('run-cost-and-tiles-v1.py')))
supervise,RUN=(q[k] for k in ('supervise','RUN'))
PY=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand5\Scripts\python.exe')
ROOT=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
MAT=Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-06a')

def command(args,log,cwd=ROOT):
    with (RUN/log).open('wb') as f:subprocess.run([str(PY),'-B',*map(str,args)],cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)

def main():
    for mode in ('off','on'):
        for suite in ('quality','performance'):
            d=json.loads((RUN/f'source7-{suite}-{mode}-v1.json').read_text())
            r=json.loads((RUN/f'source7-{suite}-{mode}-v1.resource.json').read_text())
            old=json.loads((RUN/f'perf5-wheel-{suite}-{mode}-v1.json').read_text())
            previous={x['name']:x['output']['token_ids'] for x in old['runs']}
            assert d['status']=='passed' and r['returncode']==0 and not r['remaining_descendants']
            assert all(x['output']['token_ids']==previous[x['name']] for x in d['runs'])
    supervise('source7-components-v2',[str(PY),'-B','test-source7-components-v1.py'])
    command([RUN/'freeze_candidate6.py'],'freeze-perf6-v1.log',RUN)
    command([ROOT/'build/derive_python_materials.py','--profile',ROOT/'build/profiles/strataq2perf6.json',
        '--base-profile',ROOT/'build/profiles/strataq2perf5.json','--base-materials',ROOT/'artifacts/strataq2perf5-cp312-cu130-sm120',
        '--output',MAT,'--rebuild-pinned-tensor'],'derive-perf6-v1.log')
    record=json.loads((MAT/'build-record.json').read_text())
    dest=ROOT/'artifacts/strataq2perf6-cp312-cu130-sm120';dest.mkdir()
    for name in ['build-record.json',*record['artifacts']]:
        target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(MAT/name,target)
    command([ROOT/'tools/validate_repository.py'],'build-perf6-validator-v1.log')
    command(['-m','unittest','discover','-s','tests','-v'],'build-perf6-tools-v1.log')
    (RUN/'build-perf6-offline-ok-v1.json').write_text(json.dumps({'status':'passed','version':record['custom_version'],
        'wheel_sha256':record['wheel_sha256'],'scope':'offline material identity, source7 quality/performance; ordinary wheel/Speed still pending'},indent=2)+'\n')
    print('Frozen43patches,actualpinnedrebuild,offlinefullvalidatorandtools passed; ordinarywheel/Speed pending',flush=True)

if __name__=='__main__':main()
