"""Accept the installed sixth wheel after the independent research GPU queue finishes."""
from pathlib import Path
import json
import psutil
import subprocess

RUN=Path(__file__).parent
PY=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand6\Scripts\python.exe')


def supervise(stem,command):
    status=subprocess.run([str(PY),'-B',str(RUN/'consumer-tools/supervise-v2.py'),
        '--prefix',str(RUN/stem),'--cwd',str(RUN),'--timeout','900','--',*command],cwd=RUN)
    r=json.loads((RUN/(stem+'.resource.json')).read_text())
    if status.returncode or r['returncode'] or r['stop_reason'] or r['remaining_descendants']:raise RuntimeError((stem,r))


def main():
    try:
        old=psutil.Process(16376)
        if 'run-freeze6-materials-v1.py' not in ' '.join(old.cmdline()):raise RuntimeError('CPU build owner changed')
        old.wait(timeout=1800)
    except psutil.NoSuchProcess:pass
    assert json.loads((RUN/'build-perf6-offline-ok-v1.json').read_text())['status']=='passed'
    supervise('perf6-wheel-components-v1',[str(PY),'-B','test-wheel6-components-v1.py'])
    for suite in ('quality','performance'):
        for mode,resident in (('on',5120),('off',5888)):
            stem=f'perf6-wheel-{suite}-{mode}-v1'
            command=[str(PY),'-B','probe63-wheel6.py','--output',stem+'.json','--mapped-embedding',
                '--adapt-decay','0.7','--variant','compact-mtp' if mode=='on' else 'base','--adaptive-owners',
                '--adapt-every','32','--resident',str(resident),'--dynamic','1024','--prefill','1024',
                '--state-cache-slots','1','--ple-row-cache','1048576','--suite',suite]
            if mode=='on':command+=['--mtp']
            if suite=='quality':command+=['--rebuild']
            else:command+=['--keys','english','chinese','code','70-0','diverse-4096-0','--repetitions','3','--inventory']
            supervise(stem,command)
    print('Ordinary perf6 wheel component/quality/throughput acceptance completed',flush=True)


if __name__=='__main__':main()
