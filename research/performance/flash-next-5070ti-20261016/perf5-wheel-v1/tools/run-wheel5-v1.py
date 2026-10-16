"""Accept the installed fifth wheel after the independent research GPU queue finishes."""
from pathlib import Path
import json
import psutil
import subprocess

RUN=Path(__file__).parent
PY=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand5\Scripts\python.exe')


def supervise(stem,command):
    status=subprocess.run([str(PY),'-B',str(RUN/'consumer-tools/supervise-v2.py'),
        '--prefix',str(RUN/stem),'--cwd',str(RUN),'--timeout','900','--',*command],cwd=RUN)
    r=json.loads((RUN/(stem+'.resource.json')).read_text())
    if status.returncode or r['returncode'] or r['stop_reason'] or r['remaining_descendants']:raise RuntimeError((stem,r))


def main():
    try:
        old=psutil.Process(17464)
        if 'run-head-cache-resume-v2.py' not in ' '.join(old.cmdline()):raise RuntimeError('Previous GPU owner identity changed')
        old.wait(timeout=2400)
    except psutil.NoSuchProcess:pass
    final=json.loads((RUN/'head-cache-r5568-performance-off-v2.resource.json').read_text())
    assert final['returncode']==0 and final['stop_reason'] is None and not final['remaining_descendants']
    supervise('perf5-wheel-components-v1',[str(PY),'-B','test-wheel5-components-v1.py'])
    for suite in ('quality','performance'):
        for mode,resident in (('on',4608),('off',5376)):
            stem=f'perf5-wheel-{suite}-{mode}-v1'
            command=[str(PY),'-B','probe49-wheel5.py','--output',stem+'.json','--mapped-embedding',
                '--adapt-decay','0.7','--variant','compact-mtp' if mode=='on' else 'base','--adaptive-owners',
                '--adapt-every','32','--resident',str(resident),'--dynamic','1024','--prefill','1024',
                '--state-cache-slots','1','--ple-row-cache','1048576','--suite',suite]
            if mode=='on':command+=['--mtp']
            if suite=='quality':command+=['--rebuild']
            else:command+=['--keys','english','chinese','code','70-0','diverse-4096-0','--repetitions','3','--inventory']
            supervise(stem,command)
    print('Ordinary perf5 wheel component/quality/throughput acceptance completed',flush=True)


if __name__=='__main__':main()
