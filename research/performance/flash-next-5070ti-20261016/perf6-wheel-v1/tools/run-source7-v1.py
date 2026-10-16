from pathlib import Path
import json,runpy
q=runpy.run_path(str(Path(__file__).with_name('run-cost-and-tiles-v1.py')))
supervise,RUN=(q[k] for k in ('supervise','RUN'))
PY=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand5\Scripts\python.exe')

def main():
    r=json.loads((RUN/'prefill-borrow-r4864-performance-on-v5.resource.json').read_text())
    assert r['returncode']==0 and not r['remaining_descendants']
    supervise('source7-components-v1',[str(PY),'-B','test-source7-components-v1.py'])
    for suite in ('quality','performance'):
        for mode,resident in (('on',5120),('off',5888)):
            stem=f'source7-{suite}-{mode}-v1'
            args=[str(PY),'-B','probe62-source7.py','--output',stem+'.json',
                '--mapped-embedding','--adapt-decay','0.7','--variant','compact-mtp' if mode=='on' else 'base',
                '--adaptive-owners','--adapt-every','32','--resident',str(resident),'--dynamic','1024',
                '--prefill','1024','--state-cache-slots','1','--suite',suite,'--ple-row-cache','1048576']
            if mode=='on':args+=['--mtp']
            args+=['--rebuild'] if suite=='quality' else ['--keys','english','chinese','code','70-0','diverse-4096-0','--repetitions','3']
            supervise(stem,args)
    print('Production source7 passed component, both quality/rebuild and performance suites',flush=True)

if __name__=='__main__':main()
