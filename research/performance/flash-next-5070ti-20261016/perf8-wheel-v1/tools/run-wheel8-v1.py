"""Accept actual ordinary wheel8, including full and paired MTP strategies."""
from pathlib import Path
import json,subprocess,time
R=Path(__file__).parent
P=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand8\Scripts\python.exe')

def run(stem,args):
    subprocess.run([str(P),'-B',str(R/'consumer-tools/supervise-v2.py'),'--prefix',str(R/stem),
        '--cwd',str(R),'--timeout','1200','--',str(P),'-B',*map(str,args)],cwd=R,check=True)
    d=json.loads((R/(stem+'.resource.json')).read_text());assert d['stop_reason'] is None and not d['remaining_descendants']

def main():
    run('perf8-wheel-components-v1',[R/'test-wheel8-components-v1.py'])
    run('perf8-wheel-mtp-components-v1',[R/'test-wheel8-mtp-v1.py'])
    for suite in ('quality','performance'):
        if suite=='performance':
            deadline=time.monotonic()+1200
            while not (R/'build-perf8-cpu-ok-v1.json').exists():
                if time.monotonic()>deadline:raise TimeoutError('Offline checks unfinished before TPS')
                time.sleep(2)
        for mode,resident,policy in [('off',5760,'full'),('on',4992,'paired_cjk'),('on-full',4992,'full')]:
            stem=f'perf8-wheel-{suite}-{mode}-v1'
            args=[R/'probe66-wheel8.py','--output',R/(stem+'.json'),'--resident',resident,'--dynamic',1024,
                '--prefill',1024,'--state-cache-slots',1,'--mapped-embedding','--adaptive-owners',
                '--adapt-every',32,'--adapt-decay',.7,'--suite',suite,'--draft-policy',policy,
                '--keys','english','chinese','code','70-0','diverse-4096-0']
            if mode!='off':args+=['--mtp','--variant','compact-mtp']
            if suite=='quality':args+=['--rebuild']
            run(stem,args)
    print('Ordinary wheel8 components/quality/rebuild/performance all completed',flush=True)

if __name__=='__main__':main()
