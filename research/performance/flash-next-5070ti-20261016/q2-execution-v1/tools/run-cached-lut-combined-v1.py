"""组合已独立验证的Q2 LUT和缓存副本交换，先质量后普通吞吐。"""
from pathlib import Path
import json
import subprocess

RUN=Path(__file__).parent
PY=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand4\Scripts\python.exe')


def run(stem,mode,suite,resident,dynamic,every,swaps):
    """固定总槽位的配方采用独立模型，要求完整监督退出。"""
    command=[str(PY),'-B','probe35-cached-adapt.py','--output',stem+'.json',
        '--cached-adapt','--q2-lut','--adapt-max-swaps',str(swaps),
        '--mapped-embedding','--adapt-decay','0.7','--variant','compact-mtp' if mode=='on' else 'base',
        '--adaptive-owners','--adapt-every',str(every),'--resident',str(resident),'--dynamic',str(dynamic),
        '--prefill','1024','--state-cache-slots','1','--suite',suite]
    if mode=='on':command+=['--mtp']
    if suite=='quality':command+=['--rebuild']
    else:command+=['--keys','english','chinese','code','70-0','diverse-4096-0','--repetitions','3']
    status=subprocess.run([str(PY),'-B',str(RUN/'consumer-tools/supervise-v2.py'),
        '--prefix',str(RUN/stem),'--cwd',str(RUN),'--timeout','900','--',*command],cwd=RUN)
    resource=json.loads((RUN/(stem+'.resource.json')).read_text())
    if status.returncode or resource['returncode'] or resource['stop_reason'] or resource['remaining_descendants']:
        raise RuntimeError((stem,resource))


def main():
    """标准分区先完成两模式；其后单独探索与Strata核心驻留数更接近的D0配方。"""
    for suite in ('quality','performance'):
        for mode,resident in (('on',4608),('off',5376)):
            run(f'cached-lut-{suite}-{mode}-v1',mode,suite,resident,1024,32,96)
    for suite in ('quality','performance'):
        run(f'cached-lut-d0-e8-m48-{suite}-on-v1','on',suite,5632,0,8,48)
    print('Cached LUT combined recipes completed',flush=True)


if __name__=='__main__':main()
