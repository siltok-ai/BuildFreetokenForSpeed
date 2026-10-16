"""串行启动固定对照，只有前一模型完整退出才继续下一项。"""
from pathlib import Path
import json
import subprocess
import sys

RUN=Path(__file__).parent
ST=Path(r'C:\workspace\runs\strata\flash-perf-20261016')
PY=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand4\Scripts\python.exe')
STPY=Path(r'C:\workspace\envs\strata\flash-perf\Scripts\python.exe')
SUP=RUN/'consumer-tools/supervise-v2.py'


def execute(prefix,cwd,command):
    """使用原有资源下限，并核验进程树终态后才交出GPU。"""
    dest=cwd/prefix
    result=subprocess.run([str(PY),'-B',str(SUP),'--prefix',str(dest),'--cwd',str(cwd),
        '--timeout','1800','--',*map(str,command)],cwd=RUN)
    resource=json.loads(dest.with_suffix('.resource.json').read_text())
    if result.returncode or resource['returncode'] or resource['stop_reason'] or resource['remaining_descendants']:
        raise RuntimeError((prefix,result.returncode,resource))


def main():
    """反向消融与两模式正式吞吐使用各自新进程，不并发运行重负载。"""
    for mode in ('on','off'):
        prefix=f'candidate4-wheel-performance-{mode}-v1'
        command=[PY,'-B',RUN/'probe27-wheel4.py','--output',str(RUN/(prefix+'.json')),
            '--mapped-embedding','--adapt-decay','0.7','--variant','compact-mtp' if mode=='on' else 'base',
            '--adaptive-owners','--adapt-every','32','--resident','4608' if mode=='on' else '5376',
            '--prefill','1024','--state-cache-slots','1','--suite','performance',
            '--keys','english','chinese','code','70-0','diverse-4096-0','--repetitions','3']
        if mode=='on':command+=['--mtp']
        execute(prefix,RUN,command)
    prefix='reference-mtp-pcie1-fixed-3x-v1'
    execute(prefix,ST,[STPY,'-B',ST/'probe-ablation-v2.py','--output',str(ST/(prefix+'.json')),
        '--mtp','--pcie','1','--slots','7019','--suffix','0','--repetitions','3'])
    print('All three measurements completed',flush=True)


if __name__=='__main__':main()
