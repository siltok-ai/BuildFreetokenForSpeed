"""HTTP MTP独立ABBA复测，完整保存初轮和全部复测，不与构建竞争。"""
from pathlib import Path
import json
import os
import subprocess
import time
R=Path(__file__).parent
S=Path(r'C:\workspace\wt\siltok-speed\flash-perf')
A=Path(r'C:\workspace\wt\siltok-speed\review-base-20261017')
M=Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints')
T=R/'consumer-tools'
def main():
    deadline=time.monotonic()+3600
    while not (R/'speed-ordinary-wheel-inventory-v1.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Ordinary wheel CPU preparation unfinished')
        time.sleep(2)
    os.environ.update(FREETOKEN_DISABLE_JIT='1',TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-perf-mtp8',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('PYTHONPATH',None)
    for label,version in [('baseline',2),('review1',2),('review1',3),('baseline',3)]:
        root=A if label=='baseline' else S
        cache=root/('silitok_speed/qwen36/kernel_cache_strataq2perf8' if label=='baseline' else 'silitok_speed/qwen36/kernel_cache_strataq2review1')
        python=Path(r'C:\workspace\envs\siltok-speed')/('flash-perf-cand8' if label=='baseline' else 'flash-review1')/'Scripts/python.exe'
        os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(cache),SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(cache))
        stem=f'{label}-http-on-v{version}'
        cmd=[python,'-B',T/'run-http-performance.py','--source',root,'--model',M/'flash-next-strata-q2','--kernel-cache',cache,'--protected-experts',4992,'--vram-estimate-gib',15,'--host-commit-estimate-gib',31,'--max-prefill-length',1024,'--linear-state-cache-slots',1,'--moe-q2-adapt-every',32,'--moe-q2-adapt-decay','.7','--gguf-embedding-host','--ple-row-cache-rows',1048576,'--moe-q2-prefill-workspace','--fixed-q2-rebuild-reserve-mib',0,'--vram-safety-gib','.25','--commit-safety-gib',4,'--mtp-runtime',M/'flash-next-mtp','--mtp-compact-state','--mtp-draft-policy','paired_cjk','--prompts-json',T/'frozen-prompts.json','--tokens',512,'--repetitions',3,'--output',R/(stem+'.json')]
        print('START',stem,flush=True)
        subprocess.run([str(python),'-B',str(T/'supervise-v2.py'),'--prefix',str(R/stem),'--cwd',str(root),'--timeout','1500','--',*map(str,cmd)],check=True)
        d=json.loads((R/(stem+'.resource.json')).read_text())
        if d['returncode'] or d['stop_reason'] or d['remaining_descendants']:raise RuntimeError(d)
    (R/'http-repeat-complete.json').write_text(json.dumps({'status':'passed'})+'\n')
if __name__=='__main__':main()
