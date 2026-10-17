"""固定full草稿策略的HTTP A/B，排除按墙钟选臂的影响，测核心执行开销。"""
from pathlib import Path
import json
import os
import subprocess
import time
R=Path(__file__).parent;T=R/'consumer-tools'
S=Path(r'C:\workspace\wt\siltok-speed\flash-perf');A=Path(r'C:\workspace\wt\siltok-speed\review-base-20261017')
M=Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints')
end=time.monotonic()+1800
while not (R/'http-repeat-complete.json').exists():
    if time.monotonic()>end:raise TimeoutError('ABBA unfinished')
    time.sleep(2)
os.environ.update(FREETOKEN_DISABLE_JIT='1',TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-perf-mtp8',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
for label,root,envname,cachename in [('baseline',A,'flash-perf-cand8','strataq2perf8'),('review1',S,'flash-review1','strataq2review1')]:
    cache=root/f'silitok_speed/qwen36/kernel_cache_{cachename}'
    py=Path(r'C:\workspace\envs\siltok-speed')/envname/'Scripts/python.exe'
    os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(cache),SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(cache))
    name=f'{label}-http-full-on-v1'
    cmd=[py,'-B',T/'run-http-performance.py','--source',root,'--model',M/'flash-next-strata-q2','--kernel-cache',cache,'--protected-experts',4992,'--vram-estimate-gib',15,'--host-commit-estimate-gib',31,'--max-prefill-length',1024,'--linear-state-cache-slots',1,'--moe-q2-adapt-every',32,'--moe-q2-adapt-decay','.7','--gguf-embedding-host','--ple-row-cache-rows',1048576,'--moe-q2-prefill-workspace','--fixed-q2-rebuild-reserve-mib',0,'--vram-safety-gib','.25','--commit-safety-gib',4,'--mtp-runtime',M/'flash-next-mtp','--mtp-compact-state','--mtp-draft-policy','full','--prompts-json',T/'frozen-prompts.json','--tokens',512,'--repetitions',3,'--output',R/(name+'.json')]
    print('START',name,flush=True)
    subprocess.run([str(py),'-B',str(T/'supervise-v2.py'),'--prefix',str(R/name),'--cwd',str(root),'--timeout','1500','--',*map(str,cmd)],check=True)
    d=json.loads((R/(name+'.resource.json')).read_text());assert d['returncode']==0 and d['stop_reason'] is None and not d['remaining_descendants']
(R/'http-fixed-full-complete.json').write_text(json.dumps({'status':'passed'})+'\n')
