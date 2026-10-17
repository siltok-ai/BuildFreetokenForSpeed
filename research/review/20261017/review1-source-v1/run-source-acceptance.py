"""新review1普通wheel的Speed源码、真实TCP和旧35B回归，GPU任务严格串行。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import time
R=Path(__file__).parent
T=R/'consumer-tools'
S=Path(r'C:\workspace\wt\siltok-speed\flash-perf')
P=Path(r'C:\workspace\envs\siltok-speed\flash-review1\Scripts\python.exe')
M=Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints')
C=S/'silitok_speed/qwen36/kernel_cache_strataq2review1'
def run(stem,command):
    print('START',stem,flush=True)
    subprocess.run([str(P),'-B',str(T/'supervise-v2.py'),'--prefix',str(R/stem),'--cwd',str(S),'--timeout','1500','--',*map(str,command)],check=True)
    report=json.loads((R/(stem+'.resource.json')).read_text())
    if report['returncode'] or report['stop_reason'] or report['remaining_descendants']:raise RuntimeError(report)
    print('PASS',stem,flush=True)
def main():
    deadline=time.monotonic()+1800
    while not (R/'speed-cpu-ready.json').exists():
        if time.monotonic()>=deadline:raise TimeoutError('CPU validation unfinished')
        time.sleep(2)
    if json.loads((R/'speed-cpu-ready.json').read_text())['status']!='passed':raise RuntimeError('CPU tests failed')
    os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(C),SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(C),FREETOKEN_DISABLE_JIT='1',TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-perf-mtp8',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('PYTHONPATH',None)
    for mode,resident in [('off',5760),('on',4992)]:
        stem=f'review1-source-backend-{mode}-quality-v1'
        command=[P,'-B',T/'backend8-probe.py','--source',S,'--model',M/'flash-next-strata-q2','--resident',resident,'--vram-gib',15,'--host-gib',31,'--suite','quality','--output',R/(stem+'.json')]
        if mode=='on':command+=['--mtp-runtime',M/'flash-next-mtp','--compact-mtp','--mtp-draft-policy','paired_cjk']
        run(stem,command)
    for mode,resident in [('off',5760),('on',4992)]:
        shared=['--source',S,'--model',M/'flash-next-strata-q2','--kernel-cache',C,'--protected-experts',resident,'--vram-estimate-gib',15,'--host-commit-estimate-gib',31,'--max-prefill-length',1024,'--linear-state-cache-slots',1,'--moe-q2-adapt-every',32,'--moe-q2-adapt-decay','.7','--gguf-embedding-host','--ple-row-cache-rows',1048576,'--moe-q2-prefill-workspace','--fixed-q2-rebuild-reserve-mib',0,'--vram-safety-gib','.25','--commit-safety-gib',4]
        if mode=='on':shared+=['--mtp-runtime',M/'flash-next-mtp','--mtp-compact-state','--mtp-draft-policy','paired_cjk']
        stem=f'review1-source-http-{mode}-functional-v1'
        run(stem,[P,'-B',T/'run-http-smoke-v2.py','--mode','http','--tcp','--lifecycle-output',R/(stem+'.lifecycle.json'),*shared,'--profile','flash-next','--quant','q2_0_strata','--gpu-resource-probe','nvml','--startup-timeout',600,'--output',R/(stem+'.json')])
        stem=f'review1-source-http-{mode}-performance-v1'
        run(stem,[P,'-B',T/'run-http-performance.py',*shared,'--prompts-json',T/'frozen-prompts.json','--tokens',512,'--repetitions',3,'--output',R/(stem+'.json')])
    (R/'source-flash-complete.json').write_text(json.dumps({'status':'passed'})+'\n')
    stem='review1-source-qwen36-backend-v1'
    oldmodel=Path(r'C:\workspace\models\siltok-speed\languages\qwen3.6-35b-a3b.nvfp4-modelscope')
    run(stem,[P,'-B',r'C:\workspace\tools\bin\validate-speed.py','--suite','qwen-backend','--source',S,'--materials',r'C:\workspace\wt\freetoken-mats\flash-perf','--model',oldmodel,'--profile','qwen36','--quant','nvfp4','--kernel-cache',C,'--repeats',1,'--long-repeats',1,'--tokens',64,'--long-prompt-tokens',256,'--long-output-tokens',16,'--quality','--output',R/(stem+'.json')])
    subprocess.run([str(P),'-B',str(T/'audit-qwen-regression.py'),'--report',str(R/(stem+'.json')),'--baseline',str(T/'qwen36-ft2-backend-v1.json'),'--output',str(R/'review1-source-qwen36-audit-v1.json')],check=True)
    (R/'source-acceptance-complete.json').write_text(json.dumps({'status':'passed'})+'\n')
if __name__=='__main__':main()
