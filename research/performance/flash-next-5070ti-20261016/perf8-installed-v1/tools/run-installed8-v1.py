"""Accept the actual i07 installation and its matching protected payload."""
from pathlib import Path
import json
import os
import subprocess
import time

R=Path(__file__).parent
T=R/'consumer-tools'
S=Path(r'C:\workspace\wt\siltok-speed\flash-perf')
APP=Path(r'C:\workspace\build\siltok-speed\i07')
P=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand8\Scripts\python.exe')
M=Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints')
E=R/'install-i07'
COMMIT=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()

def main():
    """Use embedded Python, protected loader and private caches under an isolated PATH."""
    installed=json.loads((E/'result.json').read_text(encoding='utf-8-sig'))
    assert installed['status']=='installed_service_stopped'
    assert not (E/'results-v1').exists()
    for name in ('tmp','config','extensions','results-v1'):(E/name).mkdir()
    (E/'config/config.yaml').write_text('{"resource":{"gpu_probe":"nvml"}}')
    results=E/'results-v1';cache=APP/'runtime/kernel-cache'
    env={k:v for k,v in os.environ.items() if not k.startswith(('SILITOK_','FREETOKEN_'))
        and k not in ('VIRTUAL_ENV','PYTHONPATH','PYTHONHOME','CUDA_PATH','CUDA_HOME')}
    env.update(PATH=';'.join([str(APP/'python/Scripts'),r'C:\Windows\System32',r'C:\Windows',r'C:\Windows\System32\WindowsPowerShell\v1.0',r'C:\Program Files\NVIDIA Corporation\NVSMI']),
        TEMP=str(E/'tmp'),TMP=str(E/'tmp'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
        SILITOK_CONFIG=str(E/'config'),SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(cache),
        FREETOKEN_KERNEL_CACHE_DIR=str(cache),FREETOKEN_DISABLE_JIT='1',
        TRITON_CACHE_DIR=r'C:\workspace\cache\triton\installed-i07\warmup',
        TORCH_EXTENSIONS_DIR=r'C:\workspace\cache\ext\installed-i07',HF_HOME=r'C:\workspace\cache\hf')
    (results/'environment.json').write_text(json.dumps({k:env[k] for k in ('PATH','TEMP','SILITOK_CONFIG',
        'FREETOKEN_KERNEL_CACHE_DIR','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR')},indent=2)+'\n')
    completed=[]
    def run(name,script,args,timeout=1500):
        """Require clean bundle identity, real resource supervision and no descendants."""
        if '-performance-' in name:
            deadline=time.monotonic()+1200
            while not (R/'perf8-preinstalled-cpu-ready-v1.json').exists():
                if time.monotonic()>deadline:raise TimeoutError('Final CPU review checks unfinished before installed TPS')
                time.sleep(2)
        (results/(name+'.environment.json')).write_text(json.dumps({k:env[k] for k in
            ('PATH','FREETOKEN_KERNEL_CACHE_DIR','FREETOKEN_DISABLE_JIT','TRITON_CACHE_DIR')},indent=2)+'\n')
        command=[str(P),'-B',str(T/'supervise-v2.py'),'--prefix',str(results/name),
            '--cwd',str(APP),'--timeout',str(timeout),'--',str(APP/'python/python.exe'),'-I','-X','utf8','-u','-B',
            str(T/'run-installed-probe.py'),'--install-root',str(APP),'--expected-commit',COMMIT,
            '--origin-report',str(results/(name+'.origin.json')),'--script',str(script),*map(str,args)]
        subprocess.run(command,env=env,cwd=APP,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        data=json.loads((results/(name+'.resource.json')).read_text())
        assert data['returncode']==0 and data['stop_reason'] is None and not data['remaining_descendants'],data
        identity=json.loads((results/(name+'.origin.json')).read_text())
        assert identity['status']=='passed' and identity['scope']=='actual_installed_payload_not_clean_os'
        completed.append(name)
        (E/'suite-progress.json').write_text(json.dumps(dict(status='running',scope=identity['scope'],
            completed=completed,commit=COMMIT,app=str(APP)),indent=2)+'\n')
    run('runtime-identity-v1',T/'installed8-runtime-identity.py',['--output',results/'runtime-identity-v1.json'])
    for mode,resident in [('off',5760),('on',4992)]:
        env['TRITON_CACHE_DIR']=rf'C:\workspace\cache\triton\installed-i07\{mode}-v1'
        shared=['--source',S,'--model',M/'flash-next-strata-q2','--kernel-cache',cache,
            '--protected-experts',resident,'--vram-estimate-gib',15,'--host-commit-estimate-gib',31,
            '--max-prefill-length',1024,'--linear-state-cache-slots',1,'--moe-q2-adapt-every',32,
            '--moe-q2-adapt-decay',.7,'--gguf-embedding-host','--ple-row-cache-rows',1048576,
            '--moe-q2-prefill-workspace','--fixed-q2-rebuild-reserve-mib',0,
            '--vram-safety-gib',.25,'--commit-safety-gib',4]
        if mode=='on':shared+=['--mtp-runtime',M/'flash-next-mtp','--mtp-compact-state','--mtp-draft-policy','paired_cjk']
        name=f'http-{mode}-functional-v1'
        run(name,T/'run-http-smoke-v2.py',['--mode','http','--packaged','--tcp',
            '--lifecycle-output',results/(name+'.lifecycle.json'),*shared,'--profile','flash-next',
            '--quant','q2_0_strata','--gpu-resource-probe','nvml','--startup-timeout',600,
            '--output',results/(name+'.json')])
        name=f'http-{mode}-performance-v1'
        run(name,T/'run-http-performance.py',['--compiled',*shared,'--prompts-json',T/'frozen-prompts.json',
            '--tokens',512,'--repetitions',3,'--output',results/(name+'.json')])
    (E/'suite-progress.json').write_text(json.dumps(dict(status='passed',
        scope='actual_installed_payload_not_clean_os',completed=completed,commit=COMMIT,app=str(APP)),indent=2)+'\n')
    print('Actual installed i07 Flash Next suites passed',flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        if E.exists():
            (E/'suite-progress.json').write_text(json.dumps(dict(status='failed',scope='actual_installed_payload_not_clean_os'))+'\n')
        raise
