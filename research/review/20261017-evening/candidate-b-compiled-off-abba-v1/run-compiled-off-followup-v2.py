"""Compare old/new protected payloads after a low initial short-input measurement."""
from pathlib import Path
import json
import os
import statistics
import subprocess
import time

R=Path(__file__).parent
T=R.parent/'review-20261017/consumer-tools'
S=Path(r'C:\workspace\wt\siltok-speed\flash-perf')
P=Path(r'C:\workspace\envs\siltok-speed\flash-review2b\Scripts\python.exe')
M=Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints\flash-next-strata-q2')
E=R/'compiled-off-followup-v2'
VARIANTS={
    'baseline':(Path(r'C:\workspace\build\siltok-speed\review1-p01\app'),
        'd42d3bae7f307b4651173768f323e250855aec98','staged-review1'),
    'candidate':(Path(r'C:\workspace\build\siltok-speed\review2b-p01\app'),
        '1982c4fd242ee007cfee9b127c2a27fd06b8bfd0','staged-review2b'),
}

def main():
    deadline=time.monotonic()+3600
    while not (R/'final-checks-v1.json').exists():
        log=(R/'finalize-validation-driver-v1.log').read_text(encoding='utf-8')
        if 'Traceback (most recent call last)' in log:raise RuntimeError('Final checks failed; timing followup stopped')
        if time.monotonic()>deadline:raise TimeoutError('Final CPU checks did not finish')
        time.sleep(2)
    assert json.loads((R/'final-checks-v1.json').read_text())['status']=='passed'
    for name in ('tmp','config','results'):(E/name).mkdir(parents=True,exist_ok=False)
    (E/'config/config.yaml').write_text('{"resource":{"gpu_probe":"nvml"}}',encoding='utf-8')
    reports={'baseline':[],'candidate':[]}
    for round_id,label in enumerate(('baseline','candidate','candidate','baseline'),1):
        app,commit,triton=VARIANTS[label]
        cache=app/'runtime/kernel-cache'
        stem=f'{label}-off-abba{round_id}-v2'
        output=E/'results'/(stem+'.json')
        env={key:value for key,value in os.environ.items() if not key.startswith(('SILITOK_','FREETOKEN_'))
             and key not in ('VIRTUAL_ENV','PYTHONPATH','PYTHONHOME','CUDA_PATH','CUDA_HOME')}
        env.update(PATH=';'.join([str(app/'python/Scripts'),r'C:\Windows\System32',r'C:\Windows',
            r'C:\Windows\System32\Wbem',r'C:\Windows\System32\WindowsPowerShell\v1.0']),
            TEMP=str(E/'tmp'),TMP=str(E/'tmp'),PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
            SILITOK_CONFIG=str(E/'config'),FREETOKEN_KERNEL_CACHE_DIR=str(cache),
            SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(cache),FREETOKEN_DISABLE_JIT='1',
            TRITON_CACHE_DIR=str(Path(r'C:\workspace\cache\triton')/triton),
            TORCH_EXTENSIONS_DIR=str(E/'extensions'),HF_HOME=r'C:\workspace\cache\hf')
        shared=['--compiled','--source',S,'--model',M,'--kernel-cache',cache,
            '--protected-experts',5760,'--vram-estimate-gib',15,'--host-commit-estimate-gib',31,
            '--max-prefill-length',1024,'--linear-state-cache-slots',1,'--moe-q2-adapt-every',32,
            '--moe-q2-adapt-decay','.7','--gguf-embedding-host','--ple-row-cache-rows',1048576,
            '--moe-q2-prefill-workspace','--fixed-q2-rebuild-reserve-mib',0,
            '--vram-safety-gib','.25','--commit-safety-gib',4,
            '--prompts-json',T/'frozen-prompts.json','--tokens',512,'--repetitions',3,'--output',output]
        command=[P,'-B',T/'supervise-v2.py','--prefix',output.with_suffix(''),'--cwd',app,'--timeout',1800,'--',
            app/'python/python.exe','-I','-X','utf8','-u','-B',T/'run-staged-probe.py',
            '--install-root',app,'--expected-commit',commit,'--origin-report',output.with_suffix('.origin.json'),
            '--script',T/'run-http-performance.py',*shared]
        print('START',stem,flush=True)
        subprocess.run(list(map(str,command)),cwd=app,env=env,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        report=json.loads(output.read_text())
        resource=json.loads(output.with_suffix('.resource.json').read_text())
        origin=json.loads(output.with_suffix('.origin.json').read_text())
        assert report['status']=='measured' and len(report['runs'])==6
        assert all(row['output_tokens']==512 for row in report['runs'])
        assert origin['status']=='passed' and origin['build_commit']==commit
        assert resource['returncode']==0 and resource['stop_reason'] is None
        assert not resource['remaining_before_cleanup'] and not resource['remaining_descendants']
        assert resource['min_gpu_free']>=256*2**20 and resource['min_commit_available']>=2**30
        assert resource['min_ram_available']>=2*2**30
        reports[label].append(output.name)
        print('PASS',stem,flush=True)
    summary={}
    for context in ('70-0','diverse-4096-0'):
        samples={label:[row['decode_tps'] for name in names
            for row in json.loads((E/'results'/name).read_text())['runs']
            if row['label'].startswith(context)] for label,names in reports.items()}
        medians={label:statistics.median(values) for label,values in samples.items()}
        summary[context]=dict(samples=samples,median_tps=medians,
            change_percent=100*(medians['candidate']/medians['baseline']-1))
    result=dict(status='measured',scope='old/new protected payload OFF ABBA;24real512 outputs; all observations retained',
        original_low_sample='staged-review2b/results/http-off-performance-v1.json',
        protocol='Same weights/options/prompts; independent embedded interpreters, isolated PATH; prior warmed task Triton caches; no CPU build/test overlap',
        variants={label:dict(app=str(row[0]),commit=row[1]) for label,row in VARIANTS.items()},reports=reports,summary=summary)
    (E/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (R/'compiled-followup-complete-v2.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()

