"""After functional acceptance, collect fresh ABBA throughput and the old35B regression serially."""
from pathlib import Path
import json
import os
import runpy
import statistics
import subprocess
import time

RUN = Path(__file__).parent
TOOLS = Path(r'C:\workspace\runs\siltok-speed\review-20261017\consumer-tools')
MODEL = Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints')
runner = runpy.run_path(str(RUN/'run-native-quality.py'))['run']
CANDIDATE = (Path(r'C:\workspace\envs\siltok-speed\flash-review2a\Scripts\python.exe'),
             Path(r'C:\workspace\wt\siltok-speed\flash-perf'), 'kernel_cache_strataq2review2a')
BASELINE = (Path(r'C:\workspace\envs\siltok-speed\flash-review1-release\Scripts\python.exe'),
            Path(r'C:\workspace\wt\siltok-speed\review2-base-20261017'), 'kernel_cache_strataq2review1')


def main():
    """Keep all four rounds and every delivered token; do not select faster measurements."""
    ready = RUN/'candidate-a-source-functional-complete.json'
    deadline = time.monotonic()+2400
    while not ready.exists():
        log = (RUN/'source-acceptance-driver-v1.log').read_text(encoding='utf-8')
        if 'Traceback (most recent call last)' in log:
            raise RuntimeError('Functional stage failed; followup queue stopped')
        if time.monotonic() >= deadline:
            raise TimeoutError('Functional stage did not complete')
        time.sleep(2)
    if 'Ran 48 tests' not in (RUN/'build-tests-after-v1.log').read_text(encoding='utf-8'):
        raise RuntimeError('Do not overlap throughput with repository checks')
    os.environ.update(FREETOKEN_DISABLE_JIT='1', TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-review2a',
                      PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('PYTHONPATH', None)
    reports = {}
    for mode, resident in [('off',5760), ('on',4992)]:
        reports[mode] = {'baseline':[], 'candidate':[]}
        for round_id, label in enumerate(('baseline','candidate','candidate','baseline'), 1):
            python, source, cache_name = BASELINE if label == 'baseline' else CANDIDATE
            cache = source/'silitok_speed/qwen36'/cache_name
            os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(cache), SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(cache))
            stem = f'{label}-http-{mode}-abba{round_id}-v1'
            shared = ['--source',source,'--model',MODEL/'flash-next-strata-q2','--kernel-cache',cache,
                      '--protected-experts',resident,'--vram-estimate-gib',15,'--host-commit-estimate-gib',31,
                      '--max-prefill-length',1024,'--linear-state-cache-slots',1,'--moe-q2-adapt-every',32,
                      '--moe-q2-adapt-decay','.7','--gguf-embedding-host','--ple-row-cache-rows',1048576,
                      '--moe-q2-prefill-workspace','--fixed-q2-rebuild-reserve-mib',0,
                      '--vram-safety-gib','.25','--commit-safety-gib',4]
            if mode == 'on':
                shared += ['--mtp-runtime',MODEL/'flash-next-mtp','--mtp-compact-state','--mtp-draft-policy','paired_cjk']
            runner(stem,[python,'-B',TOOLS/'run-http-performance.py',*shared,
                '--prompts-json',TOOLS/'frozen-prompts.json','--tokens',512,'--repetitions',3,'--output',RUN/(stem+'.json')])
            report = json.loads((RUN/(stem+'.json')).read_text(encoding='utf-8'))
            if report['status'] != 'passed' or any(row['output_tokens'] != 512 for row in report['runs']):
                raise RuntimeError('Incomplete throughput output')
            reports[mode][label].append(stem+'.json')
    summary = {}
    for mode, labels in reports.items():
        summary[mode] = {}
        for context in ('70-0','diverse-4096-0'):
            values = {label:[row['decode_tps'] for name in names
                            for row in json.loads((RUN/name).read_text(encoding='utf-8'))['runs']
                            if row['label'].startswith(context)] for label,names in labels.items()}
            medians = {key:statistics.median(data) for key,data in values.items()}
            summary[mode][context] = dict(samples=values, median_tps=medians,
                change_percent=100*(medians['candidate']/medians['baseline']-1))
    (RUN/'http-abba-summary-v1.json').write_text(json.dumps(dict(protocol='ABBA; 6x512 per input/variant/mode; all runs retained',
        reports=reports, summary=summary), indent=2)+'\n')
    (RUN/'candidate-a-source-performance-complete.json').write_text(json.dumps({'status':'passed'})+'\n')

    python, source, cache_name = CANDIDATE
    cache = source/'silitok_speed/qwen36'/cache_name
    os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(cache), SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(cache))
    stem = 'candidate-a-source-qwen36-backend-v1'
    runner(stem,[python,'-B',TOOLS/'qwen36-legacy-probe.py','--mode','backend','--source',source,
        '--model',r'C:\workspace\models\siltok-speed\languages\qwen3.6-35b-a3b.nvfp4-modelscope',
        '--profile','qwen36','--quant','nvfp4','--kernel-cache',cache,'--repeats',1,'--long-repeats',1,
        '--tokens',64,'--long-prompt-tokens',256,'--long-output-tokens',16,'--quality','--output',RUN/(stem+'.json')])
    subprocess.run([str(python),'-B',str(TOOLS/'audit-qwen-regression.py'),'--report',str(RUN/(stem+'.json')),
                    '--baseline',str(TOOLS/'qwen36-ft2-backend-v1.json'),
                    '--output',str(RUN/'candidate-a-source-qwen36-audit-v1.json')],check=True)
    (RUN/'candidate-a-source-all-complete.json').write_text(json.dumps({'status':'passed'})+'\n')


if __name__ == '__main__':
    main()
