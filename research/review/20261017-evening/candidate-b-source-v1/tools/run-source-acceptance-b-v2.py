"""Validate current Speed source with the candidate ordinary wheel and the real TCP boundary probe."""
from pathlib import Path
import json
import os
import runpy

RUN = Path(__file__).parent
OLD = Path(r'C:\workspace\runs\siltok-speed\review-20261017\consumer-tools')
SPEED = Path(r'C:\workspace\wt\siltok-speed\flash-perf')
PYTHON = Path(r'C:\workspace\envs\siltok-speed\flash-review2b\Scripts\python.exe')
MODEL = Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints')
CACHE = SPEED/'silitok_speed/qwen36/kernel_cache_strataq2review2b'
runner = runpy.run_path(str(RUN/'run-native-quality-b.py'))['run']


def main():
    """Keep quality and HTTP lifecycle evidence separate and forbid overlap with prior GPU work."""
    if json.loads((RUN/'candidate-b-model-boundaries-complete.json').read_text())['status'] != 'passed':
        raise RuntimeError('Prior GPU boundary stage is not complete')
    if json.loads((RUN/'candidate-b-cpu-ready-v2.json').read_text())['status'] != 'passed':
        raise RuntimeError('Source configuration/component tests are not complete')
    os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(CACHE), SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(CACHE),
                      FREETOKEN_DISABLE_JIT='1', TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-review2b',
                      PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('PYTHONPATH', None)
    for mode, resident in [('off', 5760), ('on', 4992)]:
        stem = f'candidate-b-source-backend-{mode}-quality-v2'
        command = [PYTHON, '-B', OLD/'backend8-probe.py', '--source', SPEED,
                   '--model', MODEL/'flash-next-strata-q2', '--resident', resident,
                   '--vram-gib', 15, '--host-gib', 31, '--suite', 'quality', '--output', RUN/(stem+'.json')]
        if mode == 'on':
            command += ['--mtp-runtime', MODEL/'flash-next-mtp', '--compact-mtp', '--mtp-draft-policy', 'paired_cjk']
        runner(stem, command)
    for mode, resident in [('off', 5760), ('on', 4992)]:
        stem = f'candidate-b-source-http-{mode}-functional-v2'
        shared = ['--source', SPEED, '--model', MODEL/'flash-next-strata-q2', '--kernel-cache', CACHE,
                  '--protected-experts', resident, '--vram-estimate-gib', 15, '--host-commit-estimate-gib', 31,
                  '--max-prefill-length', 1024, '--linear-state-cache-slots', 1,
                  '--moe-q2-adapt-every', 32, '--moe-q2-adapt-decay', '.7', '--gguf-embedding-host',
                  '--ple-row-cache-rows', 1048576, '--moe-q2-prefill-workspace', '--fixed-q2-rebuild-reserve-mib', 0,
                  '--vram-safety-gib', '.25', '--commit-safety-gib', 4]
        if mode == 'on':
            shared += ['--mtp-runtime', MODEL/'flash-next-mtp', '--mtp-compact-state', '--mtp-draft-policy', 'paired_cjk']
        runner(stem, [PYTHON, '-B', RUN/'run-http-smoke-review2b-v2.py', '--mode', 'http', '--tcp',
            '--lifecycle-output', RUN/(stem+'.lifecycle.json'), *shared, '--profile', 'flash-next',
            '--quant', 'q2_0_strata', '--gpu-resource-probe', 'nvml', '--startup-timeout', 600,
            '--output', RUN/(stem+'.json')])
    (RUN/'candidate-b-source-functional-complete-v2.json').write_text(json.dumps({'status':'passed'})+'\n')


if __name__ == '__main__':
    main()
