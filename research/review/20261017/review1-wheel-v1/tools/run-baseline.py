"""固定本轮同机基线；依次运行唯一GPU模型进程，并保留资源与完整输出。"""
from pathlib import Path
import json
import os
import subprocess

RUN = Path(__file__).parent
PREVIOUS = Path(r'C:\workspace\runs\siltok-speed\flash-perf-20261016')
SPEED = Path(r'C:\workspace\wt\siltok-speed\review-base-20261017')
PYTHON = Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand8\Scripts\python.exe')
MODELS = Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints')
CACHE = SPEED / 'silitok_speed/qwen36/kernel_cache_strataq2perf8'


def run(stem, command):
    """运行原验收入口并确认监督器未触发资源终止，全部子进程退出。"""
    prefix = RUN / stem
    print('START', stem, flush=True)
    subprocess.run([str(PYTHON), '-B', str(PREVIOUS/'consumer-tools/supervise-v2.py'),
                    '--prefix', str(prefix), '--cwd', str(PREVIOUS), '--timeout', '1500',
                    '--', *map(str, command)], check=True)
    result = json.loads(prefix.with_suffix('.resource.json').read_text())
    if result['stop_reason'] or result['remaining_descendants'] or result['returncode']:
        raise RuntimeError(result)
    print('PASS', stem, flush=True)


def main():
    """同一普通FT8 wheel，先测Speed短/4K，再测原生三类输入。"""
    os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(CACHE), SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(CACHE),
                      FREETOKEN_DISABLE_JIT='1', TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-perf-mtp8',
                      PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('PYTHONPATH', None)
    for mode, resident in [('off', 5760), ('on', 4992)]:
        stem = f'baseline-http-{mode}-v1'
        args = [PYTHON, '-B', PREVIOUS/'consumer-tools/run-http-performance.py', '--source', SPEED,
                '--model', MODELS/'flash-next-strata-q2', '--kernel-cache', CACHE,
                '--protected-experts', resident, '--vram-estimate-gib', 15, '--host-commit-estimate-gib', 31,
                '--max-prefill-length', 1024, '--linear-state-cache-slots', 1,
                '--moe-q2-adapt-every', 32, '--moe-q2-adapt-decay', '.7', '--gguf-embedding-host',
                '--ple-row-cache-rows', 1048576, '--moe-q2-prefill-workspace', '--fixed-q2-rebuild-reserve-mib', 0,
                '--prompts-json', PREVIOUS/'consumer-tools/frozen-prompts.json', '--output', RUN/(stem+'.json'),
                '--tokens', 512, '--repetitions', 3, '--vram-safety-gib', '.25', '--commit-safety-gib', 4]
        if mode == 'on':
            args += ['--mtp-runtime', MODELS/'flash-next-mtp', '--mtp-compact-state', '--mtp-draft-policy', 'paired_cjk']
        run(stem, args)
    for mode, resident in [('off', 5760), ('on', 4992)]:
        stem = f'baseline-native-{mode}-v1'
        args = [PYTHON, '-B', PREVIOUS/'probe66-wheel8.py', '--output', RUN/(stem+'.json'),
                '--resident', resident, '--dynamic', 1024, '--prefill', 1024,
                '--state-cache-slots', 1, '--mapped-embedding', '--adaptive-owners',
                '--adapt-every', 32, '--adapt-decay', '.7', '--suite', 'performance',
                '--draft-policy', 'paired_cjk' if mode == 'on' else 'full', '--keys', 'english', 'chinese', 'code']
        if mode == 'on':
            args += ['--mtp', '--variant', 'compact-mtp']
        run(stem, args)
    (RUN/'baseline-complete.json').write_text(json.dumps({'status': 'passed'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
