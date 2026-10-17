"""按FT8同协议验收普通review1 wheel的质量和状态重建。"""
from pathlib import Path
import json
import os
import subprocess

RUN = Path(__file__).parent
PREVIOUS = Path(r'C:\workspace\runs\siltok-speed\flash-perf-20261016')
PYTHON = Path(r'C:\workspace\envs\siltok-speed\flash-review1\Scripts\python.exe')
CACHE = Path(r'C:\workspace\build\freetoken-mats\review1-materials-01\kernel-cache')


def run(stem, command):
    """监督唯一模型进程及其全部子进程，资源不足不计入通过。"""
    print('START', stem, flush=True)
    subprocess.run([str(PYTHON), '-B', str(PREVIOUS/'consumer-tools/supervise-v2.py'),
                    '--prefix', str(RUN/stem), '--cwd', str(PREVIOUS), '--timeout', '1500',
                    '--', *map(str, command)], check=True)
    resource = json.loads((RUN/(stem+'.resource.json')).read_text())
    if resource['returncode'] or resource['stop_reason'] or resource['remaining_descendants']:
        raise RuntimeError(resource)
    print('PASS', stem, flush=True)


def main():
    """三模式都覆盖真实质量/扩缩容；计时只作为诊断，正式吞吐另行静默对照。"""
    os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(CACHE), SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(CACHE),
                      FREETOKEN_DISABLE_JIT='1', TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-perf-mtp8',
                      PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('PYTHONPATH', None)
    for mode, resident, policy in [('off', 5760, 'full'), ('on', 4992, 'paired_cjk'), ('on-full', 4992, 'full')]:
        stem = f'review1-wheel-quality-{mode}-v1'
        args = [PYTHON, '-B', RUN/'probe-review1-wheel.py', '--output', RUN/(stem+'.json'),
                '--resident', resident, '--dynamic', 1024, '--prefill', 1024,
                '--state-cache-slots', 1, '--mapped-embedding', '--adaptive-owners',
                '--adapt-every', 32, '--adapt-decay', '.7', '--suite', 'quality', '--rebuild',
                '--draft-policy', policy, '--keys', 'english', 'chinese', 'code', '70-0', 'diverse-4096-0']
        if mode != 'off':
            args += ['--mtp', '--variant', 'compact-mtp']
        run(stem, args)
    (RUN/'native-quality-complete.json').write_text(json.dumps({'status': 'passed'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
