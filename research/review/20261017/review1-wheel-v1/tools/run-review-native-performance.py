"""等正确性与编译队列退出后，用同一协议测改后原生三类输入。"""
from pathlib import Path
import json
import os
import subprocess
import time

RUN = Path(__file__).parent
OLD = Path(r'C:\workspace\runs\siltok-speed\flash-perf-20261016')
PYTHON = Path(r'C:\workspace\envs\siltok-speed\flash-review1\Scripts\python.exe')
CACHE = Path(r'C:\workspace\build\freetoken-mats\review1-materials-01\kernel-cache')


def main():
    """串行运行，每类3次512输出；失败时保留报告并停止队列。"""
    deadline = time.monotonic() + 1800
    while not (RUN/'native-quality-complete.json').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Native quality did not complete')
        time.sleep(2)
    receipts = json.loads((RUN/'native-helper-validation/results.json').read_text())
    if len(receipts) != 2:
        raise RuntimeError('Native compiler queue is incomplete')
    os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(CACHE), SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(CACHE),
        FREETOKEN_DISABLE_JIT='1', TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-perf-mtp8',
        PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('PYTHONPATH', None)
    for mode, resident in [('off', 5760), ('on', 4992)]:
        stem = f'review1-native-{mode}-v1'
        command = [PYTHON, '-B', RUN/'probe-review1-wheel.py', '--output', RUN/(stem+'.json'),
            '--resident', resident, '--dynamic', 1024, '--prefill', 1024, '--state-cache-slots', 1,
            '--mapped-embedding', '--adaptive-owners', '--adapt-every', 32, '--adapt-decay', '.7',
            '--suite', 'performance', '--draft-policy', 'paired_cjk' if mode == 'on' else 'full',
            '--keys', 'english', 'chinese', 'code']
        if mode == 'on':
            command += ['--mtp', '--variant', 'compact-mtp']
        print('START', stem, flush=True)
        subprocess.run([str(PYTHON), '-B', str(OLD/'consumer-tools/supervise-v2.py'),
            '--prefix', str(RUN/stem), '--cwd', str(OLD), '--timeout', '1500', '--',
            *map(str, command)], check=True)
        resource = json.loads((RUN/(stem+'.resource.json')).read_text())
        if resource['returncode'] or resource['stop_reason'] or resource['remaining_descendants']:
            raise RuntimeError(resource)
        print('PASS', stem, flush=True)
    (RUN/'native-performance-complete.json').write_text(json.dumps({'status':'passed'})+'\n')


if __name__ == '__main__':
    main()
