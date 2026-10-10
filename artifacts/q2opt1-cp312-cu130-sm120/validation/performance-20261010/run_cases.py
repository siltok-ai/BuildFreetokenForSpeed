"""顺序执行隔离 A/B，准确保存命令、环境和终态。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = 'C:/workspace/envs/speed-flash-next-q2/Scripts/python.exe'


def main():
    """一次只运行一个真实模型；每次实验的旧输出永不覆盖。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--after-pid', type=int)
    parser.add_argument('cases', nargs='+', choices=['grouped', 'combined', 'grouped-chunk4096', 'combined-chunk4096', 'baseline-repeat'])
    args = parser.parse_args()
    if args.after_pid:
        import psutil
        try:
            previous = psutil.Process(args.after_pid)
        except psutil.NoSuchProcess:
            previous = None
        if previous is not None:
            print(f'WAIT live baseline process {args.after_pid}', flush=True)
            previous.wait()
        baseline = json.loads((ROOT / 'native-baseline.json').read_text(encoding='utf-8'))
        if not baseline.get('success'):
            raise RuntimeError('baseline did not finish successfully')
    protocol = {
        'grouped': ('runtime-candidate', {'FREETOKEN_Q2_SPARSE_PREFILL': '0'}),
        'combined': ('runtime-candidate', {}),
        'grouped-chunk4096': ('runtime-candidate', {'FREETOKEN_Q2_SPARSE_PREFILL': '0'}),
        'combined-chunk4096': ('runtime-candidate', {}),
        'baseline-repeat': ('runtime-base', {}),
    }
    receipts = []
    receipt_path = ROOT / ('commands-' + '-'.join(args.cases) + '.json')
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    for name in args.cases:
        runtime, knobs = protocol[name]
        target = ROOT / f'native-{name}.json'
        if target.exists():
            raise FileExistsError(target)
        environment = os.environ.copy()
        overrides = {'Q2_EXPERIMENT_RUNTIME': str(ROOT / runtime),
                     'Q2_SPEED_ROOT': os.environ.get('Q2_SPEED_ROOT', 'C:/workspace/worktrees/speed-flash-next-q2-perf'),
                     'TRITON_CACHE_DIR': str(ROOT / 'triton-cache'), **knobs}
        environment.update(overrides)
        command = [PYTHON, '-B', '-u', str(ROOT / 'supervise.py'), '--output',
                   str(ROOT / f'native-{name}.capacity.json'), '--projected-gib', '50.3', '--',
                   PYTHON, '-B', '-u', str(ROOT / 'probe.py'), '--suite', 'full', '--output', str(target)]
        if name.endswith('chunk4096'):
            command += ['--chunk', '4096']
        receipt = {'case': name, 'command': command, 'env': overrides, 'started_unix': time.time()}
        receipts.append(receipt)
        receipt_path.write_text(json.dumps(receipts, indent=2), encoding='utf-8')
        print('START ' + name, flush=True)
        done = subprocess.run(command, env=environment)
        receipt.update(returncode=done.returncode, finished_unix=time.time())
        receipt_path.write_text(json.dumps(receipts, indent=2), encoding='utf-8')
        if done.returncode:
            raise RuntimeError(f'{name} failed: {done.returncode}')
        print('DONE ' + name, flush=True)


if __name__ == '__main__':
    main()
