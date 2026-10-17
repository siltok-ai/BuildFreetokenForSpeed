"""Run the complete frozen quality/state-rebuild protocol in three modes, one GPU model at a time."""
from pathlib import Path
import json
import os
import subprocess

RUN = Path(__file__).parent
TOOLS = Path(r'C:\workspace\runs\siltok-speed\review-20261017\consumer-tools')
PYTHON = Path(r'C:\workspace\envs\siltok-speed\flash-review2b\Scripts\python.exe')
CACHE = Path(r'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2review2b-cp312-cu130-sm120\kernel-cache')


def run(stem, command):
    """Keep resource failures and process ownership explicit in each independent run."""
    print('START', stem, flush=True)
    subprocess.run([str(PYTHON), '-B', str(TOOLS/'supervise-v2.py'), '--prefix', str(RUN/stem),
                    '--cwd', str(RUN), '--timeout', '1500', '--', *map(str, command)], check=True)
    resource = json.loads((RUN/(stem+'.resource.json')).read_text())
    if resource['returncode'] or resource['stop_reason'] or resource['remaining_descendants']:
        raise RuntimeError(resource)
    print('PASS', stem, flush=True)


def main():
    """Use the exact previous preset and requests; preserve OFF/full/paired results separately."""
    os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(CACHE), FREETOKEN_DISABLE_JIT='1',
                      TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-review2b',
                      PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    os.environ.pop('PYTHONPATH', None)
    for mode, resident, policy in [('off', 5760, 'full'), ('on', 4992, 'paired_cjk'), ('on-full', 4992, 'full')]:
        stem = f'candidate-b-native-quality-{mode}-v1'
        command = [PYTHON, '-B', RUN/'probe-candidate-b.py', '--output', RUN/(stem+'.json'),
                   '--resident', resident, '--dynamic', 1024, '--prefill', 1024,
                   '--state-cache-slots', 1, '--mapped-embedding', '--adaptive-owners',
                   '--adapt-every', 32, '--adapt-decay', '.7', '--suite', 'quality', '--rebuild',
                   '--draft-policy', policy, '--keys', 'english', 'chinese', 'code', '70-0', 'diverse-4096-0']
        if mode != 'off':
            command += ['--mtp', '--variant', 'compact-mtp']
        run(stem, command)
    (RUN/'candidate-b-native-quality-complete.json').write_text(json.dumps({'status':'passed'})+'\n')


if __name__ == '__main__':
    main()
