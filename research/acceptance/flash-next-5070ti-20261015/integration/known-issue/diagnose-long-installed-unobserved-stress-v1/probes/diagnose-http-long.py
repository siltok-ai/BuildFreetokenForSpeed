"""在安装验收的同一环境中对照源码和安装版，不改动产品及失败报告。"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess

RUN = Path(__file__).parent
SOURCE = Path(r'C:\workspace\wt\siltok-speed\flash-accept')
PYTHON = Path(r'C:\workspace\envs\siltok-speed\flash-accept\Scripts\python.exe')
INSTALL = Path(r'C:\workspace\build\siltok-speed\i03')


def main():
    """复用原始请求和资源限制，仅切换被测产品形态并使用新证据路径。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['source', 'installed'], required=True)
    parser.add_argument('--id', required=True)
    parser.add_argument('--numeric', action='store_true')
    parser.add_argument('--stress-rounds', type=int)
    parser.add_argument('--poison-empty', action='store_true')
    parser.add_argument('--ple-sync', choices=['wait', 'gate'])
    args = parser.parse_args()
    target = RUN / ('diagnose-long-' + args.id)
    target.mkdir()
    prior = RUN / 'install-i03/results-v2/http-off-functional-v1.origin.json'
    origin = json.loads(prior.read_text(encoding='utf-8'))
    forwarded = origin['arguments']
    if args.mode == 'source':
        forwarded.remove('--packaged')
    for option, name in [('--output', 'report.json'), ('--lifecycle-output', 'lifecycle.json')]:
        forwarded[forwarded.index(option) + 1] = str(target / name)
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('SILITOK_', 'FREETOKEN_')) and key not in
                   ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME', 'CUDA_PATH', 'CUDA_HOME')}
    environment.update(json.loads((prior.parent / 'probe-environment.json').read_text()))
    environment.update(PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1',
                       FREETOKEN_DISABLE_JIT='1', HF_HOME=r'C:\workspace\cache\hf')
    environment['TMP'] = environment['TEMP']
    if args.ple_sync:
        environment['FREETOKEN_PLE_SYNC'] = args.ple_sync
    command = [str(PYTHON), '-u', '-B', str(RUN / 'supervise.py'), '--prefix', str(target / 'run'),
               '--cwd', str(INSTALL), '--timeout', '1500', '--']
    if args.mode == 'installed':
        command += [str(INSTALL / 'python/python.exe'), '-I', '-X', 'utf8', '-u', '-B',
                    str(RUN / 'run-installed-probe.py'), '--install-root', str(INSTALL),
                    '--expected-commit', origin['build_commit'], '--origin-report', str(target / 'origin.json'),
                    '--script', str(RUN / 'run-http-smoke-v2.py')]
    else:
        command += [str(PYTHON), '-I', '-X', 'utf8', '-u', '-B', str(RUN / 'run-http-smoke-v2.py')]
    command += forwarded
    if args.stress_rounds:
        command = [str(RUN / 'run-http-long-stress.py') if part == str(RUN / 'run-http-smoke-v2.py')
                   else part for part in command]
        command += ['--diagnostic-rounds', str(args.stress_rounds)]
    if args.numeric or args.poison_empty:
        script_index = command.index(str(RUN / ('run-installed-probe.py' if args.mode == 'installed' else
            'run-http-long-stress.py' if args.stress_rounds else 'run-http-smoke-v2.py')))
        original_script = command[script_index]
        command[script_index:script_index + 1] = [str(RUN / 'numeric-launch.py'),
            '--diagnostic-output', str(target), '--target-script', original_script]
        if args.poison_empty:
            command[script_index + 1:script_index + 1] = ['--hook-file', str(RUN / 'worker-poison-hook.py')]
    snapshots = target / 'probes'
    snapshots.mkdir()
    scripts = ['diagnose-http-long.py', 'run-http-long-stress.py', 'run-http-smoke-v2.py',
               'numeric-launch.py', 'worker-numeric-hook.py', 'worker-poison-hook.py',
               'run-installed-probe.py', 'supervise.py']
    replacements = {}
    for name in scripts:
        shutil.copy2(RUN / name, snapshots / name)
        replacements[str(RUN / name)] = str(snapshots / name)
    command = [replacements.get(part, part) for part in command]
    (target / 'launch.json').write_text(json.dumps(dict(mode=args.mode, command=command,
        ple_sync=args.ple_sync, poison_empty=args.poison_empty,
        diagnostic_scripts={name: hashlib.sha256((snapshots / name).read_bytes()).hexdigest()
                            for name in scripts},
        environment={key: environment[key] for key in json.loads((prior.parent / 'probe-environment.json').read_text())}),
        indent=2), encoding='utf-8')
    result = subprocess.run(command, env=environment, creationflags=subprocess.CREATE_NO_WINDOW)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
