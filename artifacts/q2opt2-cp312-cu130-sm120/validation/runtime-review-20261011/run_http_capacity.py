"""Isolated packaged backend acceptance with measured commit and a safety stop."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

sys.path.insert(0, r'C:\workspace\worktrees\build-freetoken-q2-material-review\tools')
from monitor_windows_commit import snapshot

root = Path(__file__).parent / 'packaged-flash-http-49p5-002'
root.mkdir(exist_ok=False)
app = Path(r'C:\workspace\build\q2-review-20261011\app')
source = Path(r'C:\workspace\worktrees\speed-q2-material-review')
env = os.environ.copy()
for name in list(env):
    if name in {'CUDA_HOME', 'CUDA_PATH', 'PYTHONHOME', 'PYTHONPATH', 'VIRTUAL_ENV',
                'FREETOKEN_KERNEL_CACHE_DIR', 'SILITOK_FREETOKEN_RUNTIME_DIR',
                'SILITOK_FREETOKEN_KERNEL_CACHE_DIR', 'FREETOKEN_ALLOW_INVALID_MODEL',
                'FREETOKEN_INVALID_EXPERT_KEEP_LAYERS'} or name.startswith('CUDA_PATH_V'):
        env.pop(name)
system = Path(os.environ['SystemRoot'])
env['PATH'] = os.pathsep.join(map(str, (app/'python', app/'python/Scripts',
                                      system/'System32', system,
                                      system/'System32/WindowsPowerShell/v1.0')))
env.update(SPEED_REVIEW_APP=str(app),
           SPEED_REVIEW_IDENTITY_PATH=str(root/'identity.json'),
           PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
command = [str(app/'python/python.exe'), '-X', 'utf8', '-I', '-B',
           r'C:\workspace\worktrees\build-freetoken-q2-material-review\tools\packaged_speed_child.py',
           str(source/'tools/smoke/freetoken_qwen36.py'),
           '--profile', 'flash-next', '--quant', 'q2_0',
           '--model', r'C:\workspace\artifacts\freetoken-q2-review-20261011\checkpoint-finalize',
           '--kernel-cache', str(app/'runtime/kernel-cache'), '--startup-timeout', '600',
           '--host-commit-estimate-gib', '35.5', '--vram-safety-gib', '2',
           '--commit-safety-gib', '1', '--output', str(root/'report.json')]
baseline = snapshot()
if baseline['commit_limit'] - baseline['commit_used'] < 50.5 * 1024**3:
    raise RuntimeError('Insufficient initial commit for the 49.5 GiB experimental budget plus 1 GiB margin')
samples = []
started = time.perf_counter()
aborted = False
with (root/'run.log').open('w', encoding='utf-8') as log:
    child = subprocess.Popen(command, cwd=app, env=env, stdout=log, stderr=subprocess.STDOUT)
    process = psutil.Process(child.pid)
    while child.poll() is None:
        item = snapshot()
        item['seconds'] = time.perf_counter() - started
        item['processes'] = []
        try:
            family = [process, *process.children(recursive=True)]
        except psutil.NoSuchProcess:
            family = []
        for member in family:
            try:
                info = member.memory_info()
                item['processes'].append({'pid': member.pid, 'rss': info.rss, 'private': info.private})
            except psutil.NoSuchProcess:
                pass
        samples.append(item)
        if item['commit_limit'] - item['commit_used'] < 1024**3:
            aborted = True
            for member in reversed(family):
                try:
                    member.kill()
                except psutil.NoSuchProcess:
                    pass
            break
        time.sleep(.1)
    child.wait()
result = {'scope': 'packaged HTTP; 35.5 GiB host + 14 GiB GPU estimate, strict admission + 1 GiB safety',
          'command': command, 'baseline': baseline, 'returncode': child.returncode,
          'safety_abort': aborted, 'safety_margin_bytes': 1024**3,
          'sampling_interval_seconds': .1, 'samples': samples,
          'wall_seconds': time.perf_counter()-started,
          'peak_commit_used': max(v['commit_used'] for v in samples),
          'min_commit_available': min(v['commit_limit']-v['commit_used'] for v in samples),
          'min_physical_available': min(v['physical_available'] for v in samples)}
(root/'capacity.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in {'samples', 'command'}}), flush=True)
if child.returncode or aborted:
    raise SystemExit(child.returncode or 1)
from verify_packaged_speed import check_report
check_report(json.loads((root/'report.json').read_text(encoding='utf-8')), 'http')
print('PASS: compiled package, complete Flash Next HTTP with 49.5 GiB estimated commit')
