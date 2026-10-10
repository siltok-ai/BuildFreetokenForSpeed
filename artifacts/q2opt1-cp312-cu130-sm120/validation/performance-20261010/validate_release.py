"""顺序验收新冻结 wheel 的 Speed backend、HTTP 质量和旧模型。"""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SPEED = Path('C:/workspace/worktrees/speed-flash-next-q2-opt')
PYTHON = 'C:/workspace/envs/speed-flash-next-q2/Scripts/python.exe'
cache = ROOT / 'materials-final/kernel-cache'
env = os.environ.copy()
env.update(SILITOK_FREETOKEN_RUNTIME_DIR=str(ROOT / 'runtime-release'),
           SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(cache),
           TRITON_CACHE_DIR=str(ROOT / 'triton-cache'),
           COMMIT_SAFETY_GB='1', VRAM_SAFETY_GB='2')
cases = [
    ('flash-backend', SPEED / 'tools/bench/benchmark_freetoken_qwen36.py',
     ['--profile', 'flash-next', '--quant', 'q2_0', '--model',
      'C:/workspace/models/qwen3.8-flash-next.bf16-q2-experts', '--long-prompt-tokens', '4096',
      '--long-repeats', '1', '--repeats', '2', '--tokens', '64', '--quality']),
    ('flash-smoke', SPEED / 'tools/smoke/freetoken_qwen36.py',
     ['--profile', 'flash-next', '--quant', 'q2_0', '--model',
      'C:/workspace/models/qwen3.8-flash-next.bf16-q2-experts', '--commit-safety-gib', '1']),
    ('qwen36-smoke', SPEED / 'tools/smoke/freetoken_qwen36.py',
     ['--profile', 'qwen36', '--quant', 'nvfp4', '--model',
      'C:/workspace/models/qwen3.6-35b-a3b.nvfp4-modelscope']),
    ('qwen38-smoke', SPEED / 'tools/smoke/freetoken_qwen36.py',
     ['--profile', 'qwen38', '--quant', 'nvfp4', '--model',
      'C:/workspace/models/qwen3.8-27b.nvfp4-radixark', '--vram-safety-gib', '0.75']),
]
receipts = []
receipt_path = ROOT / 'release-validation-commands.json'
if receipt_path.exists():
    raise FileExistsError(receipt_path)
for name, script, flags in cases:
    target = ROOT / (name + '.json')
    command = [PYTHON, '-B', '-u', str(ROOT / 'supervise.py'), '--output',
               str(ROOT / (name + '.capacity.json')), '--projected-gib', '50' if name.startswith('flash') else '28', '--',
               PYTHON, '-B', '-u', str(script), *flags, '--kernel-cache', str(cache),
               '--startup-timeout', '600', '--output', str(target)]
    receipt = dict(case=name, command=command, env={k:env[k] for k in (
        'SILITOK_FREETOKEN_RUNTIME_DIR','SILITOK_FREETOKEN_KERNEL_CACHE_DIR','TRITON_CACHE_DIR',
        'COMMIT_SAFETY_GB','VRAM_SAFETY_GB')}, started_unix=time.time())
    receipts.append(receipt)
    receipt_path.write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    print('START ' + name, flush=True)
    done = subprocess.run(command, env=env, cwd=SPEED)
    receipt.update(returncode=done.returncode, finished_unix=time.time())
    receipt_path.write_text(json.dumps(receipts, indent=2), encoding='utf-8')
    if done.returncode:
        raise RuntimeError(f'{name} failed: {done.returncode}')
    print('DONE ' + name, flush=True)
