"""Wait for the three-mode quality queue, then run full-checkpoint boundary tests serially."""
from pathlib import Path
import json
import os
import runpy
import time

RUN = Path(__file__).parent
protocol = runpy.run_path(str(RUN/'run-native-quality-b.py'))
deadline = time.monotonic()+1800
ready = RUN/'candidate-b-native-quality-complete.json'
while not ready.exists():
    log = RUN/'native-quality-b-driver-v1.log'
    if log.exists() and 'Traceback (most recent call last)' in log.read_text(encoding='utf-8'):
        raise RuntimeError('Earlier GPU quality stage failed; boundary queue stopped')
    if time.monotonic() >= deadline:
        raise TimeoutError('Earlier GPU quality stage did not finish')
    time.sleep(2)
if json.loads(ready.read_text())['status'] != 'passed':
    raise RuntimeError('Earlier GPU quality stage not accepted')
os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(protocol['CACHE']), FREETOKEN_DISABLE_JIT='1',
                  TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-review2b', PYTHONUTF8='1',
                  PYTHONDONTWRITEBYTECODE='1')
os.environ.pop('PYTHONPATH', None)
for mode in ('off', 'on'):
    stem = f'candidate-b-model-boundaries-{mode}-v1'
    command = [protocol['PYTHON'], '-B', RUN/'probe-model-boundaries-b.py', '--output', RUN/(stem+'.json')]
    if mode == 'on':
        command.append('--mtp')
    protocol['run'](stem, command)
(RUN/'candidate-b-model-boundaries-complete.json').write_text(json.dumps({'status':'passed'})+'\n')
