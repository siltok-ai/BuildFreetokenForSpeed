"""Create an isolated ordinary-wheel environment and export every declared artifact."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

BASE=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand5')
OUT=BASE.with_name('flash-perf-cand6')
MAT=Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-06a')
REPO=Path(r'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2perf6-cp312-cu130-sm120')
record=json.loads((MAT/'build-record.json').read_text(encoding='utf-8'))
expected='fcd0d54792f9a4541cdb9ffcd7a2ca3f4a686fe3dfb1a78baac295ae1038f634'
wheel=MAT/'wheels'/record['wheel_filename']
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==expected==record['wheel_sha256']
if OUT.exists():raise FileExistsError('New environment destination exists')
subprocess.run([r'C:\workspace\tools\python312\python.exe','-m','venv',str(OUT)],check=True)
shutil.copytree(BASE/'Lib/site-packages',OUT/'Lib/site-packages',dirs_exist_ok=True)
shutil.copy2(BASE/'Scripts/ninja.exe',OUT/'Scripts/ninja.exe')
python=OUT/'Scripts/python.exe'
subprocess.run([str(python),'-m','pip','install','--no-deps','--force-reinstall',str(wheel)],check=True)
for relative,digest in record['native_sha256'].items():
    assert hashlib.sha256((OUT/'Lib/site-packages'/relative).read_bytes()).hexdigest()==digest
subprocess.run([str(python),'-c',"import freetoken,importlib.metadata as m;print(freetoken.__file__);print(m.version('freetoken'))"],check=True)
print(json.dumps(dict(environment=str(OUT),wheel_sha256=expected,export=str(REPO))))
