"""Create an independent ordinary-wheel environment and preserve prior dependency environments."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

base=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand3')
output=base.with_name('flash-perf-cand4')
materials=Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-04')
repository=Path(r'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2perf4-cp312-cu130-sm120')
wheel=materials/'wheels/freetoken-0.1.3+speed.strataq2perf4-cp312-cp312-win_amd64.whl'
expected='3b83b9fc79e8427663b775f27516d9acb13b55403b6b9fce0c202b90b363c5bc'
assert hashlib.sha256(wheel.read_bytes()).hexdigest()==expected
if output.exists() or repository.exists():raise FileExistsError('New environment/material destination exists')
subprocess.run([r'C:\workspace\tools\python312\python.exe','-m','venv',str(output)],check=True)
shutil.copytree(base/'Lib/site-packages',output/'Lib/site-packages',dirs_exist_ok=True)
shutil.copy2(base/'Scripts/ninja.exe',output/'Scripts/ninja.exe')
python=output/'Scripts/python.exe'
subprocess.run([str(python),'-m','pip','install','--no-deps','--force-reinstall',str(wheel)],check=True)
subprocess.run([str(python),'-c',"import freetoken,importlib.metadata as m;print(freetoken.__file__);print(m.version('freetoken'))"],check=True)
repository.mkdir(parents=True)
for name in ('wheels','kernel-cache'):shutil.copytree(materials/name,repository/name)
shutil.copy2(materials/'build-record.json',repository/'build-record.json')
print(json.dumps(dict(environment=str(output),wheel_sha256=expected,repository_materials=str(repository))))
