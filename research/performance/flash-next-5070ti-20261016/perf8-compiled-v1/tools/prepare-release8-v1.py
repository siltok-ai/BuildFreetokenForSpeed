"""Prepare a separate ordinary runtime for the final protected-package build."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

R=Path(__file__).parent
S=Path(r'C:\workspace\wt\siltok-speed\flash-perf')
BASE=Path(r'C:\workspace\envs\siltok-speed')
SOURCE=BASE/'flash-perf-cand8'
TARGET=BASE/'flash-perf-rel8'
P=SOURCE/'Scripts/python.exe'

def main():
    """Preserve earlier environments and bind both ordinary packages in a fresh venv."""
    deadline=time.monotonic()+1200
    while not (R/'speed8-ownership-review-ready-v2.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Ownership fix review unfinished')
        time.sleep(2)
    if TARGET.exists():raise FileExistsError(TARGET)
    wheels=R/'speed8-ordinary-wheel-v1'
    if wheels.exists():raise FileExistsError(wheels)
    # Keep source acceptance immutable: setuptools must build outside the measured worktree.
    staging=R/'speed8-ordinary-source-v1'
    if staging.exists():raise FileExistsError(staging)
    shutil.copytree(S/'silitok_speed',staging,ignore=shutil.ignore_patterns('build','.artifacts','__pycache__','.pytest_cache','*.egg-info'))
    subprocess.run([str(P),'-B','-m','pip','wheel','--no-deps','--no-build-isolation',
        '--wheel-dir',str(wheels),str(staging)],check=True)
    wheel,=wheels.glob('*.whl')
    subprocess.run([r'C:\workspace\tools\python312\python.exe','-m','venv',str(TARGET)],check=True)
    shutil.copytree(SOURCE/'Lib/site-packages',TARGET/'Lib/site-packages',dirs_exist_ok=True)
    shutil.copy2(SOURCE/'Scripts/ninja.exe',TARGET/'Scripts/ninja.exe')
    py=TARGET/'Scripts/python.exe'
    subprocess.run([str(py),'-B','-m','pip','install','--no-deps','--force-reinstall',str(wheel)],check=True)
    subprocess.run([str(py),'-B','-m','silitok_speed.qwen36.freetoken_release','self-check',
        '--kernel-cache',str(S/'silitok_speed/qwen36/kernel_cache_strataq2perf8'),'--pip-check',
        '--constraint-policy',str(S/'packaging/windows/runtime-requirements.toml')],check=True)
    payload=dict(status='prepared',runtime=str(TARGET),speed_wheel=str(wheel),
        speed_wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        source_commit=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()))
    assert not payload['source_dirty'],payload
    (R/'release8-runtime-v1.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload),flush=True)

if __name__=='__main__':main()
