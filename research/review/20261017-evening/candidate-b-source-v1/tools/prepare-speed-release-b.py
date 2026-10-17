"""Build and audit an ordinary Speed wheel, then install it into an independent full runtime."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time
import zipfile

RUN = Path(__file__).parent
SPEED = Path(r'C:\workspace\wt\siltok-speed\flash-perf')
BASE = Path(r'C:\workspace\envs\siltok-speed\flash-review2b')
TARGET = BASE.with_name('flash-review2b-release')
OLD_WHEEL, = Path(r'C:\workspace\runs\siltok-speed\review-20261017\speed-ordinary-wheel-v1').glob('*.whl')


def main():
    """Wait for GPU/timing work; verify actual packaged bytes before preparing a new release venv."""
    ready = RUN/'candidate-b-source-all-complete.json'
    deadline = time.monotonic()+3600
    while not ready.exists():
        log = (RUN/'source-followup-b-driver-v1.log').read_text(encoding='utf-8')
        if 'Traceback (most recent call last)' in log:
            raise RuntimeError('Source followup failed; release preparation stopped')
        if time.monotonic() >= deadline:
            raise TimeoutError('Source acceptance did not complete')
        time.sleep(2)
    staging, wheels = RUN/'speed-ordinary-source-v1', RUN/'speed-ordinary-wheel-v1'
    if staging.exists() or wheels.exists() or TARGET.exists():
        raise FileExistsError('Fresh wheel and environment outputs required')
    shutil.copytree(SPEED/'silitok_speed', staging,
        ignore=shutil.ignore_patterns('build','.artifacts','__pycache__','.pytest_cache','*.egg-info'))
    egg = staging/'siltok_speed.egg-info'
    egg.mkdir()
    (egg/'SOURCES.txt').write_text('\n'.join(p.relative_to(staging).as_posix() for p in staging.rglob('*') if p.is_file())+'\n')
    subprocess.run([str(BASE/'Scripts/python.exe'),'-B','-m','pip','wheel','--no-deps','--no-build-isolation',
                    '--wheel-dir',str(wheels),str(staging)], check=True)
    wheel, = wheels.glob('*.whl')
    with zipfile.ZipFile(wheel) as archive, zipfile.ZipFile(OLD_WHEEL) as old:
        names, older = set(archive.namelist()), set(old.namelist())
        dlls = [name for name in names if '/qwen36/kernel_cache' in name and name.endswith('.dll')]
        if len(dlls) != 27 or not all('kernel_cache_strataq2review2b/' in name for name in dlls):
            raise ValueError('Ordinary wheel did not select exactly the candidate cache')
        def assets(items):
            return {name for name in items if name.startswith('silitok_speed/')
                    and not name.endswith(('.py','/')) and '/qwen36/kernel_cache' not in name}
        lost = sorted(assets(older)-assets(names))
        if lost:
            raise ValueError('Non-cache resources lost: '+repr(lost))
        python_files = {name:hashlib.sha256(archive.read(name)).hexdigest() for name in names
                        if name.startswith('silitok_speed/') and name.endswith('.py')}
        for name, expected in python_files.items():
            if hashlib.sha256((SPEED/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Ordinary wheel differs from tested source: '+name)
        release = json.loads(archive.read('silitok_speed/qwen36/release.json'))
        if release['version'] != '0.1.3+speed.strataq2review2b':
            raise ValueError('Wrong runtime in ordinary product wheel')
        inventory = {name:hashlib.sha256(archive.read(name)).hexdigest() for name in sorted(names)}
    subprocess.run([r'C:\workspace\tools\python312\python.exe','-m','venv',str(TARGET)],check=True)
    shutil.copytree(BASE/'Lib/site-packages', TARGET/'Lib/site-packages', dirs_exist_ok=True)
    shutil.copy2(BASE/'Scripts/ninja.exe', TARGET/'Scripts/ninja.exe')
    subprocess.run([str(TARGET/'Scripts/python.exe'),'-B','-m','pip','install','--no-index','--no-deps',
                    '--force-reinstall',str(wheel)],check=True)
    report = dict(status='passed',scope='ordinary_speed_wheel_inventory_not_compiled_or_installer',
        wheel=str(wheel),wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),wheel_bytes=wheel.stat().st_size,
        baseline_wheel=str(OLD_WHEEL),kernel_dll_count=len(dlls),lost_non_cache_assets=lost,
        python_files=python_files,files=inventory,release_runtime=str(TARGET))
    (RUN/'speed-ordinary-wheel-inventory-v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('files','python_files')}),flush=True)
    (RUN/'candidate-b-release-env-ready.json').write_text(json.dumps({'status':'passed','environment':str(TARGET)})+'\n')


if __name__ == '__main__':
    main()
