"""构建普通Speed wheel，交叉核对旧wheel资源，再准备独立发布环境。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time
import zipfile
R=Path(__file__).parent
S=Path(r'C:\workspace\wt\siltok-speed\flash-perf')
PY=Path(r'C:\workspace\envs\siltok-speed\flash-review1\Scripts\python.exe')
OLD=Path(r'C:\workspace\runs\siltok-speed\flash-perf-20261016\speed8-ordinary-wheel-v1\siltok_speed-0.1.2.dev0+strataq2perf8-py3-none-any.whl')
TARGET=Path(r'C:\workspace\envs\siltok-speed\flash-review1-release')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    deadline=time.monotonic()+3600
    while not (R/'source-acceptance-complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('Source GPU acceptance unfinished')
        time.sleep(2)
    staging=R/'speed-ordinary-source-v1';wheels=R/'speed-ordinary-wheel-v1'
    if staging.exists() or wheels.exists() or TARGET.exists():raise FileExistsError('Fresh outputs required')
    shutil.copytree(S/'silitok_speed',staging,ignore=shutil.ignore_patterns('build','.artifacts','__pycache__','.pytest_cache','*.egg-info'))
    # 模拟旧构建的SOURCES清单仍列有历史缓存，显式package-data不得受其污染。
    egg=staging/'siltok_speed.egg-info';egg.mkdir()
    (egg/'SOURCES.txt').write_text('\n'.join(p.relative_to(staging).as_posix() for p in staging.rglob('*') if p.is_file())+'\n',encoding='utf-8')
    subprocess.run([str(PY),'-B','-m','pip','wheel','--no-deps','--no-build-isolation','--wheel-dir',str(wheels),str(staging)],check=True)
    wheel,=wheels.glob('*.whl')
    with zipfile.ZipFile(wheel) as z,zipfile.ZipFile(OLD) as old:
        names=set(z.namelist());older=set(old.namelist())
        dlls=sorted(n for n in names if n.startswith('silitok_speed/qwen36/kernel_cache') and n.endswith('.dll'))
        old_dlls=sorted(n for n in older if n.startswith('silitok_speed/qwen36/kernel_cache') and n.endswith('.dll'))
        if len(dlls)!=27 or not all('kernel_cache_strataq2review1/' in n for n in dlls):raise RuntimeError(dlls)
        def assets(items):return {n for n in items if n.startswith('silitok_speed/') and not n.endswith(('.py','/')) and '/qwen36/kernel_cache' not in n}
        lost=sorted(assets(older)-assets(names));added=sorted(assets(names)-assets(older))
        if lost:raise RuntimeError('Non-cache assets lost: '+repr(lost))
        assert 'silitok_speed/qwen36/freetoken_options.py' in names
        release=json.loads(z.read('silitok_speed/qwen36/release.json'))
        assert release['version']=='0.1.3+speed.strataq2review1'
        inventory={n:hashlib.sha256(z.read(n)).hexdigest() for n in sorted(names)}
    subprocess.run([r'C:\workspace\tools\python312\python.exe','-m','venv',str(TARGET)],check=True)
    base=PY.parent.parent
    shutil.copytree(base/'Lib/site-packages',TARGET/'Lib/site-packages',dirs_exist_ok=True)
    shutil.copy2(base/'Scripts/ninja.exe',TARGET/'Scripts/ninja.exe')
    python=TARGET/'Scripts/python.exe'
    subprocess.run([str(python),'-B','-m','pip','install','--no-index','--no-deps','--force-reinstall',str(wheel)],check=True)
    report=dict(status='passed',wheel=str(wheel),wheel_sha256=sha(wheel),wheel_bytes=wheel.stat().st_size,baseline_wheel_bytes=OLD.stat().st_size,kernel_dll_count=len(dlls),baseline_kernel_dll_count=len(old_dlls),lost_non_cache_assets=lost,added_assets=added,files=inventory,release_runtime=str(TARGET),scope='ordinary_speed_wheel_inventory_not_compiled_or_installer')
    (R/'speed-ordinary-wheel-inventory-v1.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)
if __name__=='__main__':main()
