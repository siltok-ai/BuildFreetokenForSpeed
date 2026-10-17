"""在独立venv普通安装review1，导出声明制品；不修改FT8环境或已验收材料。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

BASE = Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand8')
TARGET = BASE.with_name('flash-review1')
MATERIALS = Path(r'C:\workspace\build\freetoken-mats\review1-materials-01')
EXPORT = Path(r'C:\workspace\wt\freetoken-mats\flash-perf\artifacts\strataq2review1-cp312-cu130-sm120')
RUN = Path(__file__).parent


def main():
    """独立复制依赖和RECORD拥有的ninja原生工具，再安装精确新wheel。"""
    record = json.loads((MATERIALS/'build-record.json').read_text())
    if TARGET.exists() or EXPORT.exists():
        raise FileExistsError('Fresh environment and artifact destinations required')
    wheel = MATERIALS/'wheels'/record['wheel_filename']
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != record['wheel_sha256']:
        raise ValueError('Candidate wheel changed')
    subprocess.run([r'C:\workspace\tools\python312\python.exe', '-m', 'venv', str(TARGET)], check=True)
    shutil.copytree(BASE/'Lib/site-packages', TARGET/'Lib/site-packages', dirs_exist_ok=True)
    shutil.copy2(BASE/'Scripts/ninja.exe', TARGET/'Scripts/ninja.exe')
    python = TARGET/'Scripts/python.exe'
    subprocess.run([str(python), '-B', '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--force-reinstall', str(wheel)], check=True)
    for name, digest in record['binary_reuse']['native_sha256'].items():
        if hashlib.sha256((TARGET/'Lib/site-packages'/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Installed native bytes changed: '+name)
    for name in [*record['artifacts'], 'build-record.json']:
        source, destination = MATERIALS/name, EXPORT/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    result = dict(status='prepared_not_runtime_accepted', environment=str(TARGET), materials=str(EXPORT),
                  wheel_sha256=record['wheel_sha256'])
    (RUN/'review-env-ready.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
