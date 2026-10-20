"""使用保护包自带解释器和 AOT，在隔离环境中消费完整资源包。"""
from pathlib import Path
import argparse
import json
import os
import subprocess

R = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument('--app', type=Path, required=True)
p.add_argument('--tag', required=True)
p.add_argument('--installed', action='store_true')
args = p.parse_args()
app = args.app.resolve()
root = R/'install-i09' if args.installed else R/'protected-environment-v1'
root.mkdir(exist_ok=True)
for name in ('tmp','config'):
    (root/name).mkdir(exist_ok=True)
(root/'config/config.yaml').write_text('{"resource":{"gpu_probe":"nvml"}}')
cache_tag = 'installed-i09' if args.installed else 'portable1-protected'
env = {k:v for k,v in os.environ.items()
       if not k.startswith(('SILITOK_','FREETOKEN_','CUDA_','TORCH_'))
       and k not in ('VIRTUAL_ENV','PYTHONPATH','PYTHONHOME','INCLUDE','LIB','LIBPATH',
                    'CUDACXX','CMAKE_PREFIX_PATH','CC','CXX','NVCC')}
env.update(PATH=';'.join([str(app/'python/Scripts'),r'C:\Windows\System32',r'C:\Windows',
    r'C:\Program Files\NVIDIA Corporation\NVSMI']), TEMP=str(root/'tmp'),TMP=str(root/'tmp'),
    PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',SILITOK_CONFIG=str(root/'config'),
    SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(app/'runtime/kernel-cache'),
    FREETOKEN_KERNEL_CACHE_DIR=str(app/'runtime/kernel-cache'),FREETOKEN_DISABLE_JIT='1',
    TRITON_CACHE_DIR=rf'C:\workspace\cache\triton\{cache_tag}',
    TORCH_EXTENSIONS_DIR=rf'C:\workspace\cache\ext\{cache_tag}')
(R/(args.tag+'-environment.json')).write_text(json.dumps(dict(app=str(app),
    python=str(app/'python/python.exe'),environment={k:env[k] for k in
    ('PATH','TEMP','SILITOK_CONFIG','SILITOK_FREETOKEN_KERNEL_CACHE_DIR',
     'FREETOKEN_DISABLE_JIT','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR')}),indent=2)+'\n')
try:
    with (R/(args.tag+'-models.log')).open('wb') as log:
        subprocess.run([str(app/'python/python.exe'),'-I','-X','utf8','-u','-B',
            str(R/'http-portable-probe.py'),'--models-dir',r'C:\workspace\flash-next-resources',
            '--tag',args.tag],env=env,cwd=app,stdout=log,stderr=subprocess.STDOUT,
            check=True,timeout=1800,creationflags=subprocess.CREATE_NO_WINDOW)
    if args.installed:
        (root/'suite-progress.json').write_text(json.dumps(dict(status='passed',
            scope='actual_installed_payload_not_clean_os',tag=args.tag))+'\n')
except BaseException:
    if args.installed:
        (root/'suite-progress.json').write_text(json.dumps(dict(status='failed',
            scope='actual_installed_payload_not_clean_os',tag=args.tag))+'\n')
    raise
print(args.tag+': both MTP modes passed',flush=True)
