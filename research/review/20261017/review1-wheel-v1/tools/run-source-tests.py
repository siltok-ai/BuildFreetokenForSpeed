"""对新重放执行当前补丁涉及的真实测试；扩展只借用字节未变的FT8原生模块。"""
from pathlib import Path
import json
import os
import sys

ROOT = Path(r'C:\workspace\build\freetoken-mats\review-20261017-work\build-source')
BUILD = Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
NATIVE = Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand8\Lib\site-packages\freetoken\kernel')
CACHE = Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-08a\kernel-cache')
os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(CACHE), FREETOKEN_DISABLE_JIT='1',
                  TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-review-source',
                  PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0, str(ROOT/'python'))
import freetoken
import freetoken.kernel
freetoken.kernel.__path__.append(str(NATIVE))
import pytest

if Path(freetoken.__file__).resolve().parent != ROOT/'python/freetoken':
    raise RuntimeError(freetoken.__file__)
catalog = json.loads((BUILD/'patches/strataq2perf8-series.json').read_text())
tests = sorted({str(ROOT/name) for patch in catalog['patches'] for name in patch['files']
                if name.startswith('tests/') and name.endswith('.py')})
print(json.dumps(dict(source=str(ROOT), reused_native=str(NATIVE), tests=tests)), flush=True)
raise SystemExit(pytest.main(['-q', '--tb=short', '-p', 'no:cacheprovider', *tests]))
