"""用普通安装review1 wheel执行补丁真实测试，测试源码来自本次冻结重放。"""
from pathlib import Path
import json
import os
import sys

BUILD = Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
SOURCE = Path(r'C:\workspace\build\freetoken-mats\review1-materials-01\candidate-source\build-source')
CACHE = BUILD/'artifacts/strataq2review1-cp312-cu130-sm120/kernel-cache'
os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(CACHE), FREETOKEN_DISABLE_JIT='1',
                  TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-perf-mtp8')
import freetoken
import pytest

installed = Path(sys.prefix)/'Lib/site-packages/freetoken'
if Path(freetoken.__file__).resolve().parent != installed or freetoken.__version__ != '0.1.3+speed.strataq2review1':
    raise RuntimeError(freetoken.__file__)
catalog = json.loads((BUILD/'patches/strataq2review1-series.json').read_text())
tests = sorted({str(SOURCE/name) for patch in catalog['patches'] for name in patch['files']
                if name.startswith('tests/') and name.endswith('.py')})
print('Ordinary wheel:', freetoken.__file__, flush=True)
code = pytest.main(['-q', '--tb=short', '-p', 'no:cacheprovider', *tests])
escaped = {name: module.__file__ for name, module in sys.modules.items()
           if name.startswith('freetoken.') and getattr(module, '__file__', None)
           and not Path(module.__file__).resolve().is_relative_to(installed)}
if escaped:
    raise RuntimeError(escaped)
raise SystemExit(code)
