"""Run the selected patch tests against explicit source or an ordinary installed wheel."""
from pathlib import Path
import argparse
import json
import os
import sys

RUN = Path(__file__).parent
BUILD = Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
SOURCE = Path(r'C:\workspace\build\freetoken-mats\review2a-materials-01\candidate-source\build-source')
CACHE = BUILD/'artifacts/strataq2review2a-cp312-cu130-sm120/kernel-cache'
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path)
args = parser.parse_args()
os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(CACHE), FREETOKEN_DISABLE_JIT='1',
                  TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-review2a', PYTHONDONTWRITEBYTECODE='1')
if args.source:
    sys.path.insert(0, str(args.source/'python'))
import freetoken
import freetoken.kernel
installed = Path(sys.prefix)/'Lib/site-packages/freetoken'
expected = args.source/'python/freetoken' if args.source else installed
if Path(freetoken.__file__).resolve().parent != expected:
    raise RuntimeError(freetoken.__file__)
if args.source:
    freetoken.kernel.__path__.append(str(installed/'kernel'))
if freetoken.__version__ != '0.1.3+speed.strataq2review2a':
    raise ValueError(freetoken.__version__)
catalog = json.loads((BUILD/'patches/strataq2review2a-series.json').read_text())
tests = sorted({str(SOURCE/name) for patch in catalog['patches'] for name in patch['files']
                if name.startswith('tests/') and name.endswith('.py')})
print(json.dumps(dict(source=freetoken.__file__, python=sys.executable, tests=tests,
                      reused_native=str(installed/'kernel') if args.source else None)), flush=True)
import pytest
code = pytest.main(['-q', '--tb=short', '-rs', '-p', 'no:cacheprovider', *tests])
if not args.source:
    escaped = {name: module.__file__ for name, module in sys.modules.items()
               if name.startswith('freetoken.') and getattr(module, '__file__', None)
               and not Path(module.__file__).resolve().is_relative_to(installed)}
    if escaped:
        raise RuntimeError(escaped)
raise SystemExit(code)
