"""Run frozen source tests against the actual installed sixth wheel and its native extension."""
from pathlib import Path
import hashlib
import json
import sys
import freetoken
import freetoken.kernel
from freetoken.kernel import _row_store
import pytest

SOURCE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-06a\candidate-source\build-source')
INSTALLED=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand6\Lib\site-packages\freetoken')
assert Path(freetoken.__file__).parent.resolve()==INSTALLED.resolve()
assert hashlib.sha256(Path(_row_store.__file__).read_bytes()).hexdigest()=='59d939e443237525efd2a1f64522c9082b886b589eb70eb867fdad7b8207076f'
paths=['tests/models/qwen4_exp/test_mtp_runtime.py','tests/models/qwen4_exp/test_mtp_compact.py','tests/kvcache/test_mtp_pool.py','tests/kvcache/test_linear_state_pool_alloc.py']
status=pytest.main(['-q','--tb=short',*[str(SOURCE/name) for name in paths]])
for name,module in list(sys.modules.items()):
    file=getattr(module,'__file__',None)
    if file and (name=='freetoken' or name.startswith('freetoken.')) and str(file).endswith(('.py','.pyd')):
        assert Path(file).resolve().is_relative_to(INSTALLED.resolve()),(name,file)
print(json.dumps(dict(scope='ordinary_installed_wheel_no_source_redirect',source=str(freetoken.__file__),
    native_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (INSTALLED/'kernel').glob('*.pyd')})))
raise SystemExit(status)
