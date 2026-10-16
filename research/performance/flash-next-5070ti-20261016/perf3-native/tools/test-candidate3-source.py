"""在新Python源码上运行GPU组件，3个native模块与27AOT显式沿用相同输入。"""
from pathlib import Path
import hashlib
import json
import sys

SOURCE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate3\build-source')
NATIVE=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand2\Lib\site-packages\freetoken\kernel')
sys.path.insert(0,str(SOURCE/'python'))
import freetoken
import freetoken.kernel
freetoken.kernel.__path__.append(str(NATIVE))
import pytest


def main():
    """源码与原生身份明示；本组件运行不能冒充普通新wheel或产品验证。"""
    assert Path(freetoken.__file__).is_relative_to(SOURCE)
    print(json.dumps(dict(scope='candidate3_python_source_with_unchanged_perf2_native_inputs',
        source=str(freetoken.__file__),native_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                                   for p in NATIVE.glob('*.pyd')})),flush=True)
    paths=[
        'tests/models/qwen4_exp/test_mtp_compact.py',
        'tests/moe/test_q2_adaptive.py',
        'tests/moe/test_q2_q8_tiles.py',
        'tests/moe/test_q2_residency.py',
        'tests/models/qwen4_exp/test_gguf_gdn.py',
        'tests/models/qwen4_exp/test_mtp_runtime.py',
        'tests/kvcache/test_linear_state_pool_alloc.py',
    ]
    raise SystemExit(pytest.main(['-q','--tb=short',*[str(SOURCE/p) for p in paths]]))


if __name__=='__main__':main()
