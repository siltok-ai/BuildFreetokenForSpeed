"""验证普通perf3 wheel的真实来源后，对安装的实现运行冻结组件测试。"""
from pathlib import Path
import hashlib
import json
import sys
import zipfile
import freetoken
import pytest

RUN=Path(__file__).parent
ENV=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand3')
MATERIALS=Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-03')
SOURCE=MATERIALS/'candidate-source/build-source'


def main():
    """不更改sys.path或模块方法；核对wheel内全部Python/native文件字节。"""
    package=Path(freetoken.__file__).parent
    assert package.is_relative_to(ENV/'Lib/site-packages')
    record=json.loads((MATERIALS/'build-record.json').read_text())
    wheel=MATERIALS/'wheels'/record['wheel_filename']
    inventory={}
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('freetoken/') and name.endswith(('.py','.pyd')):
                data=(package.parent/name).read_bytes()
                assert data==archive.read(name),name
                inventory[name]=hashlib.sha256(data).hexdigest()
    print(json.dumps(dict(scope='ordinary_freetoken_wheel_not_windows_installer',
        source=str(package),version=record['custom_version'],verified_files=len(inventory))),flush=True)
    paths=['tests/models/qwen4_exp/test_mtp_compact.py','tests/moe/test_q2_adaptive.py',
           'tests/moe/test_q2_q8_tiles.py','tests/moe/test_q2_residency.py',
           'tests/models/qwen4_exp/test_gguf_gdn.py','tests/models/qwen4_exp/test_mtp_runtime.py',
           'tests/kvcache/test_linear_state_pool_alloc.py']
    code=pytest.main(['-q','--tb=short',*[str(SOURCE/p) for p in paths]])
    loaded={name:str(Path(module.__file__).resolve()) for name,module in sys.modules.items()
            if name.startswith('freetoken.') and getattr(module,'__file__',None)}
    assert all(Path(path).is_relative_to(package) for path in loaded.values())
    (RUN/'candidate3-wheel-components-v1.json').write_text(json.dumps(dict(
        scope='ordinary_freetoken_wheel_not_windows_installer',status='passed' if code==0 else 'failed',
        files=inventory,loaded=loaded,wheel_sha256=record['wheel_sha256']),indent=2),encoding='utf-8')
    raise SystemExit(code)


if __name__=='__main__':main()
