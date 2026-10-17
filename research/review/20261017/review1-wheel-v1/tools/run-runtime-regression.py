"""用同一个真实文件替换测试对照补丁前后的MTP加载实现。"""
from pathlib import Path
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source/'python'))
import freetoken
import pytest

print('FreeToken source:', freetoken.__file__, flush=True)
test = (Path(r'C:\workspace\build\freetoken-mats\review-20261017-work\build-source') /
        'tests/models/qwen4_exp/test_mtp_runtime.py')
raise SystemExit(pytest.main(['-q', '--tb=short', '-p', 'no:cacheprovider',
    str(test)+'::test_drafter_checks_the_bytes_it_actually_loads']))
