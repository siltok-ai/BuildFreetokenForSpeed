"""Test candidate Python source with exact frozen dev5 native modules only."""
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

import capture_freetoken_reference
import torch

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'candidate-source-dev6-working/build-source'
NATIVE = Path(sys.prefix) / 'Lib/site-packages/freetoken/kernel'


class NativeBaseFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'freetoken.kernel._pinned_tensor', 'freetoken.kernel._row_store', 'freetoken.kernel._cpu_moe'}:
            files = list(NATIVE.glob(fullname.rsplit('.', 1)[1] + '.*.pyd'))
            if len(files) != 1:
                raise RuntimeError('Native base identity is ambiguous')
            return importlib.util.spec_from_file_location(fullname, files[0])
        return None


sys.meta_path.insert(0, NativeBaseFinder())
sys.path.insert(0, str(SOURCE / 'python'))
import freetoken
if Path(freetoken.__file__).resolve() != SOURCE / 'python/freetoken/__init__.py':
    raise RuntimeError('Candidate source was not selected')
import pytest

code = pytest.main(['-v', str(SOURCE / 'tests/moe/test_q2_residency.py'),
                    str(SOURCE / 'tests/moe/test_q2_grouped.py'),
                    str(SOURCE / 'tests/models/qwen4_exp/test_gguf_execution.py')])
(ROOT / 'dev6-working-source-component-v1.json').write_text(json.dumps(dict(
    returncode=int(code), version=freetoken.__version__, candidate_source=str(SOURCE),
    native_base=str(NATIVE), scope='Working Python source with frozen dev5 native modules; not installed-wheel acceptance',
    passed=code == 0), indent=2), encoding='utf-8')
raise SystemExit(code)
