"""Test candidate4 Python changes with explicitly unchanged perf3 native artifacts."""
from pathlib import Path
import hashlib
import json
import sys

SOURCE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate4\build-source')
NATIVE=Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand4\Lib\site-packages\freetoken\kernel')
import freetoken
import freetoken.kernel
import pytest

assert Path(freetoken.__file__).is_relative_to(Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand4\Lib\site-packages'))
print(json.dumps(dict(scope='ordinary_perf4_wheel_no_source_redirect_with_frozen_source_test_files',
    source=str(freetoken.__file__),native_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
    for p in NATIVE.glob('*.pyd')})),flush=True)
paths=[
    'tests/models/qwen4_exp/test_gguf_execution.py::GGUFExecutionConsistencyTests::test_mapped_embedding_preserves_rows_and_changed_graph_inputs',
    'tests/models/qwen4_exp/test_gguf_execution.py::GGUFExecutionConsistencyTests::test_failed_embedding_mapping_keeps_gpu_weights_and_zero_host_budget',
    'tests/moe/test_q2_residency.py',
    'tests/moe/test_q2_adaptive.py',
]
raise SystemExit(pytest.main(['-q','--tb=short',*[str(SOURCE/p) for p in paths]]))
