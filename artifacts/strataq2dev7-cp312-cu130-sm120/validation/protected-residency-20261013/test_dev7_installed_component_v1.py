import capture_freetoken_reference
import freetoken
from pathlib import Path
import pytest
assert freetoken.__version__ == '0.1.3+speed.strataq2dev7'
root=Path(__file__).resolve().parent / 'dev7-frozen-replay-v1/build-source/tests'
raise SystemExit(pytest.main(['-v',str(root/'moe/test_q2_residency.py'),str(root/'moe/test_q2_grouped.py'),str(root/'models/qwen4_exp/test_gguf_execution.py')]))
