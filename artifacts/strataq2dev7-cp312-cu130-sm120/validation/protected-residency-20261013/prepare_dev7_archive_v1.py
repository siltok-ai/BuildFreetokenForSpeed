"""Prepare a new evidence archive for the repaired compatibility revision."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
text = (ROOT/'archive_dev6_completed_revision_v1.py').read_text(encoding='utf-8').replace('dev6', 'dev7')
text = text.replace('component_tests=23', 'component_tests=24')
text = text.replace('Raw dev7 state v1 retained: memory_ratio0.8 safely rejected KV rebuild; corrected configuration0.9 passed',
    'Raw dev7 state v2 retained: test expected two tokens from a truncated chat header, which naturally produced EOS; v3 honors EOS and verifies actual16-token prefill')
text = text.replace("{ROOT/'validate_dev7_runtime_gate_v1.py',ROOT/'validate_dev7_installed_state_v1.py'}",
    "{ROOT/'validate_dev7_runtime_gate_v1.py',ROOT/'validate_dev7_installed_state_v2.py'}")
target = ROOT/'archive_dev7_completed_revision_v1.py'
if target.exists():
    raise FileExistsError(target)
target.write_text(text, encoding='utf-8')
print(target)
