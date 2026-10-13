"""Freeze the sparse-prefill compatibility repair without replacing dev6."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PRODUCER = Path('C:/workspace/worktrees/build-freetoken-flash-next-strata-q2')
text = (ROOT / 'freeze_dev6_protected_materials_v1.py').read_text(encoding='utf-8')
text = text.replace("BASE = ROOT / 'dev5-derived-native-base-v1/candidate-source/build-source'",
                    "BASE = ROOT / 'dev6-frozen-replay-v1/build-source'")
text = text.replace('strataq2dev5', 'strataq2dev6').replace('dev6', 'dev7')
# The previous replacement also shifts the immutable base references; restore them.
text = text.replace("BASE = ROOT / 'dev7-frozen-replay-v1/build-source'",
                    "BASE = ROOT / 'dev6-frozen-replay-v1/build-source'")
text = text.replace("PRODUCER / 'patches/strataq2dev7-series.json').read_text()",
                    "PRODUCER / 'patches/strataq2dev6-series.json').read_text()")
text = text.replace("PRODUCER / 'build/profiles/strataq2dev7.json').read_text()",
                    "PRODUCER / 'build/profiles/strataq2dev6.json').read_text()")
start = text.index('    expected = {')
end = text.index('    for new in ', start)
text = text[:start] + "    expected = {'python/freetoken/version.py', 'python/freetoken/moe/q2_residency.py', 'tests/moe/test_q2_residency.py'}\n" + text[end:]
text = text.replace('0012-protected-q2-owners', '0013-sparse-prefill-admission')
text = text.replace('FTSTRATA-006', 'FTSTRATA-007')
text = text.replace('Permanent compressed Q2 GPU owners with compact pinned complement and disjoint dynamic/prefill regions',
                    'Admit the existing 16-token sparse-prefill path with 160 routed experts')
path = ROOT / 'freeze_dev7_sparse_materials_v1.py'
if path.exists():
    raise FileExistsError(path)
path.write_text(text, encoding='utf-8')
subprocess.run([sys.executable, '-B', str(path)], check=True)
subprocess.run([sys.executable, '-B', str(PRODUCER / 'tools/prepare_source.py'),
    '--profile', str(PRODUCER / 'build/profiles/strataq2dev7.json'),
    '--output', str(ROOT / 'dev7-frozen-replay-v1')], check=True)
subprocess.run([sys.executable, '-B', str(PRODUCER / 'build/derive_python_materials.py'),
    '--profile', str(PRODUCER / 'build/profiles/strataq2dev7.json'),
    '--base-profile', str(PRODUCER / 'build/profiles/strataq2dev4.json'),
    '--base-materials', str(PRODUCER / 'artifacts/strataq2dev4-cp312-cu130-sm120'),
    '--output', str(ROOT / 'dev7-derived-native-base-v1')], check=True)
