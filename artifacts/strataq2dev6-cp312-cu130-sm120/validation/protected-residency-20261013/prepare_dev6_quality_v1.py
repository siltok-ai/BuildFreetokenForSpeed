"""Keep the frozen quality protocol; select only the new installed Q2 configuration."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
out = ROOT / 'quality-gate-dev6'
if out.exists():
    raise FileExistsError(out)
out.mkdir()
shutil.copy2(ROOT/'quality-gate-dev5/protocol.json', out/'protocol.json')
source = ROOT/'run_quality_gate_dev5.py'
text = source.read_text(encoding='utf-8').replace("OUT=ROOT/'quality-gate-dev5'", "OUT=ROOT/'quality-gate-dev6'")
text = text.replace('speed.strataq2dev5', 'speed.strataq2dev6')
text = text.replace('moe_cache_size=4096,', 'moe_cache_size=11264,moe_q2_resident_experts=9216,moe_q2_dynamic_slots=1024,')
text = text.replace('memory_ratio=.8', 'memory_ratio=.9')
text = text.replace("choices=['freetoken','mixed','strata','compare']", "choices=['freetoken']")
script = ROOT/'run_quality_gate_dev6_v1.py'
if script.exists():
    raise FileExistsError(script)
script.write_text(text, encoding='utf-8')
(ROOT/'dev6-quality-protocol-identity-v1.json').write_text(json.dumps(dict(
    protocol_sha256=hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(), protocol_unchanged=True,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), derived_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    changes=['Independent output and dev6 identity', 'Explicit protected9216/dynamic1024/cache11264',
             'memory_ratio0.9 fits fixed capacity in runtime rebuild budget', 'Only run new FreeToken, reuse recorded references']),
    indent=2), encoding='utf-8')
print(script)
