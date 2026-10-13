"""Derive an installed-wheel measurement with explicit trajectory observations."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
source = ROOT / 'first_freetoken_expert_observation_v15.py'
target = ROOT / 'first_freetoken_performance_dev5_v1.py'
if target.exists():
    raise FileExistsError(target)
text = source.read_text(encoding='utf-8')
text = text.replace('strataq2dev4', 'strataq2dev5').replace('dev4-native-runtime-gate-v1.json', 'dev5-routed-runtime-gate-v1.json')
text = text.replace('native_engine_only=False,speed_acceptance=False', 'native_engine_only=True,process_engine_override=False,full_runtime_acceptance=False,speed_acceptance=False')
text = text.replace("'observational_expert_performance_measured'", "'installed_routed_decode_observational_performance_measured'")
target.write_text(text, encoding='utf-8')
(ROOT / 'dev5-performance-harness-derivation-v1.json').write_text(json.dumps(dict(
    source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    derived=str(target),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    changes=['Actual installed version dev5', 'Bounded dev5 runtime gate with hash', 'No process-local engine override', 'Cold/hot token trajectory first difference retained, not a bitwise performance gate'],
    timestamp_throughput_protocol_changed=False, release_acceptance=False),indent=2),encoding='utf-8')
