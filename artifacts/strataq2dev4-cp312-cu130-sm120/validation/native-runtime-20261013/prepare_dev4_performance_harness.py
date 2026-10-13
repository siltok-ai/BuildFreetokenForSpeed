"""Copy frozen measurement protocol with an explicit new revision gate."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=ROOT/'first_freetoken_performance_dev3_v5.py'
target=ROOT/'first_freetoken_performance_dev4_v1.py'
if target.exists():raise FileExistsError(target)
text=source.read_text(encoding='utf-8')
text=text.replace("default='0.1.3+speed.strataq2dev3'","default='0.1.3+speed.strataq2dev4'")
text=text.replace("ROOT/'dev3-correctness-quality-acceptance.json'","ROOT/'dev4-native-runtime-gate-v1.json'")
text=text.replace("if not all(acceptance['gates'].values()):", "if not acceptance['passed'] or not all(acceptance['gates'].values()):")
text=text.replace("no_mtp=True,runs=[],sampling=", "no_mtp=True,runs=[],runtime_gate=str(ROOT/'dev4-native-runtime-gate-v1.json'),runtime_gate_sha256=hashlib.sha256((ROOT/'dev4-native-runtime-gate-v1.json').read_bytes()).hexdigest(),native_engine_only=True,speed_acceptance=False,sampling=")
text=text.replace("'first_performance_measured'", "'native_decode_repair_performance_measured'")
target.write_text(text,encoding='utf-8')
(ROOT/'dev4-performance-harness-derivation-v1.json').write_text(json.dumps(dict(
    source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    derived=str(target),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    changes=['Expected actual installed dev4 version','Independent bounded dev4 runtime gate','Gate SHA in output','Native repair result naming'],
    measurement_protocol_changed=False,process_engine_override=False),indent=2),encoding='utf-8')
