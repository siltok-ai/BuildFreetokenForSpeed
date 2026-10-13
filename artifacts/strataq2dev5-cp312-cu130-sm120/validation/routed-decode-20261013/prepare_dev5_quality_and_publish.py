"""Stage an unaccepted candidate and replay the frozen quality protocol."""
import hashlib
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parent
PRODUCER=Path(r'C:\workspace\worktrees\build-freetoken-flash-next-strata-q2')
base=ROOT/'dev5-derived-native-base-v1'
dest=PRODUCER/'artifacts/strataq2dev5-cp312-cu130-sm120'
out=ROOT/'quality-gate-dev5'
script=ROOT/'run_quality_gate_dev5.py'
for p in [dest,out,script]:
    if p.exists():raise FileExistsError(p)
dest.mkdir()
for name in ['wheels','kernel-cache']:shutil.copytree(base/name,dest/name)
shutil.copy2(base/'build-record.json',dest/'build-record.json')
shutil.copy2(base/'candidate-source/preparation-record.json',dest/'preparation-record.json')
(dest/'logs').mkdir()
shutil.copy2(ROOT/'dev5-derived-native-base-v1.log',dest/'logs/derive.log')
record=json.loads((dest/'build-record.json').read_text())
for name,r in record['artifacts'].items():
    data=(dest/name).read_bytes()
    if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:raise RuntimeError('Published material changed')
out.mkdir()
shutil.copy2(ROOT/'quality-gate-dev3/protocol.json',out/'protocol.json')
source=ROOT/'run_quality_gate_dev3.py'
text=source.read_text(encoding='utf-8').replace("OUT=ROOT/'quality-gate-dev3'","OUT=ROOT/'quality-gate-dev5'")
text=text.replace("freetoken.__version__!='0.1.3+speed.strataq2dev3'","freetoken.__version__!='0.1.3+speed.strataq2dev5'")
text=text.replace('moe_cache_size=1024','moe_cache_size=4096').replace('cuda_graph_max_bs=0','cuda_graph_max_bs=1')
script.write_text(text,encoding='utf-8')
(ROOT/'dev5-quality-protocol-identity.json').write_text(json.dumps(dict(
    protocol_sha256=hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),protocol_unchanged=True,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),derived_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
    changes=['Expected installed dev5 version','Independent output directory','4096 cache slots and graph1'],
    original_reference_results=str(ROOT/'quality-gate-dev3'),native_runtime_acceptance=False),indent=2),encoding='utf-8')
print(json.dumps(dict(candidate=str(dest),quality_script=str(script),version=record['custom_version'],wheel_sha256=record['wheel_sha256'])))
