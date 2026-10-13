"""Actual state/cache/logits validation of v13, separately from throughput."""
import hashlib
import json
from pathlib import Path
import prototype_expert_dp4a_v13 as prototype
import sys
ROOT=Path(__file__).resolve().parent
prototype.install()
import candidate_native_probe as probe
out=Path(sys.argv[sys.argv.index('--output')+1])
try:probe.main()
finally:
    if out.exists():
        d=json.loads(out.read_text(encoding='utf-8'))
        d['prototype']=dict(name='expert-dp4a-v13-state-v14',process_override=True,
            source_sha256=hashlib.sha256(Path(prototype.__file__).read_bytes()).hexdigest(),
            full_engine_acceptance=False)
        d['note']='Actual installed dev4 native main plus process-local expert DP4A override; not an accepted new wheel'
        out.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
