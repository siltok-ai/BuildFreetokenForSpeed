"""Preserve actual differing outputs instead of aborting an observational run."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
original=ROOT/'first_freetoken_performance_dev4_v1.py'
target=ROOT/'first_freetoken_expert_observation_v15.py'
if target.exists():raise FileExistsError(target)
text=original.read_text(encoding='utf-8')
old="                        if not item['cold_hot_token_ids_equal']:raise RuntimeError('Cold/hot actual outputs differ')"
new="                        cold_ids=report['runs'][-1]['output']['token_ids'];hot_ids=item['output']['token_ids']\n                        item['cold_hot_first_difference']=next((i for i,(x,y) in enumerate(zip(cold_ids,hot_ids)) if x!=y), None)\n                        item['cold_hot_length_equal']=len(cold_ids)==len(hot_ids)"
if text.count(old)!=1:raise RuntimeError('Unexpected frozen harness')
text=text.replace(old,new).replace('native_engine_only=True','native_engine_only=False')
text=text.replace("'native_decode_repair_performance_measured'", "'observational_expert_performance_measured'")
target.write_text(text,encoding='utf-8')
(ROOT/'expert-observation-v15-harness-identity.json').write_text(json.dumps(dict(
    source=str(original),source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
    derived=str(target),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    full_engine_acceptance=False,reason='Record full actual outputs and divergence; this does not accept state/numeric/quality correctness'),indent=2),encoding='utf-8')
