"""Freeze a Python/Triton revision over the fully rebuilt dev4 native base."""
import difflib
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PRODUCER=Path(r'C:\workspace\worktrees\build-freetoken-flash-next-strata-q2')
BASE=ROOT/'dev4-built-native-materials-v2/build-source'
SOURCE=ROOT/'candidate-source-dev5-working/build-source'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    patch=PRODUCER/'patches/series/strataq2dev5-0011-routed-q2-expert-decode.patch'
    series_path=PRODUCER/'patches/strataq2dev5-series.json'
    profile_path=PRODUCER/'build/profiles/strataq2dev5.json'
    record_path=ROOT/'candidate-source-dev5-materials.json'
    for p in [patch,series_path,profile_path,record_path]:
        if p.exists():raise FileExistsError(p)
    version=SOURCE/'python/freetoken/version.py'
    version.write_text(version.read_text(encoding='utf-8').replace('0.1.3+speed.strataq2dev4','0.1.3+speed.strataq2dev5'),encoding='utf-8')
    changed=[];parts=[]
    for new in sorted(p for d in ['python','tests'] for p in (SOURCE/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts):
        name=new.relative_to(SOURCE).as_posix();old=BASE/name
        before=old.read_bytes() if old.exists() else b''
        after=new.read_bytes()
        if before==after:continue
        left=before.decode('utf-8').replace('\r\n','\n');right=after.decode('utf-8').replace('\r\n','\n')
        if left==right:continue
        new.write_bytes(right.replace('\n','\r\n').encode('utf-8'))
        changed.append(name);parts.append(f'diff --git a/{name} b/{name}\n')
        parts.extend(difflib.unified_diff(left.splitlines(keepends=True),right.splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name))
    expected={'python/freetoken/moe/fused_q2_q8.py','python/freetoken/version.py','tests/moe/test_q2_grouped.py'}
    if set(changed)!=expected:raise RuntimeError('Unexpected source delta: '+str(changed))
    patch.write_bytes(''.join(parts).replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
    series=json.loads((PRODUCER/'patches/strataq2dev4-series.json').read_text())
    series['patches'].append(dict(id='FTSTRATA-005',path=patch.relative_to(PRODUCER).as_posix(),sha256=sha(patch),
        kind='performance',title='Routed Q2xQ8 DP4A decode without full-cache grouping or padded tensor-core tiles',
        status='active',upstream_status='local candidate; original Triton implementation informed by pinned Strata expert design; no official contribution',files=changed))
    series_path.write_text(json.dumps(series,indent=2)+'\n',encoding='utf-8')
    profile=json.loads((PRODUCER/'build/profiles/strataq2dev4.json').read_text())
    profile['input_sha256'].pop(profile['patch_series'])
    profile.update(custom_version='0.1.3+speed.strataq2dev5',wheel_filename='freetoken-0.1.3+speed.strataq2dev5-cp312-cp312-win_amd64.whl',patch_series=series_path.relative_to(PRODUCER).as_posix())
    for p in [patch,series_path]:profile['input_sha256'][p.relative_to(PRODUCER).as_posix()]=sha(p)
    profile_path.write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8')
    record=dict(status='frozen_requires_derived_wheel_and_runtime_validation',version=profile['custom_version'],
        source=str(SOURCE),profile=str(profile_path),profile_sha256=sha(profile_path),patch_sha256=sha(patch),changed_files=changed,
        source_sha256={n:sha(SOURCE/n) for n in changed},native_inputs_changed=False,base_profile='strataq2dev4',aot_count=27,
        runtime_acceptance=False,known_observation='v13 strict 512-token cold/hot token equality failed; state/logit/quality impact still being bounded, not accepted by source tests')
    record_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(record))
if __name__=='__main__':main()
