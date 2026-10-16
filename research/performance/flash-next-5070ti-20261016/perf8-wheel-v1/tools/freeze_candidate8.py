"""Keep revision7 immutable and fix the poison-entry ordering in Python-only revision8."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).parent
B=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
S=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate9\build-source')
ST=R/'source9-patch-stages-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert "AttributeError: 'MtpRuntime' object has no attribute 'draft'" in (R/'perf7-poison-before-v1.log').read_text()
assert not (B/'build/profiles/strataq2perf8.json').exists()
series=json.loads((B/'patches/strataq2perf7-series.json').read_text())
profile=json.loads((B/'build/profiles/strataq2perf7.json').read_text())
version='0.1.3+speed.strataq2perf8'
phases=[]
for number,name,title in [(47,'python/freetoken/models/qwen4_exp/mtp_runtime.py','Keep workspace quarantine ahead of draft-policy bookkeeping'),
                          (48,'python/freetoken/version.py','Distinct eighth performance material identity')]:
    before=(S/name).read_text(encoding='utf-8')
    if number==47:
        old="        started = time.perf_counter() if self.draft.draft_policy == 'paired_cjk' else None\n        if self.engine.prefill_workspace is not None:\n            self.engine.prefill_workspace.check_usable()"
        new="        if self.engine.prefill_workspace is not None:\n            self.engine.prefill_workspace.check_usable()\n        started = time.perf_counter() if self.draft.draft_policy == 'paired_cjk' else None"
        assert before.count(old)==1
        after=before.replace(old,new)
    else:after=f'__version__ = "{version}"\n'
    for side,text in [('before',before),('after',after)]:
        p=ST/str(number)/side/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(text.encode())
    (S/name).write_bytes(after.encode())
    p=B/f'patches/series/strataq2perf8-{number:04}.patch';p.write_bytes(''.join(difflib.unified_diff(
        before.splitlines(keepends=True),after.splitlines(keepends=True),fromfile='a/'+name,tofile='b/'+name)).encode())
    relative=p.relative_to(B).as_posix()
    series['patches'].append(dict(id=f'FTPERF-{number-20:03}',path=relative,sha256=sha(p),
        kind='version' if number==48 else 'correctness',title=title,status='active',
        upstream_status='Local performance revision; reassess on upstream upgrade',files=[name]))
    profile['input_sha256'][relative]=sha(p)
    phases.append(dict(number=number,path=name))
catalog=B/'patches/strataq2perf8-series.json';catalog.write_text(json.dumps(series,indent=2)+'\n',encoding='utf-8',newline='\n')
profile['input_sha256'].pop(profile['patch_series']);profile['patch_series']=catalog.relative_to(B).as_posix()
profile['input_sha256'][profile['patch_series']]=sha(catalog)
profile.update(custom_version=version,wheel_filename=f'freetoken-{version}-cp312-cp312-win_amd64.whl')
(B/'build/profiles/strataq2perf8.json').write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8',newline='\n')
(ST/'index.json').write_text(json.dumps(dict(phases=phases),indent=2)+'\n')
print('Revision8 frozen: two Python-only patches; reuse all three native extensions and27AOT from7')
