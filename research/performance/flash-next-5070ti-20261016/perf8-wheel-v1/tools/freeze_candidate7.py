"""Freeze the validated indexed-head/paired-policy source as a new revision."""
from pathlib import Path
import difflib,hashlib,json
R=Path(__file__).parent
B=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
S=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate8\build-source')
ST=R/'source8-patch-stages-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    q=json.loads((R/'source8-quality-on-v3.json').read_text(encoding='utf-8'))
    assert q['status']=='passed' and q['score']==15 and q['thinking_score']==3 and not q['regressions']
    assert len(q['rebuild_steps'])==2 and len(q['indexed_head_checks']['exact_full_subset'])==64
    assert all(q['indexed_head_checks']['exact_full_subset'])
    for stem in ['source8-quality-on-v3','source8-performance-on-v2']:
        d=json.loads((R/(stem+'.resource.json')).read_text())
        assert d['returncode']==0 and d['stop_reason'] is None and not d['remaining_descendants']
    version='0.1.3+speed.strataq2perf7'
    outputs=[B/'build/profiles/strataq2perf7.json',B/'patches/strataq2perf7-series.json']
    assert not any(p.exists() for p in outputs)
    name='python/freetoken/version.py';before=(S/name).read_bytes();after=f'__version__ = "{version}"\n'.encode()
    for side,data in [('before',before),('after',after)]:
        p=ST/'46-package-identity'/side/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    (S/name).write_bytes(after)
    series=json.loads((B/'patches/strataq2perf6-series.json').read_text())
    profile=json.loads((B/'build/profiles/strataq2perf6.json').read_text())
    titles=['Index original Q5 draft rows without copying weights',
        'Compare full3 and CJK2 drafts with paired complete-cycle measurements',
        'Distinct seventh performance material identity']
    phases=[]
    for number,title in zip(range(44,47),titles):
        directory=next(p for p in ST.iterdir() if p.is_dir() and p.name.startswith(str(number)+'-'))
        lines=[];names=[];rows=[]
        for old in sorted((directory/'before').rglob('*')):
            if not old.is_file():continue
            name=old.relative_to(directory/'before').as_posix();new=directory/'after'/name
            assert (S/name).read_text(encoding='utf-8')==new.read_text(encoding='utf-8')
            a=old.read_text(encoding='utf-8');b=new.read_text(encoding='utf-8')
            assert a!=b
            lines.extend(difflib.unified_diff(a.splitlines(keepends=True),b.splitlines(keepends=True),
                fromfile='/dev/null' if not old.stat().st_size else 'a/'+name,tofile='b/'+name))
            names.append(name);rows.append(dict(path=name,before_sha256=sha(old),after_sha256=sha(new)))
        path=B/f'patches/series/strataq2perf7-{number:04}.patch';assert not path.exists();path.write_bytes(''.join(lines).encode())
        relative=path.relative_to(B).as_posix()
        series['patches'].append(dict(id=f'FTPERF-{number-20:03}',path=relative,sha256=sha(path),
            kind='version' if number==46 else 'performance',title=title,status='active',
            upstream_status='Local measured performance revision; reassess on upstream upgrade',files=names))
        profile['input_sha256'][relative]=sha(path);phases.append(dict(name=directory.name,files=rows))
    catalog=B/'patches/strataq2perf7-series.json';catalog.write_text(json.dumps(series,indent=2)+'\n',encoding='utf-8',newline='\n')
    profile['input_sha256'].pop(profile['patch_series']);profile['patch_series']=catalog.relative_to(B).as_posix()
    profile['input_sha256'][profile['patch_series']]=sha(catalog)
    old_spec=profile['spec_file'];spec=B/'build/specs/strataq2perf7-sm120.txt';spec.parent.mkdir(parents=True,exist_ok=True)
    spec.write_bytes((B/old_spec).read_bytes().replace(b'freetoken__gguf_mmvq_r2',b'freetoken__gguf_mmvq_r3'))
    profile['input_sha256'].pop(old_spec);profile['spec_file']=spec.relative_to(B).as_posix();profile['input_sha256'][profile['spec_file']]=sha(spec)
    profile.update(custom_version=version,wheel_filename=f'freetoken-{version}-cp312-cp312-win_amd64.whl',
        gguf_rebuild=dict(old='freetoken__gguf_mmvq_r2',new='freetoken__gguf_mmvq_r3'))
    (B/'build/profiles/strataq2perf7.json').write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8',newline='\n')
    (ST/'index.json').write_text(json.dumps(dict(phases=phases),indent=2)+'\n')
    print(json.dumps(dict(version=version,patches=len(series['patches']),scope='source validated; ordinary wheel/Speed acceptance pending',source_frozen=str(S))))

if __name__=='__main__':main()
