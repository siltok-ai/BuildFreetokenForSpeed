"""Freeze tested early host placement, heat decay and runtime admission without changing old releases."""
from pathlib import Path
import difflib
import hashlib
import json

RUN=Path(__file__).resolve().parent
ROOT=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
BASE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-03\candidate-source\build-source')
SOURCE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate4\build-source')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def text(root,name):return (root/name).read_text(encoding='utf-8')


def without_decay(name,value):
    if name.endswith('engine/config.py'):
        value=value.replace('    moe_q2_adapt_decay: float = 1.0\n','')
        value=value.replace("        if not 0 < self.moe_q2_adapt_decay <= 1:\n            raise ValueError('Online Q2 heat decay must be in (0, 1]')\n",'')
    elif name.endswith('engine/engine.py'):
        value=value.replace(", 'adapt_decay': config.moe_q2_adapt_decay",'')
    elif name.endswith('moe/q2_residency.py'):
        value=value.replace('decode_overlap=True, adapt_every=0,\n                 adapt_decay=1.0, **kwargs):',
                            'decode_overlap=True, adapt_every=0, **kwargs):')
        value=value.replace("        if not 0 < adapt_decay <= 1:\n            raise ValueError('Online Q2 heat decay must be in (0, 1]')\n",'')
        value=value.replace('        self.adapt_decay = adapt_decay\n','')
        value=value.replace(', decay=self.adapt_decay','')
    if 'adapt_decay' in value:raise ValueError('Unsplit heat decay: '+name)
    return value


def main():
    component=json.loads((RUN/'candidate4-source-components-v1.resource.json').read_text())
    assert component['returncode']==0 and not component['remaining_descendants']
    for stem in ('candidate4-source-quality-off-v1','candidate4-source-quality-on-r4608-v1'):
        quality=json.loads((RUN/(stem+'.json')).read_text(encoding='utf-8'))
        resource=json.loads((RUN/(stem+'.resource.json')).read_text())
        assert quality['status']=='passed' and quality['score']>=15 and quality['thinking_score']==3
        assert not quality['regressions'] and len(quality['rebuild_steps'])==2
        assert quality['embedding_placement']['actual']=='mapped_host' and quality['host_tables_bytes']>=273152000
        assert resource['returncode']==0 and not resource['remaining_descendants']
    version='0.1.3+speed.strataq2perf4'
    (SOURCE/'python/freetoken/version.py').write_text(f'__version__ = "{version}"\n',encoding='utf-8',newline='\n')
    shared=('python/freetoken/engine/config.py','python/freetoken/engine/engine.py','python/freetoken/moe/q2_residency.py')
    intermediate={name:without_decay(name,text(SOURCE,name)) for name in shared}
    groups=[
        ('0032','FTPERF-012','Reuse Q2 admission code across runtime layer offsets','performance',
            ['python/freetoken/moe/q2_residency.py']),
        ('0033','FTPERF-013','Place original compressed embedding in mapped host memory before cache budgeting','performance',
            ['python/freetoken/kernel/pinned.py','python/freetoken/models/qwen4_exp/gguf_weight.py',
             'python/freetoken/models/qwen4_exp/model.py','python/freetoken/engine/config.py',
             'python/freetoken/engine/engine.py','tests/models/qwen4_exp/test_gguf_execution.py']),
        ('0034','FTPERF-014','Configure recent expert heat decay without increasing maintenance frequency','performance',list(shared)),
        ('0035','FTPERF-015','Distinct immutable fourth performance wheel identity','version',['python/freetoken/version.py']),
    ]
    series=json.loads((ROOT/'patches/strataq2perf3-series.json').read_text())
    profile=json.loads((ROOT/'build/profiles/strataq2perf3.json').read_text())
    for number,ident,title,kind,names in groups:
        lines=[]
        for name in names:
            old=intermediate[name] if number=='0034' else text(BASE,name)
            new=intermediate[name] if name in intermediate and number in ('0032','0033') else text(SOURCE,name)
            if old==new:raise ValueError('Empty advertised delta: '+name)
            lines.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),
                fromfile='a/'+name,tofile='b/'+name))
        path=ROOT/f'patches/series/strataq2perf4-{number}.patch'
        if path.exists():raise FileExistsError(path)
        path.write_bytes(''.join(lines).encode('utf-8'))
        relative=path.relative_to(ROOT).as_posix()
        series['patches'].append(dict(id=ident,path=relative,sha256=sha(path),kind=kind,title=title,
            status='active',upstream_status='Local performance revision; reassess on upstream upgrade',files=names))
        profile['input_sha256'][relative]=sha(path)
    selected='patches/strataq2perf4-series.json'
    path=ROOT/selected
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(series,indent=2)+'\n',encoding='utf-8',newline='\n')
    profile['input_sha256'].pop(profile['patch_series'])
    profile.update(custom_version=version,wheel_filename=f'freetoken-{version}-cp312-cp312-win_amd64.whl',patch_series=selected)
    profile['input_sha256'][selected]=sha(path)
    path=ROOT/'build/profiles/strataq2perf4.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(version=version,patches=len(series['patches']),native_spec_unchanged=profile['spec_file'])))


if __name__=='__main__':main()
