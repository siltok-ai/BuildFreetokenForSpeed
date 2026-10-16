"""Freeze validated consolidated source6 as the fifth distinct material version."""
from pathlib import Path
import difflib
import hashlib
import json

RUN=Path(__file__).parent
ROOT=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
SOURCE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate6\build-source')
STAGES=RUN/'source6-patch-stages-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    for mode in ('on','off'):
        for suite in ('quality','performance'):
            stem=f'source6-{suite}-{mode}-v1'
            result=json.loads((RUN/(stem+'.json')).read_text(encoding='utf-8'))
            resource=json.loads((RUN/(stem+'.resource.json')).read_text())
            assert result['status']=='passed' and resource['returncode']==0 and resource['stop_reason'] is None
            assert not resource['remaining_descendants']
            if suite=='quality':
                assert result['score']>=15 and result['thinking_score']==3 and not result['regressions']
                assert len(result['rebuild_steps'])==2
    resource=json.loads((RUN/'source6-components-v2.resource.json').read_text())
    assert resource['returncode']==0 and not resource['remaining_descendants']
    assert '27 passed' in (RUN/'source6-components-v2.stdout.log').read_text(encoding='utf-8')
    phases=json.loads((STAGES/'index.json').read_text())['phases']
    final={r['path']:r['after_sha256'] for phase in phases for r in phase['files']}
    assert all(sha(SOURCE/name)==digest for name,digest in final.items())
    version='0.1.3+speed.strataq2perf5'
    selected='patches/strataq2perf5-series.json'
    profile_path=ROOT/'build/profiles/strataq2perf5.json'
    outputs=[ROOT/selected,profile_path,*[ROOT/f'patches/series/strataq2perf5-{n:04}.patch' for n in range(36,41)]]
    if any(p.exists() for p in outputs):raise FileExistsError('Use a fresh frozen revision')
    version_file='python/freetoken/version.py'
    before=(SOURCE/version_file).read_bytes()
    after=f'__version__ = "{version}"\n'.encode()
    (SOURCE/version_file).write_bytes(after)
    phase=dict(name='40-package-identity',files=[dict(path=version_file,created=False)])
    for side,data in [('before',before),('after',after)]:
        p=STAGES/phase['name']/side/version_file;p.parent.mkdir(parents=True);p.write_bytes(data)
    phases.append(phase)
    titles=['Publish owner metadata on GPU and permit zero extra dynamic quota',
        'Decode original signed Q2 bytes with byte permute',
        'Reuse valid GPU dynamic copies during protected promotion',
        'Cache bounded original PLE rows in host memory',
        'Distinct fifth performance wheel identity']
    series=json.loads((ROOT/'patches/strataq2perf4-series.json').read_text())
    profile=json.loads((ROOT/'build/profiles/strataq2perf4.json').read_text())
    for number,phase,title in zip(range(36,41),phases,titles):
        lines=[];names=[]
        for row in phase['files']:
            name=row['path'];names.append(name)
            old=(STAGES/phase['name']/'before'/name).read_text(encoding='utf-8')
            new=(STAGES/phase['name']/'after'/name).read_text(encoding='utf-8')
            if old==new:raise ValueError('Empty frozen delta: '+name)
            lines.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),
                fromfile='/dev/null' if row['created'] else 'a/'+name,tofile='b/'+name))
        p=ROOT/f'patches/series/strataq2perf5-{number:04}.patch';p.write_bytes(''.join(lines).encode())
        relative=p.relative_to(ROOT).as_posix()
        series['patches'].append(dict(id=f'FTPERF-{number-20:03}',path=relative,sha256=sha(p),
            kind='version' if number==40 else 'performance',title=title,status='active',
            upstream_status='Local performance revision; reassess on upstream upgrade',files=names))
        profile['input_sha256'][relative]=sha(p)
    p=ROOT/selected;p.write_text(json.dumps(series,indent=2)+'\n',encoding='utf-8',newline='\n')
    profile['input_sha256'].pop(profile['patch_series'])
    profile.update(custom_version=version,wheel_filename=f'freetoken-{version}-cp312-cp312-win_amd64.whl',patch_series=selected)
    profile['input_sha256'][selected]=sha(p)
    profile_path.write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(version=version,patches=len(series['patches']),rebuilt_native='row_store',unchanged_aot=27)))


if __name__=='__main__':main()
