"""Freeze an independent native-code revision, never inheriting old binaries."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PRODUCER=Path(r'C:\workspace\worktrees\build-freetoken-flash-next-strata-q2')
BASE=ROOT/'candidate-source-dev3/build-source'
SOURCE=ROOT/'candidate-source-dev4-working/build-source'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    patch=PRODUCER/'patches/series/strataq2dev4-0010-native-gguf-decode.patch'
    series_path=PRODUCER/'patches/strataq2dev4-series.json'
    profile_path=PRODUCER/'build/profiles/strataq2dev4.json'
    spec_path=PRODUCER/'build/specs/speed-strataq2dev4-sm120.txt'
    record_path=ROOT/'candidate-source-dev4-materials.json'
    for p in (patch,series_path,profile_path,spec_path,record_path):
        if p.exists():raise FileExistsError(p)
    files=[]
    for directory in ('python','tests'):
        files.extend(p for p in (SOURCE/directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.append(SOURCE/'pyproject.toml')
    changed=[];parts=[]
    for new in sorted(files):
        name=new.relative_to(SOURCE).as_posix();old=BASE/name
        before=old.read_text(encoding='utf-8') if old.exists() else ''
        after=new.read_text(encoding='utf-8')
        if before==after:continue
        changed.append(name)
        new.write_bytes(after.replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
        parts.append(f'diff --git a/{name} b/{name}\n')
        if not old.exists():parts.append('new file mode 100644\n')
        parts.extend(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
            fromfile='a/'+name if old.exists() else '/dev/null',tofile='b/'+name))
    patch.write_bytes(''.join(parts).replace('\r\n','\n').replace('\n','\r\n').encode('utf-8'))
    series=json.loads((PRODUCER/'patches/strataq2dev3-series.json').read_text())
    series['patches'].append(dict(id='FTSTRATA-004',path=patch.relative_to(PRODUCER).as_posix(),
        sha256=sha(patch),kind='performance',title='Compressed native MMVQ AOT and FP32-activation BF16 GEMV',
        status='active',upstream_status='local candidate; licensed pinned Strata/ggml kernel port; no official FreeToken contribution',
        files=sorted(changed)))
    series_path.write_text(json.dumps(series,indent=2)+'\n',encoding='utf-8')
    old_spec=PRODUCER/'build/specs/speed-sm120.txt'
    specs=old_spec.read_text(encoding='utf-8').splitlines()+['freetoken__gguf_mmvq_r1']
    spec_path.write_text('\n'.join(specs)+'\n',encoding='utf-8')
    profile=json.loads((PRODUCER/'build/profiles/strataq2dev3.json').read_text())
    profile.update(custom_version='0.1.3+speed.strataq2dev4',
        wheel_filename='freetoken-0.1.3+speed.strataq2dev4-cp312-cp312-win_amd64.whl',
        patch_series=series_path.relative_to(PRODUCER).as_posix(),spec_file=spec_path.relative_to(PRODUCER).as_posix())
    profile['input_sha256'].pop('patches/strataq2dev3-series.json')
    profile['input_sha256'].pop('build/specs/speed-sm120.txt')
    for p in (patch,series_path,spec_path):profile['input_sha256'][p.relative_to(PRODUCER).as_posix()]=sha(p)
    profile_path.write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8')
    record=dict(status='frozen_native_candidate_requires_full_build_and_runtime_validation',
        version=profile['custom_version'],profile=str(profile_path),profile_sha256=sha(profile_path),
        patch_sha256=sha(patch),changed_files=sorted(changed),source_sha256={n:sha(SOURCE/n) for n in changed},
        aot_count=len(specs),binary_reuse=False,source=str(SOURCE))
    record_path.write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(version=record['version'],changed_files=len(changed),aot_count=len(specs),binary_reuse=False)))

if __name__=='__main__':main()
