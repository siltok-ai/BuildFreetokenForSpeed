"""保留48组历史补丁，分别追加实际加载修复、维护性重构和独立版本身份。"""
from pathlib import Path
import difflib
import hashlib
import json

BUILD = Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
BASE = Path(r'C:\workspace\build\freetoken-mats\review-20261017-base\build-source')
SOURCE = Path(r'C:\workspace\build\freetoken-mats\review-20261017-work\build-source')
RUN = Path(__file__).parent
VERSION = '0.1.3+speed.strataq2review1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """只冻结明确的源码差异，后续必须重新重放并逐字节核对生成的wheel。"""
    outputs = [BUILD/'build/profiles/strataq2review1.json', BUILD/'patches/strataq2review1-series.json']
    if any(path.exists() for path in outputs):
        raise FileExistsError('Review profile already frozen; use a new revision')
    series = json.loads((BUILD/'patches/strataq2perf8-series.json').read_text())
    profile = json.loads((BUILD/'build/profiles/strataq2perf8.json').read_text())
    names = {p.relative_to(BASE).as_posix() for p in BASE.rglob('*.py')}
    names |= {p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob('*.py')}
    changed = {}
    for name in names:
        old = (BASE/name).read_text(encoding='utf-8') if (BASE/name).exists() else ''
        new = (SOURCE/name).read_text(encoding='utf-8')
        if old != new:
            changed[name] = (old, new)
    version_name = 'python/freetoken/version.py'
    changed[version_name] = ((BASE/version_name).read_text(), f'__version__ = "{VERSION}"\n')
    correctness = {'python/freetoken/models/qwen4_exp/mtp_drafter.py',
                   'tests/models/qwen4_exp/test_mtp_runtime.py'}
    phases = [
        (49, 'correctness', 'Hash exactly the MTP tensor bytes loaded into memory', correctness),
        (50, 'maintenance', 'Share GGUF readers, clarify MTP state flow and remove duplicate bookkeeping',
         set(changed) - correctness - {version_name}),
        (51, 'version', 'Distinct immutable reviewed runtime identity', {version_name}),
    ]
    records = []
    for number, kind, title, paths in phases:
        patch = BUILD/f'patches/series/strataq2review1-{number:04d}.patch'
        lines = []
        for name in sorted(paths):
            before, after = changed[name]
            lines.extend(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True),
                fromfile='a/'+name if (BASE/name).exists() else '/dev/null', tofile='b/'+name))
            with (SOURCE/name).open('w', encoding='utf-8', newline='\r\n') as stream:
                stream.write(after)
        patch.write_bytes(''.join(lines).encode('utf-8'))
        relative = patch.relative_to(BUILD).as_posix()
        series['patches'].append(dict(id=f'FTREVIEW-{number-48+6:03d}', path=relative, sha256=sha(patch),
            kind=kind, title=title, status='active', upstream_status='Local review; reassess on upstream upgrade',
            files=sorted(paths)))
        profile['input_sha256'][relative] = sha(patch)
        records.append(dict(patch=relative, files=sorted(paths)))
    catalog = outputs[1]
    catalog.write_text(json.dumps(series, indent=2)+'\n', encoding='utf-8', newline='\n')
    profile['input_sha256'].pop(profile['patch_series'])
    profile['patch_series'] = catalog.relative_to(BUILD).as_posix()
    profile['input_sha256'][profile['patch_series']] = sha(catalog)
    profile.update(custom_version=VERSION, wheel_filename=f'freetoken-{VERSION}-cp312-cp312-win_amd64.whl')
    outputs[0].write_text(json.dumps(profile, indent=2)+'\n', encoding='utf-8', newline='\n')
    (RUN/'frozen-review1.json').write_text(json.dumps(dict(version=VERSION, patches=records), indent=2)+'\n')
    print(json.dumps(dict(version=VERSION, patch_count=len(series['patches']), changed_files=len(changed))))


if __name__ == '__main__':
    main()
