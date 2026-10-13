"""Freeze the explicitly configured protected-owner revision over immutable dev5."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRODUCER = Path('C:/workspace/worktrees/build-freetoken-flash-next-strata-q2')
BASE = ROOT / 'dev6-frozen-replay-v1/build-source'
SOURCE = ROOT / 'candidate-source-dev7-working/build-source'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    patch = PRODUCER / 'patches/series/strataq2dev7-0013-sparse-prefill-admission.patch'
    series_path = PRODUCER / 'patches/strataq2dev7-series.json'
    profile_path = PRODUCER / 'build/profiles/strataq2dev7.json'
    record_path = ROOT / 'candidate-source-dev7-materials-v1.json'
    for path in (patch, series_path, profile_path, record_path):
        if path.exists():
            raise FileExistsError(path)
    components = json.loads((ROOT / 'dev7-working-source-component-v1.json').read_text())
    if not components['passed']:
        raise RuntimeError('Working-source component gate failed')
    changed, parts = [], []
    expected = {'python/freetoken/version.py', 'python/freetoken/moe/q2_residency.py', 'tests/moe/test_q2_residency.py'}
    for new in sorted(p for d in ('python', 'tests') for p in (SOURCE / d).rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts):
        name = new.relative_to(SOURCE).as_posix()
        old = BASE / name
        before, after = old.read_bytes() if old.exists() else b'', new.read_bytes()
        if before == after:
            continue
        left, right = before.decode('utf-8').replace('\r\n', '\n'), after.decode('utf-8').replace('\r\n', '\n')
        if left == right:
            continue
        if name not in expected:
            raise RuntimeError('Unexpected source change: ' + name)
        new.write_bytes(right.replace('\n', '\r\n').encode())
        changed.append(name)
        parts.append(f'diff --git a/{name} b/{name}\n')
        if not old.exists():
            parts.append('new file mode 100644\n')
        parts.extend(difflib.unified_diff(left.splitlines(keepends=True), right.splitlines(keepends=True),
                     fromfile='a/' + name if old.exists() else '/dev/null', tofile='b/' + name))
    if set(changed) != expected:
        raise RuntimeError('Missing source changes')
    patch.write_bytes(''.join(parts).replace('\n', '\r\n').encode())
    series = json.loads((PRODUCER / 'patches/strataq2dev6-series.json').read_text())
    series['patches'].append(dict(id='FTSTRATA-007', path=patch.relative_to(PRODUCER).as_posix(),
        sha256=sha(patch), kind='performance', status='active',
        title='Admit the existing 16-token sparse-prefill path with 160 routed experts',
        upstream_status='local optional candidate; frozen Strata ranking and original validated Triton ownership kernels; no official contribution',
        files=changed))
    series_path.write_text(json.dumps(series, indent=2) + '\n', encoding='utf-8')
    profile = json.loads((PRODUCER / 'build/profiles/strataq2dev6.json').read_text())
    profile['input_sha256'].pop(profile['patch_series'])
    profile.update(custom_version='0.1.3+speed.strataq2dev7',
        wheel_filename='freetoken-0.1.3+speed.strataq2dev7-cp312-cp312-win_amd64.whl',
        patch_series=series_path.relative_to(PRODUCER).as_posix())
    for path in (patch, series_path):
        profile['input_sha256'][path.relative_to(PRODUCER).as_posix()] = sha(path)
    profile_path.write_text(json.dumps(profile, indent=2) + '\n', encoding='utf-8')
    record = dict(status='frozen_requires_derived_wheel_and_installed_runtime_validation',
        version=profile['custom_version'], source=str(SOURCE), profile=str(profile_path),
        profile_sha256=sha(profile_path), patch_sha256=sha(patch), changed_files=changed,
        source_sha256={name: sha(SOURCE / name) for name in changed},
        native_inputs_changed=False, native_base_profile='strataq2dev4', aot_count=27,
        component_gate=str(ROOT / 'dev7-working-source-component-v1.json'), runtime_acceptance=False)
    record_path.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
