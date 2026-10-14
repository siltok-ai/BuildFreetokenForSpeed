"""Verify replay and archived component evidence; never grant runtime acceptance."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from prepare_source import ROOT, apply_patch, prepare, repository_path, sha256


def validate(record_path):
    record = json.loads(Path(record_path).read_text(encoding='utf-8'))
    base = repository_path(record['base_profile'])
    patch = repository_path(record['patch'])
    if sha256(base) != record['base_profile_sha256'] or sha256(patch) != record['patch_sha256']:
        raise ValueError('MTP checkpoint base or patch identity drift')
    for flag in ('full_engine_acceptance', 'profile_created', 'wheel_created', 'speed_mtp_acceptance'):
        if record[flag] is not False:
            raise ValueError('Development checkpoint cannot grant release/runtime acceptance')
    archived = {}
    for item in record['evidence']:
        packed = repository_path(item['path'])
        if sha256(packed) != item['sha256']:
            raise ValueError('MTP component evidence identity drift')
        raw = gzip.decompress(packed.read_bytes())
        if hashlib.sha256(raw).hexdigest() != item['raw_sha256']:
            raise ValueError('MTP component evidence raw identity drift')
        archived[packed.name.removesuffix('.gz')] = raw
    tests = json.loads(archived['foundation-tests-v3.json'])
    if tests['status'] != 'passed' or tests['exit_code'] != 0 or tests['source_override'] is not True:
        raise ValueError('MTP component tests do not prove the development source override')
    if b'34 passed, 12 subtests passed' not in archived['foundation-tests-v3.log']:
        raise ValueError('MTP component summary differs from raw pytest evidence')
    expected = record['replayed_source_sha256']
    if set(expected) != set(record['baseline_sha256']):
        raise ValueError('MTP checkpoint before/after inventory mismatch')
    with tempfile.TemporaryDirectory(prefix='freetoken-mtp-foundation-') as temporary:
        source, _ = prepare(base, Path(temporary) / 'source')
        before = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
        for name, digest in record['baseline_sha256'].items():
            if before.get(name) != digest:
                raise ValueError('MTP base source differs: ' + name)
        apply_patch(source, patch, Path(temporary) / 'patch.log')
        after = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
        changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
        if changed != set(expected):
            raise ValueError('MTP patch file inventory differs from actual replay')
        for name, digest in expected.items():
            if not name.endswith('.py') or after[name] != digest or tests['sha256'].get(name) != digest:
                raise ValueError('MTP replay differs from tested Python bytes: ' + name)
            ast.parse((source / name).read_text(encoding='utf-8'), filename=name)
    return dict(source_replay_verified=True, archived_component_evidence_verified=True,
                changed_python_files=len(expected), runtime_acceptance=False,
                release_profile_created=False, wheel_created=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('record', type=Path)
    print(json.dumps(validate(parser.parse_args().record)))
