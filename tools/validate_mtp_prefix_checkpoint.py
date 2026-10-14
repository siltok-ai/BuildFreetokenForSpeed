"""Validate the staged prefix-state patch and pinned Strata engine references offline."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from prepare_source import ROOT, apply_patch, prepare, repository_path, sha256
from validate_mtp_engine_checkpoint import validate as validate_foundation


def validate(record_path):
    record = json.loads(Path(record_path).read_text(encoding='utf-8'))
    previous_path = repository_path(record['predecessor'])
    if sha256(previous_path) != record['predecessor_sha256']:
        raise ValueError('MTP foundation identity drift')
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    validate_foundation(previous_path)
    patch = repository_path(record['patch'])
    if sha256(patch) != record['patch_sha256']:
        raise ValueError('MTP prefix-state patch identity drift')
    for flag in ('full_engine_acceptance', 'profile_created', 'wheel_created', 'speed_mtp_acceptance',
                 'mtp_generation_driver_wired'):
        if record[flag] is not False:
            raise ValueError('Prefix component checkpoint cannot grant generation/release acceptance')
    evidence = {}
    for item in record['evidence']:
        path = repository_path(item['path'])
        if sha256(path) != item['sha256']:
            raise ValueError('Prefix component evidence identity drift')
        raw = gzip.decompress(path.read_bytes())
        if hashlib.sha256(raw).hexdigest() != item['raw_sha256']:
            raise ValueError('Prefix component raw evidence identity drift')
        evidence[path.name.removesuffix('.gz')] = raw
    tests = json.loads(evidence['prefix-tests-v3.json'])
    if tests['status'] != 'passed' or tests['exit_code'] != 0 or tests['source_override'] is not True:
        raise ValueError('Prefix component evidence does not verify the source override')
    if b'65 passed, 12 subtests passed' not in evidence['prefix-tests-v3.log']:
        raise ValueError('Prefix test summary differs from raw evidence')
    geometry = json.loads(evidence['real-gguf-mtp-config-v1.json'])
    if geometry['status'] != 'passed' or geometry['source_override'] is not True:
        raise ValueError('Actual GGUF config geometry was not checked')
    expected = record['replayed_source_sha256']
    if set(expected) != set(record['baseline_sha256']):
        raise ValueError('Prefix-state before/after inventory mismatch')
    with tempfile.TemporaryDirectory(prefix='freetoken-mtp-prefix-') as temporary:
        source, _ = prepare(repository_path(previous['base_profile']), Path(temporary) / 'source')
        apply_patch(source, repository_path(previous['patch']), Path(temporary) / 'foundation.log')
        before = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
        for name, digest in record['baseline_sha256'].items():
            if before.get(name) != digest:
                raise ValueError('Prefix predecessor source differs: ' + name)
        apply_patch(source, patch, Path(temporary) / 'prefix.log')
        after = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
        changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
        if changed != set(expected):
            raise ValueError('Prefix patch file inventory differs from actual replay')
        for name, digest in expected.items():
            if not name.endswith('.py') or after[name] != digest or tests['source_sha256'].get(name) != digest:
                raise ValueError('Prefix replay differs from tested Python source: ' + name)
            ast.parse((source / name).read_text(encoding='utf-8'), filename=name)
    origins = json.loads((ROOT/'research/mtp/references/mtp_engine_origins.json').read_text(encoding='utf-8'))
    if origins['commit'] != '6674a0065fb96bacde33e3eb10f91a1df86f95f2':
        raise ValueError('Strata engine reference revision drift')
    for item in origins['files']:
        path = repository_path(item['path'])
        raw = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if sha256(path) != item['sha256'] or blob != item['git_blob']:
            raise ValueError('Strata engine reference bytes differ: ' + item['source_path'])
    return dict(source_replay_verified=True, archived_component_evidence_verified=True,
                changed_python_files=len(expected), strata_engine_references_verified=len(origins['files']),
                runtime_acceptance=False, release_profile_created=False, wheel_created=False,
                mtp_generation_driver_wired=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('record', type=Path)
    print(json.dumps(validate(parser.parse_args().record)))
