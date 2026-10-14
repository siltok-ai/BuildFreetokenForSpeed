"""Verify full-length greedy evidence against the frozen native outputs, without release claims."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from prepare_source import ROOT, apply_patch, prepare, repository_path, sha256
from validate_mtp_prefix_checkpoint import validate as validate_prefix


def validate(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    prior_path = repository_path(record['predecessor'])
    if sha256(prior_path) != record['predecessor_sha256']:
        raise ValueError('Scheduler predecessor identity drift')
    validate_prefix(prior_path)
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    foundation = json.loads(repository_path(prior['predecessor']).read_text(encoding='utf-8'))
    patch = repository_path(record['patch'])
    if sha256(patch) != record['patch_sha256'] or record['mtp_generation_driver_wired'] is not True:
        raise ValueError('Scheduler patch or driver identity differs')
    for flag in ('full_engine_acceptance', 'lifecycle_acceptance', 'profile_created', 'wheel_created',
                 'speed_mtp_acceptance', 'performance_acceptance'):
        if record[flag] is not False:
            raise ValueError('Greedy checkpoint cannot grant lifecycle/service/release acceptance')
    raw = {}
    for item in record['evidence']:
        p = repository_path(item['path'])
        unpacked = gzip.decompress(p.read_bytes())
        if sha256(p) != item['sha256'] or hashlib.sha256(unpacked).hexdigest() != item['raw_sha256']:
            raise ValueError('Scheduler evidence bytes drift')
        raw[p.name.removesuffix('.gz')] = unpacked
    tests = json.loads(raw['runtime-components-v2.json'])
    actual = json.loads(raw['integrated-mtp-v2.json'])
    if tests['status'] != 'passed' or b'80 passed, 12 subtests passed' not in raw['runtime-components-v2.log']:
        raise ValueError('Scheduler component evidence is incomplete')
    if actual['status'] != 'passed' or actual['source_override'] is not True or len(actual['runs']) != 4:
        raise ValueError('Full-weight scheduler run is incomplete')
    baseline_path = ROOT/'research/mtp/evidence/20261014-v2/raw/dev7-native-performance-11264-v1.json.gz'
    if sha256(baseline_path) != '6b5ef3d45530e2b136d5a929dcd50a2875e3f998c74ed2b504e777a9d7cb61a0':
        raise ValueError('Frozen native baseline drift')
    baseline = {r['label']: r for r in json.loads(gzip.decompress(baseline_path.read_bytes()))['runs']}
    for run in actual['runs']:
        ids = run['output']['token_ids']
        if len(ids) != 512 or ids != baseline[run['label']]['output']['token_ids']:
            raise ValueError('Actual scheduler outputs differ: ' + run['label'])
        if len(run['arrival_offsets']) != 512 or run['arrival_offsets'] != sorted(run['arrival_offsets']):
            raise ValueError('Actual delivery timestamps differ')
        position = run['prompt_tokens']
        for step in run['rounds']:
            if step['position'] != position or not 1 <= step['committed_rows'] <= step['draft_count'] + 1 <= 4:
                raise ValueError('Committed scheduler window accounting differs')
            if step['accepted_drafts'] != step['committed_rows'] - 1:
                raise ValueError('Accepted scheduler window accounting differs')
            position += step['committed_rows']
        if position != run['prompt_tokens'] + 511:
            raise ValueError('Native pending-token accounting differs')
    expected = record['replayed_source_sha256']
    with tempfile.TemporaryDirectory(prefix='freetoken-mtp-scheduler-') as temporary:
        source, _ = prepare(repository_path(foundation['base_profile']), Path(temporary) / 'source')
        for i, data in enumerate((foundation, prior)):
            apply_patch(source, repository_path(data['patch']), Path(temporary) / f'prior-{i}.log')
        before = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
        if any(before.get(name) != digest for name, digest in record['baseline_sha256'].items()):
            raise ValueError('Scheduler predecessor source differs')
        apply_patch(source, patch, Path(temporary) / 'scheduler.log')
        after = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*') if p.is_file()}
        if {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)} != set(expected):
            raise ValueError('Scheduler patch file inventory differs')
        for name, digest in expected.items():
            if (not name.endswith('.py') or after[name] != digest or actual['source_sha256'].get(name) != digest
                    or tests['source_sha256'].get(name) != digest):
                raise ValueError('Scheduler replay differs from tested Python bytes: ' + name)
            ast.parse((source / name).read_text(encoding='utf-8'), filename=name)
    return dict(source_replay_verified=True, component_evidence_verified=True, actual_512_token_runs_verified=4,
                frozen_native_output_ids_equal=True, mtp_generation_driver_wired=True,
                lifecycle_acceptance=False, performance_acceptance=False, speed_mtp_acceptance=False)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('record', type=Path)
    print(json.dumps(validate(p.parse_args().record)))
