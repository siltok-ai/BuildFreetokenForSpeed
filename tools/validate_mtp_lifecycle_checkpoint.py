"""Check actual lifecycle deliveries and bounded-window replay without release claims."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from prepare_source import apply_patch, prepare, repository_path, sha256
from validate_mtp_scheduler_checkpoint import validate as validate_scheduler


def validate(path):
    record = json.loads(Path(path).read_text(encoding='utf-8'))
    prior_path = repository_path(record['predecessor'])
    if sha256(prior_path) != record['predecessor_sha256']:
        raise ValueError('Lifecycle predecessor identity drift')
    validate_scheduler(prior_path)
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    prefix = json.loads(repository_path(prior['predecessor']).read_text(encoding='utf-8'))
    foundation = json.loads(repository_path(prefix['predecessor']).read_text(encoding='utf-8'))
    patch = repository_path(record['patch'])
    if sha256(patch) != record['patch_sha256'] or record['basic_scheduler_lifecycle_verified'] is not True:
        raise ValueError('Lifecycle patch/acceptance identity differs')
    for flag in ('full_engine_acceptance', 'speed_mtp_acceptance', 'profile_created', 'wheel_created', 'performance_acceptance'):
        if record[flag] is not False:
            raise ValueError('Development lifecycle evidence cannot grant release/service acceptance')
    raw = {}
    for item in record['evidence']:
        p = repository_path(item['path'])
        data = gzip.decompress(p.read_bytes())
        if sha256(p) != item['sha256'] or hashlib.sha256(data).hexdigest() != item['raw_sha256']:
            raise ValueError('Lifecycle evidence byte drift')
        raw[p.name.removesuffix('.gz')] = data
    actual = json.loads(raw['mtp-lifecycle-v5.json'])
    components = json.loads(raw['lifecycle-components-v1.json'])
    resources = json.loads(raw['mtp-lifecycle-v7-resource.json'])
    if actual['status'] != 'passed' or components['status'] != 'passed' or resources['returncode'] != 0:
        raise ValueError('Actual lifecycle/components/resource run did not pass')
    if b'83 passed, 12 subtests passed' not in raw['lifecycle-components-v1.log']:
        raise ValueError('Lifecycle component log does not support the declared tests')
    runs = {r['label']: r for r in actual['runs']}
    if len(runs) != len(actual['runs']):
        raise ValueError('Duplicate lifecycle run labels')
    for label in ('cap-1', 'cap-2', 'cap-3', 'cap-4', 'cap-5', 'cap-17', 'stop-source',
                  'multi-token-stop', 'actual-eos', 'ignore-eos', 'non-greedy-fallback',
                  'after-valid-rebuild', 'chunked-prefill'):
        native, mtp = runs[label+'-native'], runs[label+'-mtp']
        if native['native_target_only'] is not True or mtp['native_target_only'] is not False:
            raise ValueError('Lifecycle comparison is not target-native versus MTP')
        if native['output']['token_ids'] != mtp['output']['token_ids'] or native['deliveries'] != mtp['deliveries']:
            raise ValueError('Lifecycle actual output/terminal delivery differs: ' + label)
        if sum(d['finished'] for d in mtp['deliveries']) != 1 or not mtp['deliveries'][-1]['finished']:
            raise ValueError('Lifecycle terminal ordering differs: ' + label)
        if label.startswith('cap-') and len(mtp['deliveries']) != int(label[4:]):
            raise ValueError('Lifecycle output cap differs')
    if runs['non-greedy-fallback-mtp']['rounds']:
        raise ValueError('Non-greedy request incorrectly used speculative argmax')
    if runs['multi-token-stop-mtp']['deliveries'][-1]['finish_reason'] != 'stop':
        raise ValueError('Stop-string test did not terminate at a stop')
    if len(runs['inflight-window-cancel']['deliveries']) != 1 or len(actual['abort_records']) != 1:
        raise ValueError('Cancellation leaked a verified bundle into client delivery')
    if actual['errors'][-1]['error'] != 'request aborted':
        raise ValueError('Cancellation terminal acknowledgement differs')
    if runs['cancel-prefix-reuse']['output']['token_ids'] != runs['stop-source-mtp']['output']['token_ids']:
        raise ValueError('Cancellation prefix reuse output differs')
    # These eight artificial prompts terminate at EOS after one or two tokens.
    # They prove repeated cleanup, not sustained decode or retained-prefix eviction.
    if any(not runs[f'prefix-pressure-{i}']['deliveries'][-1]['finished']
           or runs[f'prefix-pressure-{i}']['deliveries'][-1]['finish_reason'] not in ('stop', 'length')
           for i in range(8)):
        raise ValueError('Repeated cache cleanup run did not finish successfully')
    if record['lifecycle_checks'] != actual['checks'] or any(c['matched'] is not True for c in actual['checks']):
        raise ValueError('Lifecycle check summary differs')
    expected = record['replayed_source_sha256']
    with tempfile.TemporaryDirectory(prefix='freetoken-mtp-lifecycle-') as temporary:
        source, _ = prepare(repository_path(foundation['base_profile']), Path(temporary)/'source')
        for i, data in enumerate((foundation, prefix, prior)):
            apply_patch(source, repository_path(data['patch']), Path(temporary)/f'prior-{i}.log')
        before = {p.relative_to(source).as_posix():sha256(p) for p in source.rglob('*') if p.is_file()}
        if any(before.get(name) != digest for name, digest in record['baseline_sha256'].items()):
            raise ValueError('Lifecycle predecessor source differs')
        apply_patch(source, patch, Path(temporary)/'lifecycle.log')
        after = {p.relative_to(source).as_posix():sha256(p) for p in source.rglob('*') if p.is_file()}
        if {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)} != set(expected):
            raise ValueError('Lifecycle patch inventory differs')
        for name, digest in expected.items():
            if after[name] != digest or components['source_sha256'].get(name) != digest:
                raise ValueError('Lifecycle replay differs from tested source')
            if name.startswith('python/') and actual['source_sha256'].get(name) != digest:
                raise ValueError('Lifecycle actual model source differs')
            ast.parse((source/name).read_text(encoding='utf-8'), filename=name)
    return dict(basic_scheduler_lifecycle_verified=True, source_replay_verified=True,
                lifecycle_checks=len(actual['checks']), full_engine_acceptance=False, speed_mtp_acceptance=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('record', type=Path)
    print(json.dumps(validate(parser.parse_args().record)))
