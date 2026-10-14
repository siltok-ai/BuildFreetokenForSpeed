"""Verify fail-before/pass-after cancellation and the independent fixed profile."""
from __future__ import annotations
import ast
import gzip
import hashlib
import json
from pathlib import Path
import tempfile

from prepare_source import prepare,repository_path,sha256
from validate_mtp_installed_candidate import validate as validate_parent


def validate(path):
    record=json.loads(Path(path).read_text(encoding='utf-8'))
    prior=repository_path(record['predecessor'])
    if sha256(prior)!=record['predecessor_sha256']:
        raise ValueError('Final-abort predecessor identity drift')
    validate_parent(prior)
    profile=repository_path(record['profile'])
    if sha256(profile)!=record['profile_sha256'] or sha256(repository_path(record['patch']))!=record['patch_sha256']:
        raise ValueError('Final-abort profile/patch identity drift')
    if any(record[k] is not False for k in ('speed_mtp_acceptance','performance_acceptance','full_engine_acceptance')):
        raise ValueError('CPU cancellation regression cannot grant GPU or service acceptance')
    raw={}
    for item in record['evidence']:
        p=repository_path(item['path'])
        data=gzip.decompress(p.read_bytes())
        if sha256(p)!=item['sha256'] or hashlib.sha256(data).hexdigest()!=item['raw_sha256']:
            raise ValueError('Final-abort evidence byte drift')
        raw[p.name.removesuffix('.gz')]=data
    before=json.loads(raw['final-abort-gate-v1.json'])
    fixed=json.loads(raw['final-abort-gate-v2.json'])
    suite=json.loads(raw['final-components-v1.json'])
    if before['status']!='failed' or fixed['status']!='passed' or suite['status']!='passed':
        raise ValueError('Final-abort regression did not fail before and pass after')
    if b'assert req.aborted' not in raw['final-abort-gate-v1.log'] or b'1 passed' not in raw['final-abort-gate-v2.log']:
        raise ValueError('Final-abort logs do not support the stated regression')
    if b'84 passed, 12 subtests passed' not in raw['final-components-v1.log']:
        raise ValueError('Final-abort component suite evidence is incomplete')
    expected=record['replayed_source_sha256']
    scheduler='python/freetoken/scheduler/scheduler.py'
    if before['source_sha256'][scheduler]!=record['baseline_sha256'][scheduler]:
        raise ValueError('Failed regression did not use the predecessor scheduler')
    with tempfile.TemporaryDirectory(prefix='freetoken-mtp-final-abort-') as temp:
        base,_=prepare(repository_path(record['base_profile']),Path(temp)/'base')
        source,_=prepare(profile,Path(temp)/'fixed')
        for name,digest in expected.items():
            if (sha256(base/name)!=record['baseline_sha256'][name] or sha256(source/name)!=digest
                    or fixed['source_sha256'][name]!=digest or suite['source_sha256'][name]!=digest):
                raise ValueError('Final-abort replay differs from tested source')
            ast.parse((source/name).read_text(encoding='utf-8'))
    return dict(version='0.1.3+speed.strataq2mtp2',source_replay_verified=True,final_abort_fail_before_pass_after=True,
                components='84 passed, 12 subtests passed',full_engine_acceptance=False,speed_mtp_acceptance=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('record',type=Path)
    print(json.dumps(validate(p.parse_args().record)))
