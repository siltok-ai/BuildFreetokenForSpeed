"""Verify the candidate wheel and actual installed anchor/eviction evidence."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from prepare_source import ROOT, prepare, repository_path, sha256
from validate_mtp_lifecycle_checkpoint import validate as validate_lifecycle


def validate(path):
    record=json.loads(Path(path).read_text(encoding='utf-8'))
    prior=repository_path(record['predecessor'])
    if sha256(prior)!=record['predecessor_sha256']:
        raise ValueError('Installed candidate lifecycle identity drift')
    validate_lifecycle(prior)
    profile_path=repository_path(record['profile'])
    materials=repository_path(record['materials'])
    if sha256(profile_path)!=record['profile_sha256'] or sha256(materials/'build-record.json')!=record['build_record_sha256']:
        raise ValueError('MTP profile/material identity drift')
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/verify_materials.py'),str(materials)],check=True,capture_output=True)
    build=json.loads((materials/'build-record.json').read_text(encoding='utf-8'))
    if build['custom_version']!=record['version'] or build['wheel_sha256']!=record['wheel_sha256']:
        raise ValueError('MTP installed version/wheel identity differs')
    if record['installed_native_sha256']!=build['binary_reuse']['native_sha256']:
        raise ValueError('MTP installed native identity differs')
    if record['source_override'] is not False or record['installed_ownership_verified'] is not True:
        raise ValueError('Source override cannot grant installed candidate acceptance')
    if any(record[k] is not False for k in ('speed_mtp_acceptance','performance_acceptance','full_engine_acceptance')):
        raise ValueError('Ownership evidence cannot grant service/performance/full acceptance')
    raw={}
    for item in record['evidence']:
        p=repository_path(item['path'])
        data=gzip.decompress(p.read_bytes())
        if sha256(p)!=item['sha256'] or hashlib.sha256(data).hexdigest()!=item['raw_sha256']:
            raise ValueError('Installed MTP evidence byte drift')
        raw[p.name.removesuffix('.gz')]=data
    actual=json.loads(raw['mtp1-installed-ownership-v1.json'])
    resources=json.loads(raw['mtp1-installed-ownership-v1-resource.json'])
    if actual['status']!='passed' or actual['source_override'] is not False or resources['returncode']!=0:
        raise ValueError('Installed ownership run did not pass')
    if actual['checks']!=record['ownership_checks']:
        raise ValueError('Installed ownership check summary differs')
    runs={r['label']:r for r in actual['runs']}
    for left,right in (('anchor-native','anchor-mtp'),('anchor-fork-native','anchor-fork-mtp')):
        if runs[left]['deliveries']!=runs[right]['deliveries']:
            raise ValueError('Actual installed anchor/fork outputs differ')
    if not all(actual['checks'][0]['state_exact_equal'].values()) or runs['anchor-fork-mtp']['prefills'][0][0]!=64:
        raise ValueError('Installed anchor state/reuse differs')
    pressure=actual['checks'][2]
    if pressure['retained_prefix_before_pressure']!=384 or pressure['oldest_prefix_after_pressure']>=384:
        raise ValueError('Retained-prefix pressure did not evict an old snapshot')
    for i in range(8):
        run=runs[f'retained-pressure-{i}']
        if run['own_retained_prefix']!=384 or len(run['deliveries'])!=17 or run['deliveries'][-1]['finish_reason']!='length':
            raise ValueError('Retained-prefix pressure length/ownership evidence differs')
    with tempfile.TemporaryDirectory(prefix='freetoken-mtp-installed-') as temp:
        source,profile=prepare(profile_path,Path(temp)/'source')
        expected={p.relative_to(source/'python/freetoken').as_posix():sha256(p)
            for p in (source/'python/freetoken').rglob('*.py')}
        if expected!=actual['package_python_sha256'] or profile['custom_version']!=actual['version']:
            raise ValueError('Actual installed Python differs from the frozen profile')
    return dict(version=record['version'],installed_profile_python_verified=True,installed_ownership_verified=True,
        actual_anchor_and_eviction_runs=len(actual['runs']),native_identity_verified=True,aot_count=build['aot_count'],
        full_engine_acceptance=False,speed_mtp_acceptance=False,performance_acceptance=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('record',type=Path)
    print(json.dumps(validate(p.parse_args().record)))
