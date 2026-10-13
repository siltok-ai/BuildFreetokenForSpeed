"""Verify a Python/Triton expert revision with the rebuilt dev4 native base."""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile
from prepare_source import ROOT, load_profile, prepare, sha256
from validate_python_revision import validate as validate_python_revision
from validate_native_revision import validate as validate_native_revision, source_files
import sys
sys.path.insert(0,str(ROOT/'build'))
from derive_python_materials import validate_python_only_changes

def validate(profile_path,materials=None):
    profile,series=load_profile(profile_path)
    base_profile=ROOT/'build/profiles/strataq2dev4.json'
    with tempfile.TemporaryDirectory(prefix='routed-expert-replay-') as temporary:
        source,_=prepare(profile_path,Path(temporary)/'source')
        base,_=prepare(base_profile,Path(temporary)/'base')
        changed=validate_python_only_changes(base,source)
        if changed!=set(series['patches'][-1]['files']):raise ValueError('Routed revision patch/source delta differs')
        expected={'python/freetoken/moe/fused_q2_q8.py','python/freetoken/version.py','tests/moe/test_q2_grouped.py'}
        if changed!=expected:raise ValueError('Routed revision has unexpected scope')
        if profile['spec_file']!='build/specs/speed-strataq2dev4-sm120.txt':raise ValueError('Native specification changed')
        result=dict(version=profile['custom_version'],source_replay_verified=True,changed_files=len(changed),native_inputs_unchanged=True,
            runtime_acceptance=False)
        if materials is not None:
            materials=Path(materials)
            validate_python_revision(profile_path,materials)
            record=json.loads((materials/'build-record.json').read_text(encoding='utf-8'))
            if record['binary_reuse']['base_materials']!='strataq2dev4-cp312-cu130-sm120':raise ValueError('Unexpected native base')
            if set(record['binary_reuse']['changed_source_files'])!=changed:raise ValueError('Derived source delta differs')
            native=validate_native_revision(base_profile,ROOT/'artifacts/strataq2dev4-cp312-cu130-sm120')
            if not native['full_native_compilation_recorded']:raise ValueError('Unproven native base')
            with zipfile.ZipFile(materials/'wheels'/profile['wheel_filename']) as wheel:
                csrc={p.relative_to(source/'python').as_posix():p.read_bytes() for p in (source/'python/freetoken/kernel/csrc').rglob('*') if p.is_file()}
                if csrc!={n:wheel.read(n) for n in wheel.namelist() if n.startswith('freetoken/kernel/csrc/')}:raise ValueError('Derived CUDA source bytes differ')
                for license_name in ['LICENSE-strata.txt','LICENSE-ggml.txt']:
                    if not any(n.endswith('/'+license_name) for n in wheel.namelist()):raise ValueError('Missing inherited license')
            result.update(wheel_and_binary_identity_verified=True,licensed_native_base_verified=True,aot_count=record['aot_count'],
                wheel_sha256=record['wheel_sha256'],build_record_sha256=sha256(materials/'build-record.json'))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,default=ROOT/'build/profiles/strataq2dev5.json')
    parser.add_argument('--materials',type=Path)
    args=parser.parse_args()
    print(json.dumps(validate(args.profile,args.materials)))
if __name__=='__main__':main()
