"""Check current FreeToken materials and the matching Speed runtime contract offline."""
from pathlib import Path
import argparse
import json

from current_release import ROOT, load_current
from prepare_source import load_profile
from validate_python_revision import validate


def verify(contract_path, speed_root=None):
    """Validate build inputs and ABI compatibility; this does not run a model."""
    contract = json.loads(Path(contract_path).read_text(encoding='utf-8'))
    if contract.get('kind') != 'flash_next_portable_resource_consumer':
        raise ValueError('Unsupported current consumer kind')
    profile, _ = load_profile(ROOT/contract['profile'])
    if profile['custom_version'] != contract['version']:
        raise ValueError('Consumer version differs from profile')
    materials = ROOT/contract['materials']
    if speed_root is not None:
        release = json.loads((Path(speed_root)/'silitok_speed/runtimes/freetoken/release.json')
                            .read_text(encoding='utf-8-sig'))
        expected = dict(version=profile['custom_version'], upstream_commit=profile['source_commit'],
                        python_minor=[3, 12], platform='win_amd64', torch=profile['torch'],
                        cuda=profile['cuda'], apache_tvm_ffi=profile['apache_tvm_ffi'], sm=profile['sm'])
        for key, value in expected.items():
            if release.get(key) != value:
                raise ValueError('Speed runtime contract differs: '+key)
        record = json.loads((materials/'build-record.json').read_text(encoding='utf-8'))
        specs = (ROOT/profile['spec_file']).read_text(encoding='utf-8').splitlines()
        if (release['wheel']['path'] != 'wheels/'+profile['wheel_filename']
                or sorted(release['wheel']['native_modules']) != sorted(record['native_modules'])
                or release['kernel_cache']['dll_count'] != len(specs)
                or release['spec_names'] != specs):
            raise ValueError('Speed wheel/AOT inventory differs')
        if (release['mtp_runtime']['tensor_count'] != contract['mtp']['tensor_count']
                or release['mtp_runtime']['tensor_bytes'] != contract['mtp']['runtime_bytes']):
            raise ValueError('Speed MTP layout differs')
    validate(ROOT/contract['profile'], materials, ROOT/contract['base_profile'])
    return dict(status='verified', scope='offline_materials_and_compatibility',
                version=contract['version'], speed_contract_checked=speed_root is not None,
                runtime_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path,
                        default=ROOT/load_current()['consumer_contract'])
    parser.add_argument('--speed-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.contract, args.speed_root)))


if __name__ == '__main__':
    main()
