"""核对 Clover 候选源码与 Speed 合同；二进制需显式传入另行检查。"""
import argparse
import ast
import json
from pathlib import Path
import tempfile

from prepare_source import ROOT, load_profile, prepare, sha256


def verify(speed_root=None, materials=None):
    """重放所有补丁并解析源码，明确区分源码检查和新材料检查。"""
    profile_path = ROOT/'build/profiles/cloverq8.json'
    candidate = json.loads((ROOT/'integration/speed/candidate-cloverq8.json').read_text(encoding='utf-8'))
    for name, digest in candidate['input_sha256'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or sha256(path) != digest:
            raise ValueError('Clover candidate input changed: '+name)
    profile, series = load_profile(profile_path)
    specs = (ROOT/profile['spec_file']).read_text(encoding='utf-8').splitlines()
    previous = json.loads((ROOT/'patches/mtpdecode2-series.json').read_text(encoding='utf-8'))
    if series['patches'][:-1] != previous['patches']:
        raise ValueError('Historical patches changed')
    for name in series['patches'][-1]['files']:
        if name.startswith('docs/') or '/evidence/' in name:
            raise ValueError('Historical documentation/evidence entered the patch')
    with tempfile.TemporaryDirectory(prefix='clover-source-') as directory:
        source, _ = prepare(profile_path, Path(directory)/'prepared')
        for path in source.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        version = (source/'python/freetoken/version.py').read_text(encoding='utf-8').strip()
        if version != '__version__ = "'+profile['custom_version']+'"':
            raise ValueError('Candidate version disagrees with replay')
    if speed_root is not None:
        release = json.loads((Path(speed_root)/'silitok_speed/runtimes/freetoken/release.json').read_text(encoding='utf-8'))
        native = ['freetoken/kernel/'+name+'.cp312-win_amd64.pyd' for name in profile['native_modules']]
        if (release['version'] != profile['custom_version']
                or release['wheel']['path'] != 'wheels/'+profile['wheel_filename']
                or sorted(release['wheel']['native_modules']) != sorted(native)
                or release['spec_names'] != specs or release['kernel_cache']['dll_count'] != len(specs)
                or release['producer_profile'] != 'build/profiles/cloverq8.json'
                or any(release[k] != profile[k] for k in ('torch','cuda','apache_tvm_ffi','sm'))):
            raise ValueError('Speed and Clover producer contracts differ')
    if materials is not None:
        import subprocess
        import sys
        record = json.loads((Path(materials)/'build-record.json').read_text(encoding='utf-8'))
        if (record['custom_version'] != profile['custom_version']
                or record['input_sha256'] != profile['input_sha256'] or record['aot_count'] != len(specs)
                or sorted(record['native_modules']) != sorted(
                    'freetoken/kernel/'+name+'.cp312-win_amd64.pyd' for name in profile['native_modules'])):
            raise ValueError('Materials do not match the candidate source/profile')
        subprocess.run([sys.executable, '-B', str(ROOT/'tools/verify_materials.py'), str(materials)], check=True)
    return dict(status='verified', scope='source_and_contract', version=profile['custom_version'],
        patches=len(series['patches']), aot_specs=len(specs), materials_checked=materials is not None,
        gpu_executed=False, runtime_qualified=False)


def main():
    """允许在 GPU 忙碌时完成候选审查，不接触设备。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--speed-root', type=Path)
    parser.add_argument('--materials', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.speed_root, args.materials)))


if __name__ == '__main__':
    main()
