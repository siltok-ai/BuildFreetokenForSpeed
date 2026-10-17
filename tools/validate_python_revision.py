"""离线重放性能修订，核对 wheel 源码、原生复用和配套库存，不冒充 GPU 验收。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from prepare_source import ROOT, load_profile, prepare, sha256


def validate(profile_path, materials, base_profile_path=None):
    """核对新版本冻结输入和实际材料，以防旧 binary 或 Python 文件被混入。"""
    profile, series = load_profile(profile_path)
    for patch in series['patches']:
        changes = subprocess.run(['git', 'apply', '--numstat', str(ROOT / patch['path'])],
                                 check=True, capture_output=True, encoding='utf-8').stdout
        if {line.split('\t', 2)[2] for line in changes.splitlines()} != set(patch['files']):
            raise ValueError(f"patch file catalog differs from actual diff: {patch['path']}")
    materials = Path(materials).resolve(strict=True)
    record = json.loads((materials / 'build-record.json').read_text(encoding='utf-8'))
    for key in ('source_commit', 'custom_version', 'wheel_filename', 'torch', 'cuda', 'sm', 'input_sha256'):
        if record[key] != profile[key]:
            raise ValueError(f'Python revision differs from frozen profile: {key}')
    subprocess.run([sys.executable, '-B', str(ROOT / 'tools/verify_materials.py'), str(materials)], check=True)
    wheel = materials / 'wheels' / profile['wheel_filename']
    with tempfile.TemporaryDirectory(prefix='q2opt-source-replay-') as directory:
        source, _ = prepare(profile_path, Path(directory) / 'source')
        if base_profile_path is not None:
            # 新复用修订直接证明相邻来源差异，无需为每个版本复制一套验证器。
            sys.path.insert(0, str(ROOT / 'build'))
            from derive_python_materials import validate_python_only_changes
            base_source, base_profile = prepare(base_profile_path, Path(directory) / 'base')
            changed = validate_python_only_changes(base_source, source)
            if sorted(changed) != record['binary_reuse']['changed_source_files']:
                raise ValueError('Python-only source delta differs from reuse receipt')
        expected = {p.relative_to(source / 'python').as_posix(): p.read_bytes()
                    for p in (source / 'python/freetoken').rglob('*.py')}
        with zipfile.ZipFile(wheel) as archive:
            actual = {name: archive.read(name) for name in archive.namelist()
                      if name.startswith('freetoken/') and name.endswith('.py')}
            if expected != actual:
                raise ValueError('optimized wheel Python source differs from frozen patch replay')
            native = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in record['native_modules']}
            if native != record['binary_reuse']['native_sha256']:
                raise ValueError('optimized wheel native reuse identity mismatch')
    base_name = record['binary_reuse']['base_materials']
    if Path(base_name).name != base_name:
        raise ValueError('base materials must name a repository artifact directory')
    base_record = ROOT / 'artifacts' / base_name / 'build-record.json'
    if sha256(base_record) != record['binary_reuse']['base_build_record_sha256']:
        raise ValueError('base binary provenance changed')
    baseline = json.loads(base_record.read_text(encoding='utf-8'))
    if base_profile_path is not None:
        for key in ('custom_version', 'input_sha256', 'wheel_filename'):
            if baseline[key] != base_profile[key]:
                raise ValueError('Binary reuse base differs from its explicit profile: '+key)
    with zipfile.ZipFile(base_record.parent / 'wheels' / baseline['wheel_filename']) as archive:
        if native != {name: hashlib.sha256(archive.read(name)).hexdigest() for name in native}:
            raise ValueError('native modules differ from accepted base wheel')
    for name, entry in record['artifacts'].items():
        if name.endswith('.dll') and entry != baseline['artifacts'][name]:
            raise ValueError(f'AOT binary differs from accepted base: {name}')
    if record['status'] == 'validated_speed_source_q2_performance_revision':
        evidence = materials / 'validation/performance-20261010'
        index = json.loads((evidence / 'evidence.json').read_text(encoding='utf-8'))
        for name, frozen in index['files'].items():
            path = (evidence / name).resolve()
            if not path.is_relative_to(evidence.resolve()) or path.stat().st_size != frozen['bytes'] or sha256(path) != frozen['sha256']:
                raise ValueError(f'performance evidence identity mismatch: {name}')
        summary = json.loads((evidence / 'validation-summary.json').read_text(encoding='utf-8'))
        if summary['wheel_sha256'] != record['wheel_sha256'] or summary['version'] != profile['custom_version']:
            raise ValueError('performance evidence is bound to a different wheel')
    print(json.dumps({'verified': True, 'scope': 'offline_source_and_binary_identity_only',
                      'version': profile['custom_version'], 'patches': len(series['patches']),
                      'python_files': len(expected), 'aot_count': record['aot_count']}))


def main():
    """允许对仓库材料或独立输出做同样的冻结身份验证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT / 'build/profiles/q2opt1.json')
    parser.add_argument('--materials', type=Path, default=ROOT / 'artifacts/q2opt1-cp312-cu130-sm120')
    parser.add_argument('--base-profile', type=Path,
                        help='Recompute the Python-only source delta against this frozen base')
    args = parser.parse_args()
    validate(args.profile, args.materials, args.base_profile)


if __name__ == '__main__':
    main()
