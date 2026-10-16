"""离线核对性能候选的完整源码、wheel、复用边界及实际GGUF重建回执。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from prepare_source import ROOT, prepare, load_profile
sys.path.insert(0, str(ROOT / 'build'))
from derive_python_materials import validate_gguf_rebuild_changes

GGUF_OLD="freetoken__gguf_mmvq_r2"
GGUF_NEW="freetoken__gguf_mmvq_r3"


def validate(profile_path, materials):
    """分别验证每个Python/原生源文件；26项复用和1项重建均绑定实际字节。"""
    profile, _ = load_profile(profile_path)
    if profile.get('gguf_rebuild')!={'old':GGUF_OLD,'new':GGUF_NEW}:raise ValueError('Wrong bounded GGUF revision')
    materials = Path(materials).resolve(strict=True)
    record = json.loads((materials / 'build-record.json').read_text())
    subprocess.run([sys.executable, '-B', str(ROOT / 'tools/verify_materials.py'), str(materials)], check=True)
    for key in ('custom_version', 'input_sha256', 'source_commit', 'wheel_filename'):
        if record[key] != profile[key]:raise ValueError(f'material profile mismatch: {key}')
    base_materials = ROOT / 'artifacts/strataq2perf6-cp312-cu130-sm120'
    base_record = json.loads((base_materials / 'build-record.json').read_text())
    digest = hashlib.sha256((base_materials / 'build-record.json').read_bytes()).hexdigest()
    if record['binary_reuse']['base_build_record_sha256'] != digest:
        raise ValueError('selective rebuild base material identity changed')
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        source, _ = prepare(profile_path, directory / 'candidate')
        base_source, _ = prepare(ROOT / 'build/profiles/strataq2perf6.json', directory / 'base')
        changed = validate_gguf_rebuild_changes(base_source, source, GGUF_OLD, GGUF_NEW)
        if sorted(changed) != record['binary_reuse']['changed_source_files']:
            raise ValueError('selective rebuild source delta differs from receipt')
        with zipfile.ZipFile(materials / 'wheels' / profile['wheel_filename']) as wheel:
            code = [n for n in wheel.namelist() if n.startswith('freetoken/') and
                    (n.endswith(('.py', '.cuh', '.cu', '.h', '.hpp', '.cpp')) or n.endswith('configs/q2_residency/strata_flash_next.json'))]
            for name in code:
                if wheel.read(name) != (source / 'python' / name).read_bytes():
                    raise ValueError(f'wheel source differs: {name}')
            actual_py = {n for n in code if n.endswith('.py')}
            expected_py = {p.relative_to(source / 'python').as_posix() for p in (source / 'python/freetoken').rglob('*.py')}
            if actual_py != expected_py:raise ValueError('wheel Python inventory incomplete')
            with zipfile.ZipFile(base_materials / 'wheels' / base_record['wheel_filename']) as old:
                for name in base_record['native_modules']:
                    if wheel.read(name) != old.read(name):raise ValueError(f'native module reuse differs: {name}')
        rebuilt = record['selective_rebuild']
        if (rebuilt['spec'], rebuilt['replaces']) != (GGUF_NEW, GGUF_OLD):
            raise ValueError('GGUF r3 rebuild identity missing')
        expected_inputs = {p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in source.rglob('*') if p.is_file() and
                           (p.is_relative_to(source / 'python/freetoken/kernel/csrc') or
                            p.relative_to(source).as_posix() in ('tools/build_windows_kernel_cache.py',
                              'python/freetoken/kernel/gguf_native.py', 'python/freetoken/kernel/aot.py'))}
        if rebuilt['source_sha256'] != expected_inputs:
            raise ValueError('rebuilt GGUF compiler inputs differ from frozen source')
    old_dlls = {n: r for n, r in base_record['artifacts'].items() if n.endswith('.dll') and Path(n).stem != GGUF_OLD}
    new_dlls = {n: r for n, r in record['artifacts'].items() if n.endswith('.dll') and Path(n).stem != GGUF_NEW}
    if len(old_dlls) != 26 or new_dlls != old_dlls:
        raise ValueError('other 26 AOT binaries differ from their accepted base')
    name = 'kernel-cache/' + rebuilt['artifact']['path']
    actual = record['artifacts'][name]
    if actual != {'bytes': rebuilt['artifact']['size'], 'sha256': rebuilt['artifact']['sha256']}:
        raise ValueError('rebuilt GGUF bytes differ from compiler receipt')
    if hashlib.sha256((materials / 'gguf-aot-build.log').read_bytes()).hexdigest() != rebuilt['log_sha256']:
        raise ValueError('GGUF compiler log changed')
    return dict(version=profile['custom_version'], python_files=len(actual_py),
                wheel_source_files=len(code), reused_aot=26, rebuilt_aot=GGUF_NEW, verified=True)


def main():
    """显式传入候选，保持历史默认指针不变。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT / 'build/profiles/strataq2perf7.json')
    parser.add_argument('--materials', type=Path, default=ROOT / 'artifacts/strataq2perf7-cp312-cu130-sm120')
    args = parser.parse_args()
    print(json.dumps(validate(args.profile, args.materials)))


if __name__ == '__main__':main()
