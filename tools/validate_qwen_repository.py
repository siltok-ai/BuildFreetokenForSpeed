"""验证 Qwen 双模型候选的冻结来源、补丁范围和实际材料身份。"""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from prepare_source import ROOT, DEFAULT_PROFILE, extract_archive, load_profile, prepare, repository_path, sha256


def tree_hashes(root):
    """比较实际文件字节，防止浏览树或补丁清单掩盖额外改动。"""
    return {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}


def main():
    """重放当前唯一 profile；GPU 状态由独立验收记录裁决。"""
    profile, series = load_profile(DEFAULT_PROFILE)
    lock = json.loads(repository_path(profile['upstream_lock']).read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='qwen-source-') as temporary:
        temporary = Path(temporary)
        original = temporary / 'original'
        extract_archive(repository_path(profile['source_archive']), original)
        original_hashes = tree_hashes(original)
        if lock['snapshot_modified'] or original_hashes != tree_hashes(repository_path(lock['browse_tree'])):
            raise ValueError('official snapshot differs from its frozen archive')
        source, _ = prepare(DEFAULT_PROFILE, temporary / 'prepared')
        actual = tree_hashes(source)
        changed = {n for n in original_hashes.keys() | actual.keys() if original_hashes.get(n) != actual.get(n)}
        declared = [n for p in series['patches'] for n in p['files']]
        if len(declared) != len(set(declared)) or changed != set(declared):
            raise ValueError('patch catalog differs from actual source changes')
        contract = json.loads((ROOT / 'integration/speed/consumer-contract.json').read_text(encoding='utf-8'))
        materials = repository_path(contract['materials'])
        record = json.loads((materials / 'build-record.json').read_text(encoding='utf-8'))
        if record['input_sha256'] != profile['input_sha256'] or record['source_commit'] != lock['commit']:
            raise ValueError('materials were built from different inputs')
        if contract['version'] != profile['custom_version'] or contract['wheel']['sha256'] != record['wheel_sha256']:
            raise ValueError('consumer wheel identity mismatch')
        if contract['kernel_cache']['manifest_sha256'] != sha256(materials / 'kernel-cache/speed-kernel-manifest.json'):
            raise ValueError('consumer kernel identity mismatch')
        with zipfile.ZipFile(materials / 'wheels' / profile['wheel_filename']) as wheel:
            names = wheel.namelist()
            for name in names:
                if name.startswith('freetoken/') and name.endswith('.py') and wheel.read(name) != (source / 'python' / name).read_bytes():
                    raise ValueError(f'wheel source mismatch: {name}')
            native = {Path(n).name.split('.')[0] for n in names if n.endswith('.pyd')}
            if native != set(profile['native_modules']):
                raise ValueError('native inventory differs from profile')
    subprocess.run([sys.executable, '-B', str(ROOT / 'tools/verify_materials.py'), str(materials)], check=True)
    for directory in ('build', 'tools', 'tests'):
        for p in (ROOT / directory).rglob('*.py'):
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    print(json.dumps({'verified': True, 'source_commit': lock['commit'], 'changed_files': sorted(changed),
                      'patch_count': len(series['patches']), 'aot_count': record['aot_count'],
                      'runtime_acceptance': contract['validation_status']}))


if __name__ == '__main__':
    main()
