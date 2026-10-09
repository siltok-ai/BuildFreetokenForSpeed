"""复制后核对定制wheel和DLL的完整库存及实际SHA-256。"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from email.parser import BytesParser

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('materials', type=Path)
    args = parser.parse_args()
    root = args.materials.resolve(strict=True)
    record = json.loads((root / 'build-record.json').read_text(encoding='utf-8'))
    for name, expected in record['artifacts'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('artifact path escapes materials directory')
        if path.stat().st_size != expected['bytes']:
            raise ValueError(f'artifact size mismatch: {name}')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected['sha256']:
                raise ValueError(f'artifact SHA-256 mismatch: {name}')
    dlls = [name for name in record['artifacts'] if name.endswith('.dll')]
    if len(dlls) != record['aot_count']:
        raise ValueError('AOT inventory count mismatch')
    wheel = root / 'wheels' / record['wheel_filename']
    with wheel.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != record['wheel_sha256']:
            raise ValueError('wheel identity differs from record')
    actual_dlls = {path.relative_to(root).as_posix() for path in (root / 'kernel-cache').rglob('*.dll')}
    if actual_dlls != set(dlls):
        raise ValueError('extra or missing DLL in kernel cache')
    with zipfile.ZipFile(wheel) as archive:
        if sorted(name for name in archive.namelist() if name.endswith('.pyd')) != sorted(record['native_modules']):
            raise ValueError('wheel native module inventory mismatch')
        metadata = [name for name in archive.namelist() if name.endswith('.dist-info/METADATA')]
        if len(metadata) != 1 or BytesParser().parsebytes(archive.read(metadata[0]))['Version'] != record['custom_version']:
            raise ValueError('wheel metadata version mismatch')
    manifest = json.loads((root / 'kernel-cache/speed-kernel-manifest.json').read_text(encoding='utf-8'))
    for key, expected in {'freetoken_version': record['custom_version'], 'torch_version': record['torch'],
                          'torch_cuda': record['cuda'], 'python_minor': [3, 12], 'cuda_capabilities': [[12, 0]]}.items():
        if manifest[key] != expected:
            raise ValueError(f'Speed kernel manifest contract mismatch: {key}')
    listed = {'kernel-cache/' + entry['path']: entry['size'] for entry in manifest['artifacts']}
    if listed != {name: record['artifacts'][name]['bytes'] for name in dlls}:
        raise ValueError('Speed kernel manifest inventory mismatch')
    if set(manifest['spec_names']) != {Path(name).stem for name in dlls}:
        raise ValueError('Speed kernel manifest specifications mismatch')
    print(json.dumps({'custom_version': record['custom_version'], 'aot_count': len(dlls),
                      'wheel_sha256': record['wheel_sha256'], 'status': record['status'], 'verified': True}))

if __name__ == '__main__':
    main()
