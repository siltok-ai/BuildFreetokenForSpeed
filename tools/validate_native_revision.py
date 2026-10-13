"""Replay a native revision and verify rebuilt materials without granting runtime acceptance."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from prepare_source import ROOT, load_profile, prepare, repository_path, sha256


def source_files(source):
    names = [p for directory in ('python', 'tests') for p in (source/directory).rglob('*') if p.is_file()]
    names.append(source/'pyproject.toml')
    return {p.relative_to(source).as_posix(): sha256(p) for p in names}


def validate(profile_path, materials=None):
    profile, series = load_profile(profile_path)
    for patch in series['patches']:
        result = subprocess.run(['git', 'apply', '--numstat', str(repository_path(patch['path']))],
            capture_output=True, encoding='utf-8', check=True)
        if {line.split('\t', 2)[2] for line in result.stdout.splitlines()} != set(patch['files']):
            raise ValueError('Native revision patch catalog differs: '+patch['path'])
    with tempfile.TemporaryDirectory(prefix='speed-native-replay-') as temporary:
        source, _ = prepare(profile_path, Path(temporary)/'candidate')
        base, _ = prepare(ROOT/'build/profiles/strataq2dev3.json', Path(temporary)/'base')
        actual, before = source_files(source), source_files(base)
        changed = {n for n in actual.keys() | before.keys() if actual.get(n) != before.get(n)}
        if changed != set(series['patches'][-1]['files']):
            raise ValueError('Native revision source delta differs from its catalog')
        for name in changed:
            if name.endswith('.py'):
                ast.parse((source/name).read_text(encoding='utf-8'), filename=name)
        version_tree = ast.parse((source/'python/freetoken/version.py').read_text(encoding='utf-8'))
        if not any(isinstance(n, ast.Constant) and n.value == profile['custom_version'] for n in ast.walk(version_tree)):
            raise ValueError('Native revision source version differs')
        vendor = source/'python/freetoken/kernel/csrc/third_party/strata_mmvq'
        origin = json.loads((vendor/'UPSTREAM.json').read_text(encoding='utf-8'))
        if origin['commit'] != '6674a0065fb96bacde33e3eb10f91a1df86f95f2' or origin['external_runtime_dependency']:
            raise ValueError('Unexpected native MMVQ source or runtime dependency')
        for entry in origin['files']:
            path = (vendor/entry.get('ported_path', entry['upstream_path'])).resolve()
            if not path.is_relative_to(vendor.resolve()):
                raise ValueError('Native vendor path escapes its directory')
            if hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest() != entry['ported_lf_sha256']:
                raise ValueError('Native vendor source identity differs: '+str(path))
        specs = repository_path(profile['spec_file']).read_text(encoding='utf-8').splitlines()
        old_specs = repository_path('build/specs/speed-sm120.txt').read_text(encoding='utf-8').splitlines()
        if specs != old_specs + ['freetoken__gguf_mmvq_r1']:
            raise ValueError('Native revision must retain old specifications and append MMVQ')
        result = dict(version=profile['custom_version'], source_replay_verified=True, changed_files=len(changed),
            licensed_vendor_files_verified=len(origin['files']), aot_count=len(specs), runtime_acceptance=False)
        if materials is not None:
            materials = Path(materials).resolve(strict=True)
            record = json.loads((materials/'build-record.json').read_text(encoding='utf-8'))
            for key in ('source_commit','custom_version','wheel_filename','torch','cuda','apache_tvm_ffi','sm','input_sha256'):
                if record[key] != profile[key]:
                    raise ValueError('Native build record differs from profile: '+key)
            if record.get('binary_reuse') or record['aot_count'] != len(specs) or record['patches'] != series['patches']:
                raise ValueError('Native revision cannot inherit binary-reuse evidence')
            subprocess.run([sys.executable,'-B',str(ROOT/'tools/verify_materials.py'),str(materials)],check=True)
            with zipfile.ZipFile(materials/'wheels'/profile['wheel_filename']) as wheel:
                expected = {p.relative_to(source/'python').as_posix():p.read_bytes()
                    for p in (source/'python/freetoken').rglob('*') if p.is_file() and (p.suffix=='.py' or 'csrc' in p.parts)}
                actual = {n:wheel.read(n) for n in wheel.namelist() if n.startswith('freetoken/')
                    and (n.endswith('.py') or '/csrc/' in n)}
                if actual != expected:
                    raise ValueError('Native wheel Python/CUDA sources differ from replay')
                if not all(any(n.endswith('/'+license_name) for n in wheel.namelist())
                    for license_name in ('LICENSE-strata.txt','LICENSE-ggml.txt')):
                    raise ValueError('Native wheel lacks vendor licenses')
            kernel = json.loads((materials/'kernel-cache/build-metadata.json').read_text(encoding='utf-8'))
            if kernel['source_commit'] != profile['source_commit'] or kernel['source_patch_series_sha256'] != profile['input_sha256'][profile['patch_series']]:
                raise ValueError('Native AOT source provenance differs')
            wheel_log = (materials/'logs/wheel.log').read_text(encoding='utf-8',errors='replace')
            if not all("building 'freetoken.kernel."+n+"' extension" in wheel_log for n in ('_pinned_tensor','_cpu_moe','_row_store')):
                raise ValueError('Native extensions lack full compilation evidence')
            result.update(wheel_and_aot_identity_verified=True,full_native_compilation_recorded=True,
                wheel_sha256=record['wheel_sha256'],build_record_sha256=sha256(materials/'build-record.json'),
                wheel_log_sha256=sha256(materials/'logs/wheel.log'),aot_log_sha256=sha256(materials/'logs/aot.log'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,default=ROOT/'build/profiles/strataq2dev4.json')
    parser.add_argument('--materials',type=Path)
    args=parser.parse_args()
    print(json.dumps(validate(args.profile,args.materials)))

if __name__=='__main__':main()
