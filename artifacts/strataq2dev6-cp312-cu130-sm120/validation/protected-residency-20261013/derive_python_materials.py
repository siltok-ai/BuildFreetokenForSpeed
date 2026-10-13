"""在原生输入不变时生成新版本 Python 引擎材料，完整保留旧二进制及来源。"""
import argparse
import base64
import csv
import hashlib
import io
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_source import ROOT, prepare, sha256

# 这些改动不影响原生 ABI、AOT 代码或规格，允许按字节复用已验收二进制。
PYTHON_ONLY_CHANGES = {
    'python/freetoken/moe/fused_q2_0.py',
    'python/freetoken/layers/moe.py',
    'python/freetoken/layers/quantization/moe/q2_0.py',
    'python/freetoken/version.py',
    'tests/moe/test_q2_grouped.py',
    'python/freetoken/models/qwen4_exp/q2_source.py',
    'tests/moe/test_q2_source.py',
    # 显式 GGUF 候选仅改变 Python/Triton；不包含 C++、CUDA 源码或 AOT spec。
    'python/freetoken/attention/qsa_sparse.py',
    'python/freetoken/engine/engine.py',
    'python/freetoken/kernel/gguf_quant.py',
    'python/freetoken/kernel/triton/gguf_gdn.py',
    'python/freetoken/kernel/triton/qsa/attend.py',
    'python/freetoken/kernel/triton/qsa/score.py',
    'python/freetoken/kvcache/__init__.py',
    'python/freetoken/kvcache/base.py',
    'python/freetoken/kvcache/linear_state_pool.py',
    'python/freetoken/kvcache/qsa_pool.py',
    'python/freetoken/layers/quantization/configs/base.py',
    'python/freetoken/layers/quantization/configs/gguf.py',
    'python/freetoken/layers/quantization/linear/gguf.py',
    'python/freetoken/layers/quantization/scheme.py',
    'python/freetoken/models/qwen4_exp/gdn.py',
    'python/freetoken/models/qwen4_exp/gguf_source.py',
    'python/freetoken/models/qwen4_exp/gguf_weight.py',
    'python/freetoken/models/qwen4_exp/model.py',
    'python/freetoken/models/qwen4_exp/moe.py',
    'python/freetoken/models/qwen4_exp/ple_disk.py',
    'python/freetoken/models/qwen4_exp/ple.py',
    'python/freetoken/models/qwen4_exp/weight.py',
    'python/freetoken/moe/fused_q2_q8.py',
    'tests/models/qwen4_exp/test_gguf_gdn.py',
    'tests/models/qwen4_exp/test_gguf_source.py',
    'tests/models/qwen4_exp/test_gguf_execution.py',
    'tests/models/qwen4_exp/test_gguf_ple.py',
    # 保护驻留修订只增加 Python/Triton、配置和固定排名数据，原生输入仍须逐字节一致。
    'python/freetoken/engine/config.py',
    'python/freetoken/server/args.py',
    'python/freetoken/moe/q2_residency.py',
    'python/freetoken/moe/configs/q2_residency/strata_flash_next.json',
    'tests/moe/test_q2_residency.py',
}


def tree_hashes(root):
    """按源文件身份核对差异，防止把原生改动当作纯 Python 修订。"""
    return {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}


def validate_python_only_changes(base_source, source):
    """仅允许明确列出的 Python 算子/来源读取修订，原生或 AOT 输入差异立即拒绝。"""
    old, new = tree_hashes(base_source), tree_hashes(source)
    changed = {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    if not changed <= PYTHON_ONLY_CHANGES:
        raise ValueError(f'binary reuse forbidden for source changes: {sorted(changed - PYTHON_ONLY_CHANGES)}')
    return changed


def derive(profile_path, base_profile_path, base_materials, output):
    """重放新旧冻结源码并核对差异，重新封装 Python，复用且追溯原生字节。"""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new output directory; accepted artifacts are never replaced')
    base_materials = Path(base_materials).resolve(strict=True)
    subprocess.run([sys.executable, '-B', str(ROOT / 'tools/verify_materials.py'), str(base_materials)], check=True)
    source, profile = prepare(profile_path, output / 'candidate-source')
    base_source, base_profile = prepare(base_profile_path, output / 'base-source')
    changed = validate_python_only_changes(base_source, source)
    if profile['custom_version'] == base_profile['custom_version']:
        raise ValueError('changed Python source requires a new custom version')
    for key in ('source_commit', 'torch', 'cuda', 'apache_tvm_ffi', 'sm', 'spec_file', 'requirements_lock'):
        if profile[key] != base_profile[key]:
            raise ValueError(f'binary reuse contract changed: {key}')
    for key in (profile['spec_file'], profile['requirements_lock']):
        if profile['input_sha256'][key] != base_profile['input_sha256'][key]:
            raise ValueError(f'binary reuse input changed: {key}')
    base_record = json.loads((base_materials / 'build-record.json').read_text(encoding='utf-8'))
    for key in ('source_commit', 'custom_version', 'wheel_filename', 'torch', 'cuda', 'apache_tvm_ffi', 'sm', 'input_sha256'):
        if base_record[key] != base_profile[key]:
            raise ValueError(f'base materials differ from frozen profile: {key}')
    base_wheel = base_materials / 'wheels' / base_profile['wheel_filename']
    old_dist = f"freetoken-{base_profile['custom_version']}.dist-info"
    new_dist = f"freetoken-{profile['custom_version']}.dist-info"
    members = {}
    with zipfile.ZipFile(base_wheel) as archive:
        for name in archive.namelist():
            if name.endswith('/RECORD'):
                continue
            data = archive.read(name)
            if (name.startswith('freetoken/') and name.endswith('.py')) or (
                    name.startswith('freetoken/moe/configs/') and name.endswith('.json')):
                if data != (base_source / 'python' / name).read_bytes():
                    raise ValueError(f'base wheel differs from frozen source: {name}')
                data = (source / 'python' / name).read_bytes()
            target = name.replace(old_dist + '/', new_dist + '/', 1)
            if target == new_dist + '/METADATA':
                data = re.sub(rb'(?m)^Version: [^\r\n]*', f"Version: {profile['custom_version']}".encode(), data)
            members[target] = data
    for path in (source / 'python/freetoken').rglob('*.py'):
        members[path.relative_to(source / 'python').as_posix()] = path.read_bytes()
    # 与 setuptools 的 package-data 规则一致；新排名必须进入 wheel 和 RECORD。
    for path in (source / 'python/freetoken/moe/configs').rglob('*.json'):
        members[path.relative_to(source / 'python').as_posix()] = path.read_bytes()
    records = io.StringIO(newline='')
    writer = csv.writer(records, lineterminator='\n')
    for name, data in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([name, 'sha256=' + digest, len(data)])
    writer.writerow([new_dist + '/RECORD', '', ''])
    members[new_dist + '/RECORD'] = records.getvalue().encode()
    wheels = output / 'wheels'
    wheels.mkdir()
    wheel = wheels / profile['wheel_filename']
    with zipfile.ZipFile(wheel, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    cache = output / 'kernel-cache'
    shutil.copytree(base_materials / 'kernel-cache', cache)
    manifest_path = cache / 'speed-kernel-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['freetoken_version'] = profile['custom_version']
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    metadata_path = cache / 'build-metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata['source_patch_series_sha256'] = profile['input_sha256'][profile['patch_series']]
    metadata['binary_reuse'] = {'base_materials': base_materials.name,
                              'base_build_record_sha256': sha256(base_materials / 'build-record.json'),
                              'reason': 'Allowlisted Python-only revision; native/AOT source and inputs identical'}
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    native_hashes = {name: hashlib.sha256(members[name]).hexdigest() for name in base_record['native_modules']}
    with zipfile.ZipFile(base_wheel) as archive:
        if native_hashes != {name: hashlib.sha256(archive.read(name)).hexdigest() for name in native_hashes}:
            raise ValueError('native module bytes changed during repack')
    files = [wheel, *(cache / name.removeprefix('kernel-cache/') for name in base_record['artifacts'] if name.endswith('.dll')),
             manifest_path, metadata_path]
    # 复用构建环境身份，不能继承旧 wheel 的运行验收或证据路径。
    inherited = {key: base_record[key] for key in ('format', 'python', 'build_python', 'native_modules',
                                                  'dependencies', 'cuda_compiler', 'msvc_compiler', 'aot_count')}
    record = {**inherited, **profile, 'status': 'derived_requires_runtime_validation',
              'producer': 'build/derive_python_materials.py',
              'producer_sha256': {'build/derive_python_materials.py': sha256(Path(__file__))},
              'patches': json.loads((output / 'candidate-source/preparation-record.json').read_text())['patches'],
              'binary_reuse': {**metadata['binary_reuse'], 'base_wheel_sha256': sha256(base_wheel),
                               'native_sha256': native_hashes, 'changed_source_files': sorted(changed)},
              'wheel_sha256': sha256(wheel),
              'artifacts': {p.relative_to(output).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files}}
    (output / 'build-record.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    subprocess.run([sys.executable, '-B', str(ROOT / 'tools/verify_materials.py'), str(output)], check=True)
    print(json.dumps({'wheel': str(wheel), 'sha256': record['wheel_sha256'], 'status': record['status']}))


def main():
    """显式指定新旧 profile、基础制品和新输出目录，以保证可重放。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--base-profile', type=Path, default=ROOT / 'build/profiles/q2exp1.json')
    parser.add_argument('--base-materials', type=Path, default=ROOT / 'artifacts/q2exp1-cp312-cu130-sm120')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    derive(args.profile, args.base_profile, args.base_materials, args.output)


if __name__ == '__main__':
    main()
