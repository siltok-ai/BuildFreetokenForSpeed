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
    # A mapped CUDA byte view reuses the existing pinned extension ABI unchanged.
    'python/freetoken/kernel/pinned.py',
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
    'python/freetoken/kernel/gguf_native.py',
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
    # MTP 调度、状态与 Triton 草稿图；逐文件白名单，仍禁止原生源码与 AOT 输入变化。
    'python/freetoken/attention/linear.py',
    'python/freetoken/core.py',
    'python/freetoken/kvcache/mtp_pool.py',
    'python/freetoken/models/config.py',
    'python/freetoken/models/qwen4_exp/mtp_draft_graph.py',
    'python/freetoken/models/qwen4_exp/mtp_drafter.py',
    'python/freetoken/models/qwen4_exp/mtp_runtime.py',
    'python/freetoken/models/qwen4_exp/mtp_state.py',
    'python/freetoken/scheduler/cache.py',
    'python/freetoken/scheduler/scheduler.py',
    'python/freetoken/scheduler/status.py',
    'tests/kvcache/test_linear_state_pool_alloc.py',
    'tests/kvcache/test_mtp_pool.py',
    'tests/kvcache/test_qsa_pool.py',
    'tests/models/qwen4_exp/test_mtp_runtime.py',
    'tests/scheduler/test_hybrid_cache_manager.py',
    'tests/scheduler/test_abort_inflight_prefill.py',
    # Compact transactions and online ownership exchange are Python/Triton only.
    'python/freetoken/kernel/triton/gguf_gdn_compact.py',
    'python/freetoken/moe/q2_adaptive.py',
    'tests/models/qwen4_exp/test_mtp_compact.py',
    'tests/moe/test_q2_adaptive.py',
    'tests/moe/test_q2_q8_tiles.py',
    'tests/moe/test_q2_lut.py',
    'tests/kernels/test_row_cache.py',
    'python/freetoken/moe/prefill_workspace.py',
    'tests/moe/test_prefill_workspace.py',
    'python/freetoken/models/qwen4_exp/mtp_draft_policy.py',
    'python/freetoken/models/qwen4_exp/mtp_draft_vocab.py',
    'tests/models/qwen4_exp/test_mtp_draft_policy.py',
    # 私有 MTP 验证元数据、卷积及对应 GPU 回归，不改变原生 ABI 或 AOT 规格。
    'python/freetoken/kernel/triton/gguf_gdn_conv.py',
    'python/freetoken/kernel/triton/mtp_metadata.py',
    'tests/models/qwen4_exp/test_mtp_decode_kernels.py',
    'python/freetoken/checkpoint/gguf_reader.py',
    # Stop matching, request-local budgets and sampler bounds do not change native/AOT inputs.
    'python/freetoken/engine/sample.py',
    'tests/engine/test_sampling_bounds.py',
    'tests/scheduler/test_stop_boundaries.py',
    # Offline result projection changes Python state only; native and AOT sources stay frozen.
    'python/freetoken/llm/llm.py',
    'tests/llm/test_stop_projection.py',
}


def tree_hashes(root):
    """按源文件身份核对差异，防止把原生改动当作纯 Python 修订。"""
    return {p.relative_to(root).as_posix(): sha256(p) for p in root.rglob('*') if p.is_file()}


def source_changes(before, after):
    """为每个复用边界使用同一份逐文件差异计算。"""
    old, new = tree_hashes(before), tree_hashes(after)
    return {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}


def validate_python_only_changes(base_source, source):
    """仅允许明确列出的 Python 算子/来源读取修订，原生或 AOT 输入差异立即拒绝。"""
    changed = source_changes(base_source, source)
    if not changed <= PYTHON_ONLY_CHANGES:
        raise ValueError(f'binary reuse forbidden for source changes: {sorted(changed - PYTHON_ONLY_CHANGES)}')
    return changed


GGUF_HEADER = 'python/freetoken/kernel/csrc/jit/gguf_mmvq.cuh'
GGUF_AOT = 'python/freetoken/kernel/aot.py'
WINDOWS_BUILDER = 'tools/build_windows_kernel_cache.py'
GGUF_OLD = 'freetoken__gguf_mmvq_r1'
GGUF_NEW = 'freetoken__gguf_mmvq_r2'
ROW_STORE_CPP = 'python/freetoken/kernel/csrc/row_store/row_store_ext.cpp'
PINNED_TENSOR_CPP = 'python/freetoken/kernel/csrc/pinned_tensor.cpp'
_EXTENSION_SOURCES = {'_row_store': ROW_STORE_CPP, '_pinned_tensor': PINNED_TENSOR_CPP}


def validate_row_store_changes(base_source, source):
    """单独重编行存储时只开放该C++入口，其他原生/AOT输入仍逐字节保持。"""
    return _validate_extension_changes(base_source, source, '_row_store')


def rebuild_row_store(source, output, torch_version):
    """实际编译一个CPU扩展；旧的另外两个扩展及27项AOT不进入此编译过程。"""
    return _rebuild_extension(source, output, torch_version, '_row_store')


def validate_pinned_tensor_changes(base_source, source):
    """工作区原生改动只开放pinned扩展入口；其他扩展和AOT输入必须保持。"""
    return _validate_extension_changes(base_source, source, '_pinned_tensor')


def rebuild_pinned_tensor(source, output, torch_version):
    """实际重编CUDA主存扩展并保存回执，不触碰其余两个扩展和27项AOT。"""
    return _rebuild_extension(source, output, torch_version, '_pinned_tensor')


def _validate_extension_changes(base_source, source, module):
    """每次只开放指定扩展的真实源码变化，不扩大原生复用边界。"""
    changed = source_changes(base_source, source)
    native_source = _EXTENSION_SOURCES[module]
    if native_source not in changed or not changed <= PYTHON_ONLY_CHANGES | {native_source}:
        label = module.removeprefix('_').replace('_', '-')
        raise ValueError(f'{label} selective rebuild source boundary differs: {sorted(changed - PYTHON_ONLY_CHANGES)}')
    return changed


def _rebuild_extension(source, output, torch_version, module):
    """执行独立编译器，核对源码、产物与回执；保留两个扩展原有目录和记录格式。"""
    name = module.removeprefix('_')
    label = name.replace('_', '-')
    native_source = _EXTENSION_SOURCES[module]
    destination = output / f'rebuilt-{label}'
    command = [sys.executable, '-B', str(ROOT/f'build/compile_{name}.py'),
               '--source', str(source/native_source), '--output', str(destination), '--torch', torch_version]
    log = output / f'{label}-build.log'
    with log.open('wb') as stream:
        subprocess.run(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT, check=True)
    receipt_path = destination/'build-receipt.json'
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt['source_sha256'] != sha256(source/native_source):
        raise ValueError(f'{label} compiler receipt belongs to different source')
    binary = (destination/receipt['artifact']['path']).resolve(strict=True)
    if not binary.is_relative_to(destination.resolve()) or sha256(binary) != receipt['artifact']['sha256']:
        raise ValueError(f'{label} compiler output differs from its receipt')
    return binary, dict(module=module, source=native_source, command=command,
        receipt=receipt, receipt_path=receipt_path.relative_to(output).as_posix(),
        receipt_sha256=sha256(receipt_path), log_sha256=sha256(log))


def validate_gguf_rebuild_changes(base_source, source, old_spec=GGUF_OLD, new_spec=GGUF_NEW):
    """只允许GGUF独立头文件及其命名/路径引号改变，其他原生输入仍须相同。"""
    changed = source_changes(base_source, source)
    native = {GGUF_HEADER, GGUF_AOT, WINDOWS_BUILDER}
    if not changed <= PYTHON_ONLY_CHANGES | native or not {GGUF_HEADER, GGUF_AOT} <= changed:
        raise ValueError(f'GGUF selective rebuild source boundary differs: {sorted(changed - PYTHON_ONLY_CHANGES)}')
    expected = (base_source / GGUF_AOT).read_text().replace(old_spec.removeprefix('freetoken__'), new_spec.removeprefix('freetoken__'))
    if (source / GGUF_AOT).read_text() != expected:
        raise ValueError('unrelated AOT spec changed during GGUF-only rebuild')
    before = (base_source / WINDOWS_BUILDER).read_text()
    expected = before.replace('f"ldflags = /LIBPATH:{cuda_lib} cudart.lib ",',
                              'f\'ldflags = /LIBPATH:"{cuda_lib}" cudart.lib \',')
    if (source / WINDOWS_BUILDER).read_text() != expected:
        raise ValueError('builder change exceeds quoting the CUDA library path')
    # gguf_mmvq.cuh是单独编译入口，任何其他CUDA源引用它都会使选择性重建失效。
    for path in (source / 'python/freetoken/kernel/csrc').rglob('*'):
        if path.is_file() and path.suffix in ('.cuh', '.cu', '.h', '.cpp') and path != source / GGUF_HEADER:
            if 'gguf_mmvq.cuh' in path.read_text(encoding='utf-8'):
                raise ValueError(f'GGUF header is shared by another native input: {path}')
    return changed


def rebuild_gguf_cache(source, output, base_wheel, native_modules, cache, old_spec=GGUF_OLD, new_spec=GGUF_NEW):
    """用冻结r2源码实际编译一个AOT，保留编译日志、输入哈希和实际工具链身份。"""
    with zipfile.ZipFile(base_wheel) as wheel:
        for name in native_modules:
            target = source / 'python' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(wheel.read(name))
    rebuilt = output / 'rebuilt-cache'
    command = [sys.executable, '-B', str(source / WINDOWS_BUILDER), '--output', str(rebuilt),
               '--build-dir', str(output / 'gguf-aot-build'), '--spec', new_spec,
               '--arch', '12.0', '--jobs', '1', '--keep-build']
    log = output / 'gguf-aot-build.log'
    with log.open('wb') as stream:
        subprocess.run(command, cwd=source, stdout=stream, stderr=subprocess.STDOUT, check=True)
    old_dir = (cache / old_spec).resolve(strict=True)
    if old_dir.parent != cache.resolve():
        raise ValueError('old GGUF cache path escapes the new private output')
    shutil.rmtree(old_dir)
    shutil.copytree(rebuilt / new_spec, cache / new_spec)
    dll = cache / new_spec / (new_spec + '.dll')
    compiler = subprocess.run(['nvcc', '--version'], capture_output=True, check=True).stdout.decode('utf-8', errors='replace')
    msvc = subprocess.run(['cl'], capture_output=True)
    inputs = {p.relative_to(source).as_posix(): sha256(p) for p in source.rglob('*')
              if p.is_file() and (p.is_relative_to(source / 'python/freetoken/kernel/csrc')
                                  or p.relative_to(source).as_posix() in (WINDOWS_BUILDER, 'python/freetoken/kernel/gguf_native.py', GGUF_AOT))}
    return {'spec': new_spec, 'replaces': old_spec, 'command': command,
            'cuda_compiler': compiler, 'msvc_compiler': (msvc.stdout + msvc.stderr).decode('utf-8', errors='replace'),
            'build_python': sys.executable, 'source_sha256': inputs, 'log_sha256': sha256(log),
            'metadata': json.loads((rebuilt / 'build-metadata.json').read_text()),
            'artifact': {'path': dll.relative_to(cache).as_posix(), 'size': dll.stat().st_size, 'sha256': sha256(dll)}}


def derive(profile_path, base_profile_path, base_materials, output, *, rebuild_gguf=False, rebuild_row_store_module=False, rebuild_pinned_module=False):
    """重放新旧源码，默认仅复用原生字节；显式r2模式实际重建独立GGUF AOT。"""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new output directory; accepted artifacts are never replaced')
    if sum((rebuild_gguf, rebuild_row_store_module, rebuild_pinned_module)) > 1:
        raise ValueError('Select one independently bounded native rebuild at a time')
    base_materials = Path(base_materials).resolve(strict=True)
    subprocess.run([sys.executable, '-B', str(ROOT / 'tools/verify_materials.py'), str(base_materials)], check=True)
    source, profile = prepare(profile_path, output / 'candidate-source')
    base_source, base_profile = prepare(base_profile_path, output / 'base-source')
    gguf_revision=profile.get('gguf_rebuild', {})
    old_gguf=gguf_revision.get('old', GGUF_OLD)
    new_gguf=gguf_revision.get('new', GGUF_NEW)
    if not all(re.fullmatch(r'freetoken__gguf_mmvq_r[1-9][0-9]*', name) for name in (old_gguf,new_gguf)) or old_gguf==new_gguf:
        raise ValueError('Invalid independently versioned GGUF kernel names')
    boundary = (validate_pinned_tensor_changes if rebuild_pinned_module else
                validate_row_store_changes if rebuild_row_store_module else
                (lambda before,after:validate_gguf_rebuild_changes(before,after,old_gguf,new_gguf)) if rebuild_gguf else validate_python_only_changes)
    changed = boundary(base_source, source)
    if profile['custom_version'] == base_profile['custom_version']:
        raise ValueError('changed Python source requires a new custom version')
    for key in ('source_commit', 'torch', 'cuda', 'apache_tvm_ffi', 'sm', 'requirements_lock'):
        if profile[key] != base_profile[key]:
            raise ValueError(f'binary reuse contract changed: {key}')
    for key in (profile['requirements_lock'],):
        if profile['input_sha256'][key] != base_profile['input_sha256'][key]:
            raise ValueError(f'binary reuse input changed: {key}')
    old_specs = (ROOT / base_profile['spec_file']).read_text().splitlines()
    new_specs = (ROOT / profile['spec_file']).read_text().splitlines()
    expected_specs = [s.replace(old_gguf, new_gguf) for s in old_specs] if rebuild_gguf else old_specs
    if new_specs != expected_specs or (not rebuild_gguf and profile['spec_file'] != base_profile['spec_file']):
        raise ValueError('AOT inventory changed beyond the selected GGUF rebuild')
    if not rebuild_gguf and profile['input_sha256'][profile['spec_file']] != base_profile['input_sha256'][base_profile['spec_file']]:
        raise ValueError('binary reuse AOT specification input changed')
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
    if rebuild_gguf:
        members[GGUF_HEADER.removeprefix('python/')] = (source / GGUF_HEADER).read_bytes()
    row_rebuild = None
    row_member = None
    if rebuild_row_store_module:
        row_member = next(name for name in base_record['native_modules'] if Path(name).name.startswith('_row_store.'))
        binary, row_rebuild = rebuild_row_store(source, output, profile['torch'])
        base_profile_file = Path(base_profile_path)
        if not base_profile_file.is_absolute():
            base_profile_file = ROOT/base_profile_file
        row_rebuild['base_profile'] = base_profile_file.resolve(strict=True).relative_to(ROOT.resolve()).as_posix()
        members[row_member] = binary.read_bytes()
        members[ROW_STORE_CPP.removeprefix('python/')] = (source/ROW_STORE_CPP).read_bytes()
    pinned_rebuild = None
    pinned_member = None
    if rebuild_pinned_module:
        pinned_member = next(name for name in base_record['native_modules'] if Path(name).name.startswith('_pinned_tensor.'))
        binary, pinned_rebuild = rebuild_pinned_tensor(source, output, profile['torch'])
        base_profile_file = Path(base_profile_path)
        if not base_profile_file.is_absolute():
            base_profile_file = ROOT/base_profile_file
        pinned_rebuild['base_profile'] = base_profile_file.resolve(strict=True).relative_to(ROOT.resolve()).as_posix()
        members[pinned_member] = binary.read_bytes()
        members[PINNED_TENSOR_CPP.removeprefix('python/')] = (source/PINNED_TENSOR_CPP).read_bytes()
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
    rebuild = rebuild_gguf_cache(source, output, base_wheel, base_record['native_modules'], cache,old_gguf,new_gguf) if rebuild_gguf else None
    manifest_path = cache / 'speed-kernel-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['freetoken_version'] = profile['custom_version']
    if rebuild:
        manifest['spec_names'] = new_specs
        manifest['artifacts'] = [({'path': rebuild['artifact']['path'], 'size': rebuild['artifact']['size']}
                                  if Path(a['path']).stem == old_gguf else a) for a in manifest['artifacts']]
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    metadata_path = cache / 'build-metadata.json'
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    metadata['source_patch_series_sha256'] = profile['input_sha256'][profile['patch_series']]
    metadata['binary_reuse'] = {'base_materials': base_materials.name,
                              'base_build_record_sha256': sha256(base_materials / 'build-record.json'),
                              'reason': 'Allowlisted Python-only revision; native/AOT source and inputs identical'}
    if pinned_rebuild:
        metadata['binary_reuse']['reason'] = 'Two native modules and all AOT inputs unchanged; pinned-tensor extension actually rebuilt'
        metadata['pinned_tensor_rebuild'] = pinned_rebuild
    if row_rebuild:
        metadata['binary_reuse']['reason'] = 'Two native modules and all AOT inputs unchanged; row-store CPU extension actually rebuilt'
        metadata['row_store_rebuild'] = row_rebuild
    if rebuild:
        metadata['binary_reuse']['reason'] = 'Three native modules and 26 independent AOT inputs unchanged; '+new_gguf+' actually rebuilt'
        metadata['selective_rebuild'] = rebuild
        metadata['artifacts'] = [({'name': new_gguf, **rebuild['artifact']}
                                  if a['name'] == old_gguf else a) for a in metadata['artifacts']]
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    native_hashes = {name: hashlib.sha256(members[name]).hexdigest() for name in base_record['native_modules']}
    with zipfile.ZipFile(base_wheel) as archive:
        reused = {name: digest for name,digest in native_hashes.items() if name not in (row_member, pinned_member)}
        if reused != {name: hashlib.sha256(archive.read(name)).hexdigest() for name in reused}:
            raise ValueError('native module bytes changed during repack')
    files = [wheel, *(cache / a['path'] for a in manifest['artifacts']),
             manifest_path, metadata_path]
    if rebuild:
        files.append(output / 'gguf-aot-build.log')
    if row_rebuild:
        files.extend([output/'row-store-build.log',output/row_rebuild['receipt_path']])
    if pinned_rebuild:
        files.extend([output/'pinned-tensor-build.log',output/pinned_rebuild['receipt_path']])
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
    if rebuild:
        record['selective_rebuild'] = rebuild
        record['status'] = 'partially_rebuilt_requires_runtime_validation'
        record['binary_reuse']['aot_sha256'] = {name: data['sha256'] for name, data in record['artifacts'].items()
                                               if name.endswith('.dll') and Path(name).stem != new_gguf}
    if row_rebuild:
        record['status'] = 'partially_rebuilt_requires_runtime_validation'
        record['row_store_rebuild'] = row_rebuild
        record['native_sha256'] = native_hashes
        record['binary_reuse']['native_sha256'] = reused
        record['producer_sha256']['build/compile_row_store.py'] = sha256(ROOT/'build/compile_row_store.py')
    if pinned_rebuild:
        record['status'] = 'partially_rebuilt_requires_runtime_validation'
        record['pinned_tensor_rebuild'] = pinned_rebuild
        record['native_sha256'] = native_hashes
        record['binary_reuse']['native_sha256'] = reused
        record['producer_sha256']['build/compile_pinned_tensor.py'] = sha256(ROOT/'build/compile_pinned_tensor.py')
    (output / 'build-record.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    subprocess.run([sys.executable, '-B', str(ROOT / 'tools/verify_materials.py'), str(output)], check=True)
    print(json.dumps({'wheel': str(wheel), 'sha256': record['wheel_sha256'], 'status': record['status']}))


def main():
    """显式指定新旧 profile、基础制品和新输出目录，以保证可重放。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--base-profile', type=Path, required=True)
    parser.add_argument('--base-materials', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rebuild-gguf', action='store_true', help='rebuild isolated GGUF r2 AOT; reject other native changes')
    parser.add_argument('--rebuild-row-store', action='store_true', help='actually rebuild only the row-store CPU extension')
    parser.add_argument('--rebuild-pinned-tensor', action='store_true', help='actually rebuild only the pinned-tensor CUDA runtime extension')
    args = parser.parse_args()
    derive(args.profile, args.base_profile, args.base_materials, args.output,
           rebuild_gguf=args.rebuild_gguf, rebuild_row_store_module=args.rebuild_row_store,
           rebuild_pinned_module=args.rebuild_pinned_tensor)


if __name__ == '__main__':
    main()
