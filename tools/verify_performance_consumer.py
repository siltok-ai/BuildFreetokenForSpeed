"""离线核对性能材料、质量、实际SSE和编译包身份，明确暂存/安装范围。"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import statistics
import subprocess
import zipfile

from prepare_source import load_profile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'integration/speed/consumer-contract-strataq2perf2.json'


def require(condition, message):
    """合同不满足时立即失败，不能把缺失验收降为警告。"""
    if not condition:
        raise ValueError(message)


def contained(root, name):
    """只允许引用对应仓库或证据目录内的真实文件。"""
    path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root.resolve()), f'Path escapes evidence root: {name}')
    return path


def read(path):
    """读取冻结 JSON，不导入产品或创建 CUDA 上下文。"""
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    """按原始字节验证材料和证据身份。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def performance(report):
    """从实际到达时序重算六次 512 输出及中位数，拒绝只填统计数字的报告。"""
    require(report['status'] == 'measured' and report['worker_pids_gone'], 'HTTP run incomplete')
    require(len(report['runs']) == 6, 'Expected six actual HTTP outputs')
    grouped = {}
    for row in report['runs']:
        times = row['arrival_offsets']
        context = row['label'].rsplit('-', 1)[0]
        require(context in ('70-0', 'diverse-4096-0'), 'Unrecognized performance input')
        require(row['usage']['prompt_tokens'] == {'70-0': 70, 'diverse-4096-0': 4096}[context],
                'Frozen performance input length changed')
        require(row['output_tokens'] == row['usage']['completion_tokens'] ==
                row['content_chunk_count'] == len(times) == 512, 'Not 512 actual SSE outputs')
        require(row['done_events'] == 1 and row['finish_reason'] == 'length', 'SSE terminal mismatch')
        require(all(math.isfinite(t) for t in times) and
                all(b >= a for a, b in zip(times, times[1:])) and times[-1] > times[0],
                'Invalid token arrival times')
        rate = 511 / (times[-1] - times[0])
        require(math.isclose(rate, row['decode_tps'], rel_tol=1e-7), 'Decode timing mismatch')
        grouped.setdefault(context, []).append(rate)
    result = {}
    for context, rates in grouped.items():
        require(len(rates) == 3, 'Expected three repetitions per context')
        result[context] = statistics.median(rates)
        require(math.isclose(result[context], report['contexts'][context]['median_tps'], rel_tol=1e-7),
                'HTTP median mismatch')
    return result


def installation(before, ownership, result, bundle_sha256):
    """将真实安装回执绑定到新建测试目录及服务，拒绝把暂存目录冒充安装验收。"""
    root = PureWindowsPath(ownership['install_path'])
    require(root.parent == PureWindowsPath(r'C:\workspace\build\siltok-speed') and
            len(root.name) == 3 and root.name[0] == 'i' and root.name[1:].isdigit(),
            'Not an isolated installer instance')
    require(PureWindowsPath(before['install_path']) == root and
            not any(before[key] for key in ('install_present', 'data_present', 'service',
                                            'event_source_present', 'uninstall_keys')),
            'Installation baseline was not fresh')
    service_path = ownership['service']['path']
    executable = service_path.split('"')[1] if service_path.startswith('"') else service_path.split()[0]
    require(ownership['phase'] == 'installed' and ownership['install_present'] and
            ownership['data_present'] and ownership['bundle_sha256'] == bundle_sha256 and
            PureWindowsPath(executable) == root / 'SiltokSpeed.exe', 'Installer ownership mismatch')
    require(result['status'] == 'installed_service_stopped' and result['launcher_admin'] is True and
            result['installer_exitcode'] == 0 and PureWindowsPath(result['install_path']) == root,
            'Installation did not complete with an administrator token')
    return root


def verify(contract_path=DEFAULT, speed_root=None):
    """验证新合同及其真实测量范围，不借用旧单提交或逐 token 相同的通过条件。"""
    contract = read(contract_path)
    require(contract['kind'] == 'flash_next_5070ti_performance_consumer', 'Wrong consumer kind')
    require(contract['mtp_default_enabled'] is False and contract['clean_os_validated'] is False,
            'Default MTP or clean-OS scope changed')
    scope = contract['runtime_scope']
    require(scope in ('compiled_staged_payload_not_installer', 'actual_installed_payload_not_clean_os'),
            'Unknown compiled runtime scope')
    require(contract['actual_install_validated'] == (scope == 'actual_installed_payload_not_clean_os'),
            'Staged runtime cannot claim actual installation acceptance')
    profile = contained(ROOT, contract['profile'])
    require(sha(profile) == contract['profile_sha256'], 'Performance profile changed')
    frozen_profile, _ = load_profile(profile)
    materials = contained(ROOT, contract['materials'])
    from validate_performance_revision import validate
    validate(profile, materials)
    wheel = contained(materials, contract['wheel']['path'])
    require(sha(wheel) == contract['wheel']['sha256'], 'FreeToken wheel changed')
    record = read(materials / 'build-record.json')
    require(record['custom_version'] == contract['version'] and
            record['wheel_sha256'] == contract['wheel']['sha256'] and
            record['input_sha256'] == frozen_profile['input_sha256'], 'Material version mismatch')
    for name, entry in record['artifacts'].items():
        path = contained(materials, name)
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                f'Material artifact changed: {name}')
    index_path = contained(ROOT, contract['evidence']['index'])
    require(sha(index_path) == contract['evidence']['sha256'], 'Evidence index changed')
    index = read(index_path)
    require(index['scope'] == scope, 'Evidence scope differs from consumer contract')
    directory = index_path.parent
    for name, entry in index['files'].items():
        path = contained(directory, name)
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                f'Evidence changed: {name}')

    def report(name):
        """每个角色必须引用已冻结文件，附带成功退出的资源监督报告。"""
        relative = index['reports'][name]
        require(relative in index['files'], f'Unindexed report: {name}')
        path = contained(directory, relative)
        resource_name = path.with_suffix('.resource.json').relative_to(directory).as_posix()
        require(resource_name in index['files'], f'Missing supervised resource report: {name}')
        resource = read(contained(directory, resource_name))
        require(resource['returncode'] == 0 and resource['stop_reason'] is None and
                not resource['remaining_descendants'], f'Incomplete process cleanup: {name}')
        return read(path)

    installed_root = None
    if scope == 'actual_installed_payload_not_clean_os':
        installed_ownership = read(contained(directory, index['reports']['installation-ownership']))
        installed_root = installation(
            read(contained(directory, index['reports']['installation-before'])),
            installed_ownership,
            read(contained(directory, index['reports']['installation-result'])),
            contract['bundle_sha256'])
        log_name = 'installation/install.log.gz'
        require(log_name in index['files'] and hashlib.sha256(gzip.decompress(
            contained(directory, log_name).read_bytes())).hexdigest() ==
            installed_ownership['install_log_sha256'], 'Installer log identity changed')

    with zipfile.ZipFile(wheel) as archive:
        expected_code = {name: hashlib.sha256(archive.read(name)).hexdigest()
                         for name in archive.namelist()
                         if name.startswith('freetoken/') and name.endswith(('.py', '.pyd'))}
    expected_aot = {p.relative_to(materials / 'kernel-cache').as_posix(): sha(p)
                    for p in (materials / 'kernel-cache').rglob('*.dll')}
    require(len(expected_code) == 459 and len(expected_aot) == 27, 'Runtime inventory changed')
    for mode in ('off', 'on'):
        quality = report('backend-' + mode + '-quality-v1')
        require(quality['status'] == 'passed' and quality['score'] >= 15 and
                quality['thinking_score'] == 3 and not quality['regressions'], 'Quality regression')
        require(quality['worker_pids_gone'] and quality['cancel']['worker_reused'], 'Backend lifecycle failed')
        identity = quality['identity']
        require(identity['freetoken_version'] == contract['version'] and
                identity['freetoken_installed_code']['files'] == expected_code and
                identity['aot_dlls']['files'] == expected_aot, 'Installed runtime differs from materials')
        functional = report('http-' + mode + '-functional-v1')
        require(functional['worker_pids_gone'] and len(functional['quality_probes']) == 4 and
                all(row['passed'] and not row['truncated'] for row in functional['quality_probes']),
                'HTTP output quality failed')
        path = contained(directory, index['reports']['http-' + mode + '-functional-v1'])
        lifecycle_name = path.with_suffix('.lifecycle.json').relative_to(directory).as_posix()
        require(lifecycle_name in index['files'], 'HTTP lifecycle evidence missing')
        lifecycle = read(contained(directory, lifecycle_name))
        require(lifecycle['status'] == 'passed' and all(lifecycle[key] for key in
                ('same_worker_after_cancel', 'original_exited_before_reload', 'replacement_exited')),
                'HTTP cancel/reload lifecycle failed')

    measured = {}
    for name in ('http-ft2-off-clean', 'http-perf2-off-clean', 'http-perf2-on-clean',
                 'http-perf2-matched-clean', 'http-off-performance-v1',
                 'http-on-performance-v1', 'http-off-matched-performance-v1'):
        value = report(name)
        measured[name] = performance(value)
        require(value['prompts_sha256'] == contract['prompts_sha256'], 'Performance prompt changed')
    base = measured['http-ft2-off-clean']
    gains = {scope: {key: values[key] / base[key] - 1 for key in base}
             for scope, values in measured.items() if scope in
             ('http-perf2-off-clean', 'http-off-performance-v1')}
    require(all(value > 0 for group in gains.values() for value in group.values()),
            'No end-to-end improvement over the measured FT2 baseline')

    for name, relative in index['reports'].items():
        if name.endswith('-v1'):
            origin_name = Path(relative).with_suffix('.origin.json').as_posix()
            require(origin_name in index['files'], f'Missing installed identity: {name}')
            origin = read(contained(directory, origin_name))
            require(origin['status'] == 'passed' and not origin['build_dirty'] and
                    origin['scope'] == scope and
                    origin['build_commit'] == contract['speed_commit'] and
                    origin['bundle_sha256'] == contract['bundle_sha256'], 'Installed identity changed')
            if installed_root:
                require(PureWindowsPath(origin['python']) == installed_root / 'python/python.exe' and
                        PureWindowsPath(origin['speed_origin']) == installed_root /
                        'python/Lib/site-packages/silitok_speed/__init__.pyd',
                        'Runtime escaped the actual installer instance')
    artifact = read(contained(directory, index['reports']['artifact']))
    require(artifact['source_commit'] == contract['speed_commit'] and
            artifact['bundle_sha256'] == contract['bundle_sha256'] and
            artifact['installer_sha256'] == contract['installer_sha256'], 'Installer identity mismatch')
    old = read(contained(directory, index['reports']['qwen36-quality']))
    require(old['status'] == 'passed' and all(old['checks'].values()), 'Old Qwen regression')
    report('qwen36-backend-v1')
    report('qwen36-http-v1')
    h3 = report('h3-http-v1')
    media = read(contained(directory, index['reports']['h3-media-check']))
    require(h3['status'] == 'passed' and h3['media']['video']['decoded_frames'] == 124 and
            h3['media']['video']['width'] == 832 and h3['media']['video']['height'] == 480 and
            media['audio_all_finite'] and media['audio_peak'] > 0, 'Packaged H3 regression')
    cleanup = read(contained(directory, index['reports']['cleanup']))
    require(cleanup['ok'] and all(cleanup['after_checks'].values()), 'Owned installation cleanup incomplete')
    require(cleanup['scope'] == scope, 'Cleanup scope differs from runtime acceptance')
    require(contract['known_issues'] == ['intermittent-long-output-token0: unresolved'],
            'Known unresolved output issue omitted')

    if speed_root:
        speed_root = Path(speed_root).resolve(strict=True)
        release = read(speed_root / 'silitok_speed/qwen36/release.json')
        require(release['version'] == contract['version'] and release['wheel'] == contract['speed_release_wheel']
                and release['producer_commit'] == contract['materials_commit'], 'Speed release binding mismatch')
        require(contract['speed_source_files_normalization'] == 'lf', 'Unknown source hash normalization')
        for name, digest in contract['speed_source_files'].items():
            content = contained(speed_root, name).read_bytes().replace(b'\r\n', b'\n')
            require(hashlib.sha256(content).hexdigest() == digest, f'Speed source changed: {name}')
        cache = speed_root / 'silitok_speed/qwen36' / release['kernel_cache']['package_path']
        require({p.relative_to(cache).as_posix(): sha(p) for p in cache.rglob('*.dll')} == expected_aot,
                'Speed bundled AOT differs')
        subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor',
                        contract['materials_commit'], 'HEAD'], check=True)
    return dict(status='verified', version=contract['version'], speed_commit=contract['speed_commit'],
                scope='local_5070ti_' + scope, performance=measured, gains=gains)


def main():
    """显式新合同入口，不重写历史 current-release 的含义。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, default=DEFAULT)
    parser.add_argument('--speed-root', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.contract, args.speed_root)))


if __name__ == '__main__':
    main()
