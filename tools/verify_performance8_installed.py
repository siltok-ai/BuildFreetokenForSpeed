"""核对perf8实际安装、HTTP与卸载清场，复用独立编译载荷证明。"""
import argparse,hashlib,json
from pathlib import Path,PureWindowsPath
from zipfile import ZipFile
from verify_performance_consumer import contained,performance,read,require,sha
from verify_performance8_consumer import frozen_index
from verify_performance8_delivery import verify as verify_compiled

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'integration/speed/consumer-contract-strataq2perf8-installed.json'
SCOPE='actual_installed_payload_not_clean_os'


def verify(contract_path=DEFAULT,speed_root=None):
    """安装身份、原始请求时序和清理快照均必须存在，不能由暂存目录推定实装通过。"""
    contract=read(contract_path)
    require(contract['kind']=='flash_next_5070ti_installed_performance_consumer'
        and contract['runtime_scope']==SCOPE,'Wrong installed acceptance scope')
    require(contract['actual_install_validated'] is True and contract['clean_os_validated'] is False
        and contract['mtp_default_enabled'] is False,'Wrong installed release claims')
    path=contained(ROOT,contract['compiled_contract']['path'])
    require(sha(path)==contract['compiled_contract']['sha256'],'Compiled contract changed')
    compiled=read(path);compiled_result=verify_compiled(path,speed_root=speed_root)
    for key in ('version','materials','profile','speed_commit','known_issues'):
        require(contract[key]==compiled[key],'Installed/compiled identity differs: '+key)
    directory,index=frozen_index(contract['evidence'])
    require(index['scope']==SCOPE and index['speed_commit']==contract['speed_commit'],'Wrong installed evidence')
    def indexed(name):
        """每个附件必须属于逐文件SHA冻结索引。"""
        require(name in index['files'],'Unindexed installed attachment: '+name)
        return contained(directory,name)
    progress_path=indexed(index['progress'])
    progress=read(progress_path)
    required={'runtime-identity-v1',*[f'http-{mode}-{suite}-v1' for mode in ('off','on') for suite in ('functional','performance')]}
    require(progress['status']=='passed' and progress['scope']==SCOPE
        and set(progress['completed'])==required and progress['commit']==contract['speed_commit'],
        'Installed suites incomplete')
    before=read(indexed('installation/before.json'));installed=read(indexed('installation/installed.json'))
    result=read(indexed('installation/result.json'));after=read(indexed('installation/cleanup/after.json'))
    require(not before['install_present'] and not before['data_present'] and before['service'] is None
        and not before['event_source_present'] and not before['uninstall_keys'],'Existing user installation was adopted')
    require(installed['install_present'] and installed['service'] is not None
        and result['status']=='installed_service_stopped' and result['launcher_admin']
        and result['installer_exitcode']==0,'Installer did not establish the owned instance')
    require(result['installer_sha256']==compiled['installer_artifact']['sha256']
        and result['bundle_sha256']==compiled['speed_bundle']['sha256'],'Actual installation differs from accepted build')
    cleanup_checks={'install_directory_absent','programdata_absent','service_absent','owned_processes_absent',
        'service_registry_absent','event_source_absent','uninstall_keys_absent','bundle_caches_absent',
        'installed_test_trees_absent','logged_uninstaller_temps_absent','owned_shortcuts_absent',
        'source_preserved','models_preserved','venvs_preserved','tools_preserved'}
    require(after['ok'] and after['apply'] and after['uninstaller_exit']==0
        and after['scope']==SCOPE and after['actual_install_validated']
        and after['suite_progress_sha256']==sha(progress_path)
        and cleanup_checks<=set(after['after_checks']) and all(after['after_checks'].values()),
        'Installed cleanup or protected development assets failed')
    install_root=PureWindowsPath(result['install_path'])
    require(PureWindowsPath(before['install_path'])==install_root==PureWindowsPath(installed['install_path'])
        ==PureWindowsPath(after['install_path'])==PureWindowsPath(progress['app']),
        'Installation/cleanup ownership paths differ')
    def report(role):
        """实际自带解释器和保护模块必须来自同一安装目录，并有完整资源/进程清场证据。"""
        path=indexed(index['reports'][role])
        resource=read(indexed(path.with_suffix('.resource.json').relative_to(directory).as_posix()))
        origin=read(indexed(path.with_suffix('.origin.json').relative_to(directory).as_posix()))
        require(resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants']
            and resource['min_gpu_free']>=256*2**20 and resource['min_ram_available']>=2*2**30
            and resource['min_commit_available']>=2**30,'Installed resource or process cleanup failed: '+role)
        require(origin['status']=='passed' and origin['scope']==SCOPE and not origin['build_dirty']
            and origin['build_commit']==contract['speed_commit'] and origin['bundle_sha256']==compiled['speed_bundle']['sha256']
            and PureWindowsPath(origin['python'])==install_root/'python/python.exe'
            and PureWindowsPath(origin['speed_origin'])==install_root/compiled['speed_bundle']['path'],
            'Wrong actual installed interpreter/protected bundle: '+role)
        return read(path)
    identity=report('runtime-identity')
    source=read(contained(ROOT,compiled['source_contract']['path']))
    materials=contained(ROOT,contract['materials'])
    with ZipFile(contained(materials,source['wheel']['path'])) as wheel:
        code={n:hashlib.sha256(wheel.read(n)).hexdigest() for n in wheel.namelist()
            if n.startswith('freetoken/') and n.endswith(('.py','.pyd'))}
    aot={p.relative_to(materials/'kernel-cache').as_posix():sha(p) for p in (materials/'kernel-cache').rglob('*.dll')}
    source_dir,source_index=frozen_index(source['evidence'])
    source_backend=read(contained(source_dir,source_index['reports']['backend-off-quality']))
    require(identity['status']=='passed' and identity['scope']==SCOPE
        and identity['freetoken_version']==contract['version']
        and identity['freetoken_installed_code']['files']==code and identity['aot_dlls']['files']==aot
        and identity['packages']==source_backend['identity']['packages'],
        'Installed wheel, native cache or runtime versions differ')
    measured={}
    for mode in ('off','on'):
        functional=report('http-'+mode+'-functional')
        require(functional['worker_pids_gone'] and len(functional['quality_probes'])==4
            and all(r['passed'] and not r['truncated'] for r in functional['quality_probes']),
            'Actual installed HTTP quality incomplete')
        name=Path(index['reports']['http-'+mode+'-functional']).with_suffix('.lifecycle.json').as_posix()
        lifecycle=read(indexed(name))
        require(lifecycle['status']=='passed' and all(lifecycle[k] for k in
            ('same_worker_after_cancel','original_exited_before_reload','replacement_exited'))
            and lifecycle['non_greedy']['usage']['completion_tokens']>0,'Installed cancel/sampling/reload failed')
        perf=report('http-'+mode+'-performance')
        require(perf['prompts_sha256']==source['prompts_sha256'],'Installed HTTP inputs differ')
        measured[mode]=performance(perf)
    return dict(status='verified',version=contract['version'],speed_commit=contract['speed_commit'],scope=SCOPE,
        performance=measured,mtp_gain={k:measured['on'][k]/measured['off'][k]-1 for k in measured['off']},
        actual_install_validated=True,clean_os_validated=False,cleanup_verified=True,
        compiled_scope=compiled_result['scope'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,default=DEFAULT)
    parser.add_argument('--speed-root',type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.contract,args.speed_root),ensure_ascii=False))
