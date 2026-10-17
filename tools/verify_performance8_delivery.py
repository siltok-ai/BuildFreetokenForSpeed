"""核对perf8及同几何后续编译载荷，区分暂存包、安装器与全新系统。"""
import argparse
import hashlib
import json
from pathlib import Path, PureWindowsPath
from zipfile import ZipFile

from verify_performance_consumer import contained, performance, read, require, sha
from verify_performance8_consumer import frozen_index, quality, verify as verify_source

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'integration/speed/consumer-contract-strataq2perf8-compiled.json'
SCOPE='compiled_staged_payload_not_installer'


def runtime_python(files):
    """使用与真实封装器相同的目录边界，排除构建缓存和封装脚本。"""
    return {name:digest for name,digest in files.items()
        if Path(name).name!='setup_compile.py' and not any(
            part in {'__pycache__','.pytest_cache','build','.artifacts'} or part.endswith('.egg-info')
            for part in Path(name).parts)}


def verify(contract_path=DEFAULT,payload_root=None,speed_root=None):
    """将实际解释器/保护模块/源码清单绑定到同一套质量、HTTP和资源证据。"""
    contract=read(contract_path)
    require(contract['kind']=='flash_next_5070ti_compiled_performance_consumer'
        and contract['runtime_scope']==SCOPE,'Wrong compiled acceptance scope')
    require(contract['actual_install_validated'] is False and contract['clean_os_validated'] is False,
        'Staged payload cannot claim installer or clean-OS acceptance')
    source_path=contained(ROOT,contract['source_contract']['path'])
    require(sha(source_path)==contract['source_contract']['sha256'],'Source contract changed')
    source=read(source_path)
    source_result=verify_source(source_path,speed_root)
    for key in ('version','materials','profile','speed_commit'):
        require(contract[key]==source[key],'Compiled/source identity differs: '+key)
    directory,index=frozen_index(contract['evidence'])
    require(index['scope']==SCOPE and index['speed_commit']==contract['speed_commit'],
        'Wrong compiled evidence scope or commit')
    def indexed(name):
        """所有被读取的验收附件都必须属于冻结索引。"""
        require(name in index['files'],'Unindexed runtime attachment: '+name)
        return contained(directory,name)
    manifest=read(indexed(index['public_manifest']))
    require(manifest['build_commit']==contract['speed_commit'] and not manifest['build_dirty'],
        'Compiled bundle was not built from the clean source checkpoint')
    expected=runtime_python(source['speed_tested_python_files'])
    actual={name:row['source_sha256'] for name,row in manifest['modules'].items()}
    require(actual==expected,'Compiled runtime Python differs from the source-tested product')
    materials=contained(ROOT,contract['materials'])
    with ZipFile(contained(materials,source['wheel']['path'])) as archive:
        wheel_code={name:hashlib.sha256(archive.read(name)).hexdigest()
            for name in archive.namelist() if name.startswith('freetoken/') and name.endswith(('.py','.pyd'))}
    aot={p.relative_to(materials/'kernel-cache').as_posix():sha(p)
        for p in (materials/'kernel-cache').rglob('*.dll')}
    native_root,_=frozen_index(source['native_evidence'])
    source_directory,source_index=frozen_index(source['evidence'])
    def report(role):
        """逐角色核对原始监督与封装器身份，不能只相信最后的汇总数字。"""
        name=index['reports'][role]
        path=indexed(name)
        resource=read(indexed(Path(name).with_suffix('.resource.json').as_posix()))
        require(resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants'],
            'Runtime resource or process cleanup failed: '+role)
        require(resource['min_gpu_free']>=256*2**20 and resource['min_ram_available']>=2*2**30
            and resource['min_commit_available']>=2**30,'Actual resource floor crossed: '+role)
        origin=read(indexed(Path(name).with_suffix('.origin.json').as_posix()))
        require(origin['status']=='passed' and origin['scope']==SCOPE
            and origin['build_commit']==contract['speed_commit'] and not origin['build_dirty']
            and origin['build_version']==manifest['build_version']
            and origin['bundle_sha256']==contract['speed_bundle']['sha256'],
            'Wrong embedded/protected runtime identity: '+role)
        product=PureWindowsPath(origin['speed_origin'])
        require(product.name=='__init__.pyd' and PureWindowsPath(origin['python'])==product.parents[3]/'python.exe',
            'Runtime did not use its own embedded Python: '+role)
        return read(path)
    measured={}
    for mode,resident in [('off',5760),('on',4992)]:
        backend=report('backend-'+mode+'-quality')
        native_reports = source.get('native_quality_reports', {
            value: f'runtime/perf8-wheel-quality-{value}-v1.json' for value in ('off', 'on')})
        quality(backend,read(contained(native_root,native_reports[mode])))
        identity=backend['identity']
        require(identity['freetoken_version']==contract['version']
            and identity['freetoken_installed_code']['files']==wheel_code
            and identity['aot_dlls']['files']==aot
            and identity['speed_bundle_sha256']==contract['speed_bundle']['sha256'],
            'Loaded wheel/native cache or protected bundle changed')
        options=backend['options']
        source_backend=read(contained(source_directory,source_index['reports']['backend-'+mode+'-quality']))
        require(options==source_backend['options'] and identity['packages']==source_backend['identity']['packages'],
            'Compiled/source options or runtime dependency versions differ')
        require(options['moe_q2_resident_experts']==resident and options['moe_cache_size']==resident+2048
            and options['moe_q2_prefill_workspace'] and options['fixed_q2_rebuild_reserve_mib']==0
            and options['linear_state_cache_slots']==1 and options['mtp_compact_state']==(mode=='on'),
            'Compiled runtime configuration differs from source acceptance')
        functional=report('http-'+mode+'-functional')
        require(functional['worker_pids_gone'] and len(functional['quality_probes'])==4
            and all(row['passed'] and not row['truncated'] for row in functional['quality_probes']),
            'Compiled HTTP quality incomplete')
        name=index['reports']['http-'+mode+'-functional']
        lifecycle=read(indexed(Path(name).with_suffix('.lifecycle.json').as_posix()))
        require(lifecycle['status']=='passed' and all(lifecycle[key] for key in
            ('same_worker_after_cancel','original_exited_before_reload','replacement_exited'))
            and lifecycle['non_greedy']['usage']['completion_tokens']>0,
            'Compiled HTTP sampling/cancel/reload incomplete')
        value=report('http-'+mode+'-performance')
        require(value['prompts_sha256']==source['prompts_sha256'],'Compiled HTTP inputs changed')
        measured[mode]=performance(value)
    if payload_root is not None:
        bundle=contained(payload_root,contract['speed_bundle']['path'])
        require(bundle.stat().st_size==contract['speed_bundle']['bytes']
            and sha(bundle)==contract['speed_bundle']['sha256'],'Local protected payload changed')
    return dict(status='verified',version=contract['version'],speed_commit=contract['speed_commit'],
        scope=SCOPE,source_scope=source_result['scope'],performance=measured,
        actual_install_validated=False,clean_os_validated=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,default=DEFAULT)
    parser.add_argument('--payload-root',type=Path)
    parser.add_argument('--speed-root',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.contract,args.payload_root,args.speed_root),ensure_ascii=False))
