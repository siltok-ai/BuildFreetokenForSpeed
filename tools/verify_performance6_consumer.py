"""核对perf6源码Speed消费证据；此检查点不宣称安装器已经验收。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from prepare_source import load_profile
from validate_pinned_tensor_revision import validate as validate_materials
from verify_performance_consumer import contained, performance, read, require, sha

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'integration/speed/consumer-contract-strataq2perf6.json'
SCOPE='speed_source_with_ordinary_freetoken_wheel_not_installer'


def frozen_index(reference):
    """先验全部原始文件的SHA，不能只读取手写汇总。"""
    path=contained(ROOT,reference['index'])
    require(sha(path)==reference['sha256'],'Evidence index changed')
    index=read(path)
    records=index['files']
    rows=records.items() if isinstance(records,dict) else ((row['path'],row) for row in records)
    for name,record in rows:
        item=contained(path.parent,name)
        require(item.stat().st_size==record['bytes'] and sha(item)==record['sha256'],
                'Evidence changed: '+name)
    return path.parent,index


def quality(report, native):
    """按真实逐题结果重算分数，并与已验收的同配置原生输出逐token交叉核对。"""
    require(report['status']=='passed' and report['worker_pids_gone'] and
            report['cancel']['worker_reused'],'Backend quality/lifecycle incomplete')
    source={row['name']:row for row in native['runs']}
    expected_names={row['name'] for row in native['runs'] if 'quality_pass' in row}
    rows={row['name']:row for row in report['runs'] if 'quality_pass' in row and row['name'] not in ('warmup','after_cancel')}
    require(set(rows)==expected_names,'Quality cases differ from the frozen native suite')
    for name,row in rows.items():
        require(row['final']['tokens']==source[name]['output']['token_ids'] and
                row['quality_pass']==source[name]['quality_pass'],'Native/Speed quality differs: '+name)
    score=sum(row['quality_pass'] for name,row in rows.items() if not name.startswith('thinking-'))
    thinking=sum(row['quality_pass'] for name,row in rows.items() if name.startswith('thinking-'))
    require(score==report['score']==15 and thinking==report['thinking_score']==3 and
            not report['regressions'],'Quality regression')


def verify(contract_path=DEFAULT,speed_root=None):
    """将材料、源码身份、真实HTTP时序及资源监督合成可回溯的阶段检查点。"""
    contract=read(contract_path)
    require(contract['kind']=='flash_next_5070ti_source_performance_consumer' and
            contract['runtime_scope']==SCOPE,'Wrong source consumer scope')
    require(contract['mtp_default_enabled'] is False and contract['actual_install_validated'] is False
            and contract['clean_os_validated'] is False,'Source checkpoint cannot claim installer acceptance')
    require(contract['known_issues']==['intermittent-long-output-token0: unresolved'],
            'Unresolved long-output issue was omitted')
    profile=contained(ROOT,contract['profile'])
    require(sha(profile)==contract['profile_sha256'],'Profile changed')
    data,_=load_profile(profile)
    require(data['custom_version']==contract['version']=='0.1.3+speed.strataq2perf6','Wrong material version')
    materials=contained(ROOT,contract['materials'])
    validate_materials(profile,materials)
    wheel=contained(materials,contract['wheel']['path'])
    require(sha(wheel)==contract['wheel']['sha256'],'Wheel changed')
    with zipfile.ZipFile(wheel) as archive:
        code={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
              if name.startswith('freetoken/') and name.endswith(('.py','.pyd'))}
    require(len(code)==462,'Unexpected wheel inventory')
    aot={p.relative_to(materials/'kernel-cache').as_posix():sha(p)
         for p in (materials/'kernel-cache').rglob('*.dll')}
    require(len(aot)==27,'Incomplete native cache')
    native_root,native_index=frozen_index(contract['native_evidence'])
    require(native_index['scope']=='Ordinary installed FreeToken wheel; Speed remains separate acceptance',
            'Native checkpoint scope changed')
    directory,index=frozen_index(contract['evidence'])
    require(index['scope']==SCOPE,'Source evidence scope changed')

    def report(role):
        """所有实测角色都必须有成功退出的资源监督证据。"""
        relative=index['reports'][role]
        require(relative in index['files'],'Unindexed role: '+role)
        path=contained(directory,relative)
        resource_relative=path.with_suffix('.resource.json').relative_to(directory).as_posix()
        require(resource_relative in index['files'],'Missing resource supervision: '+role)
        resource=read(contained(directory,resource_relative))
        require(resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants'],
                'Resource or cleanup failure: '+role)
        return read(path)

    measured={}
    for mode,resident in [('off',5760),('on',4992)]:
        backend=report('backend-'+mode+'-quality')
        quality(backend,read(native_root/f'runtime/perf6-wheel-quality-{mode}-v1.json'))
        identity=backend['identity']
        require(identity['freetoken_version']==contract['version'] and
                identity['freetoken_installed_code']['files']==code and identity['aot_dlls']['files']==aot,
                'Tested FreeToken differs from the frozen wheel/cache')
        require(identity['speed_python']['files']==contract['speed_tested_python_files'],
                'Tested source identity changed')
        options=backend['options']
        require(options['moe_q2_resident_experts']==resident and options['moe_cache_size']==resident+2048
                and options['max_prefill_length']==1024 and options['linear_state_cache_slots']==1
                and options['moe_q2_adapt_every']==32 and options['mtp_compact_state']==(mode=='on')
                and options['moe_q2_adapt_decay']==0.7 and options['gguf_embedding_host']
                and options['ple_row_cache_rows']==1048576
                and options['moe_q2_prefill_workspace'] and options['fixed_q2_rebuild_reserve_mib']==0,
                'Measured configuration changed')
        functional=report('http-'+mode+'-functional')
        require(functional['worker_pids_gone'] and len(functional['quality_probes'])==4 and
                all(row['passed'] and not row['truncated'] for row in functional['quality_probes']),
                'HTTP quality incomplete')
        lifecycle_path=Path(index['reports']['http-'+mode+'-functional']).with_suffix('.lifecycle.json').as_posix()
        require(lifecycle_path in index['files'],'Missing HTTP lifecycle')
        lifecycle=read(contained(directory,lifecycle_path))
        require(lifecycle['status']=='passed' and all(lifecycle[key] for key in
                ('same_worker_after_cancel','original_exited_before_reload','replacement_exited')),
                'HTTP cancel/reload lifecycle failed')
        value=report('http-'+mode+'-performance')
        measured[mode]=performance(value)
        require(value['prompts_sha256']==contract['prompts_sha256'],'HTTP input identity changed')
    gain={key:measured['on'][key]/measured['off'][key]-1 for key in measured['off']}
    if speed_root is not None:
        speed_root=speed_root.resolve(strict=True)
        release=read(contained(speed_root,'silitok_speed/qwen36/release.json'))
        require(release['version']==contract['version'] and release['wheel']==contract['speed_release_wheel']
                and release['producer_commit']==contract['materials_commit'],'Speed material binding changed')
        for name,digest in contract['speed_source_files'].items():
            payload=contained(speed_root,name).read_bytes().replace(b'\r\n',b'\n')
            require(hashlib.sha256(payload).hexdigest()==digest,'Speed source changed: '+name)
        head=subprocess.check_output(['git','-C',str(speed_root),'rev-parse','HEAD'],text=True).strip()
        require(head==contract['speed_commit'],'Speed checkpoint commit changed')
    return dict(status='verified',version=contract['version'],scope=SCOPE,speed_commit=contract['speed_commit'],
                performance=measured,mtp_gain_vs_best_off=gain,actual_install_validated=False)


def main():
    """默认核对perf6源码消费合同，保留历史perf2安装版入口。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,default=DEFAULT)
    parser.add_argument('--speed-root',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.contract,args.speed_root),ensure_ascii=False))


if __name__=='__main__':main()
