"""源码HTTP验收完成并提交后生成独立合同，保留安装版perf2合同原貌。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

RUN=Path(__file__).parent
BUILD=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
SPEED=Path(r'C:\workspace\wt\siltok-speed\flash-perf')
sys.path.insert(0,str(BUILD/'tools'))
from verify_performance_consumer import performance


def sha(path):
    """原始字节身份用于归档，源码可搬迁比较另行统一LF。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """缺少任何模式、真实时序或退出证据则停止，不生成部分成功合同。"""
    assert not subprocess.check_output(['git','-C',str(SPEED),'status','--porcelain'],text=True).strip()
    head=subprocess.check_output(['git','-C',str(SPEED),'rev-parse','HEAD'],text=True).strip()
    release=json.loads((SPEED/'silitok_speed/qwen36/release.json').read_text())
    roles={}
    tested_python=None
    for mode in ('off','on'):
        for role in ('backend-quality','http-functional','http-performance'):
            name=f'speed-perf3-{role}-{mode}-v1'
            data=json.loads((RUN/(name+'.json')).read_text(encoding='utf-8'))
            resource=json.loads((RUN/(name+'.resource.json')).read_text())
            assert resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants']
            assert data['worker_pids_gone']
            if role=='backend-quality':
                assert data['status']=='passed' and data['score']==15 and data['thinking_score']==3 and not data['regressions']
                inventory=data['identity']['speed_python']['files']
                if tested_python is None:tested_python=inventory
                else:assert inventory==tested_python
                key=f'backend-{mode}-quality'
            elif role=='http-functional':
                assert len(data['quality_probes'])==4 and all(row['passed'] for row in data['quality_probes'])
                lifecycle=json.loads((RUN/(name+'.lifecycle.json')).read_text(encoding='utf-8'))
                assert lifecycle['status']=='passed' and lifecycle['replacement_exited']
                key=f'http-{mode}-functional'
            else:
                performance(data)
                key=f'http-{mode}-performance'
            roles[key]=name+'.json'
    for name,digest in tested_python.items():
        assert sha(SPEED/'silitok_speed'/name)==digest,'Tested product Python changed: '+name
    output=BUILD/'research/performance/flash-next-5070ti-20261016/perf3-source'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    def copy(source,name):
        """严格保留相对目录和原始字节；不复制checkpoint junction或权重。"""
        target=output/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    reports={}
    for role,name in roles.items():
        reports[role]='runs/'+name
        for suffix in ('.json','.resource.json','.stdout.log','.stderr.log'):
            source=(RUN/name).with_suffix(suffix)
            copy(source,'runs/'+source.name)
        if 'functional' in role:
            source=(RUN/name).with_suffix('.lifecycle.json')
            copy(source,'runs/'+source.name)
    for name in ('speed-perf3-adapter-tests-v1.log','speed-perf3-adapter-tests-v2.log',
                 'speed-perf3-adapter-tests-v3.log','speed-perf3-tools-tests-v1.log',
                 'perf3-consumer-tests-v1.log','build-all-tests-v6.log','build-full-validator-v5.log',
                 'perf3-config-off-v1.log','perf3-config-on-v1.log','perf3-source-http-summary-v1.json'):
        copy(RUN/name,'support/'+name)
    for mode in ('off','on'):
        root=Path(r'C:\workspace\cfg\siltok-speed\flash-perf')/f'perf3-{mode}-v1'
        for name in ('manifest.json','speed-inputs.json','speed-config/config.yaml'):
            copy(root/name,f'config/{mode}/'+name)
    for name in ('backend3-probe.py','run-identity.py','run-http-performance.py','run-http-smoke-v2.py','supervise.py','frozen-prompts.json'):
        copy(RUN/'consumer-tools'/name,'tools/'+name)
    copy(Path(__file__),'tools/'+Path(__file__).name)
    files={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(output.rglob('*')) if p.is_file()}
    scope='speed_source_with_ordinary_freetoken_wheel_not_installer'
    index=dict(format=1,scope=scope,speed_commit=head,reports=reports,files=files)
    path=output/'index.json'
    path.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    native=BUILD/'research/performance/flash-next-5070ti-20261016/perf3-native/index.json'
    profile=BUILD/'build/profiles/strataq2perf3.json'
    changed=subprocess.check_output(['git','-C',str(SPEED),'diff','--name-only','f13ffb99','HEAD'],text=True).splitlines()
    source_files={name:hashlib.sha256((SPEED/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                  for name in changed if '/kernel_cache_' not in name}
    contract=dict(format=1,kind='flash_next_5070ti_source_performance_consumer',runtime_scope=scope,
        version=release['version'],materials_commit=release['producer_commit'],speed_commit=head,
        materials='artifacts/strataq2perf3-cp312-cu130-sm120',profile=profile.relative_to(BUILD).as_posix(),
        profile_sha256=sha(profile),wheel={k:release['wheel'][k] for k in ('path','sha256')},
        speed_release_wheel=release['wheel'],speed_source_files_normalization='lf',
        speed_source_files=source_files,speed_tested_python_files=tested_python,
        mtp_default_enabled=False,actual_install_validated=False,clean_os_validated=False,
        prompts_sha256=sha(RUN/'consumer-tools/frozen-prompts.json'),
        evidence=dict(index=path.relative_to(BUILD).as_posix(),sha256=sha(path)),
        native_evidence=dict(index=native.relative_to(BUILD).as_posix(),sha256=sha(native)),
        known_issues=['intermittent-long-output-token0: unresolved'])
    target=BUILD/'integration/speed/consumer-contract-strataq2perf3.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(archived_files=len(files),speed_commit=head,scope=scope)))


if __name__=='__main__':main()
