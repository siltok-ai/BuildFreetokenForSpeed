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
            name=(f'speed6-source-backend-{mode}-quality-v3' if role=='backend-quality' else f'speed6-source-http-{mode}-{role.removeprefix("http-")}-v3')
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
    output=BUILD/'research/performance/flash-next-5070ti-20261016/perf6-source-v1'
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
    for name in ('speed6-options-v4.log','speed6-tools-v1.log','speed6-q2-tools-v1.log','build-perf6-validator-v3.log','build-perf6-tools-v3.log','perf6-config-off-v2.log','perf6-config-on-v2.log','perf6-source-http-summary-v1.json'):
        copy(RUN/name,'support/'+name)
    for name in ('speed6-stdio-before-v1.log','speed6-stdio-after-v1.log'):
        copy(RUN/name,'diagnostics/'+name)
    for revision in (1,2):
        stem=f'speed6-source-http-off-functional-v{revision}'
        for suffix in ('.resource.json','.stdout.log','.stderr.log'):
            source=RUN/(stem+suffix)
            copy(source,'diagnostics/'+source.name)
    note=dict(scope='failed_diagnostics_not_acceptance',
        v1='First cold non-greedy request corrupted the worker protocol; D4 reproduced and fixed stdout isolation.',
        v2='After D4, raw resource monitor terminated at253849600B free VRAM. The exact256MiB transient matches Triton cold-autotune cache buffer; no acceptance claimed.',
        v3='Accepted roles below use128 fewer resident experts in both modes and a fresh compiler cache; external256MiB floor unchanged.')
    (output/'diagnostics/README.json').write_text(json.dumps(note,indent=2)+'\n',encoding='utf-8')
    for mode in ('off','on'):
        root=Path(r'C:\workspace\cfg\siltok-speed\flash-perf')/f'perf6-{mode}-v2'
        for name in ('manifest.json','speed-inputs.json','speed-config/config.yaml'):
            copy(root/name,f'config/{mode}/'+name)
    for name in ('backend6-probe.py','run-identity.py','run-http-performance.py','run-http-smoke-v2.py','supervise-v2.py','frozen-prompts.json'):
        copy(RUN/'consumer-tools'/name,'tools/'+name)
    copy(Path(__file__),'tools/'+Path(__file__).name)
    files={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(output.rglob('*')) if p.is_file()}
    scope='speed_source_with_ordinary_freetoken_wheel_not_installer'
    index=dict(format=1,scope=scope,speed_commit=head,reports=reports,files=files)
    path=output/'index.json'
    path.write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    native=BUILD/'research/performance/flash-next-5070ti-20261016/perf6-wheel-v1/index.json'
    profile=BUILD/'build/profiles/strataq2perf6.json'
    changed=subprocess.check_output(['git','-C',str(SPEED),'diff','--name-only','f13ffb99','HEAD'],text=True).splitlines()
    source_files={name:hashlib.sha256((SPEED/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                  for name in changed if '/kernel_cache_' not in name}
    contract=dict(format=1,kind='flash_next_5070ti_source_performance_consumer',runtime_scope=scope,
        version=release['version'],materials_commit=release['producer_commit'],speed_commit=head,
        materials='artifacts/strataq2perf6-cp312-cu130-sm120',profile=profile.relative_to(BUILD).as_posix(),
        profile_sha256=sha(profile),wheel={k:release['wheel'][k] for k in ('path','sha256')},
        speed_release_wheel=release['wheel'],speed_source_files_normalization='lf',
        speed_source_files=source_files,speed_tested_python_files=tested_python,
        mtp_default_enabled=False,actual_install_validated=False,clean_os_validated=False,
        prompts_sha256=sha(RUN/'consumer-tools/frozen-prompts.json'),
        evidence=dict(index=path.relative_to(BUILD).as_posix(),sha256=sha(path)),
        native_evidence=dict(index=native.relative_to(BUILD).as_posix(),sha256=sha(native)),
        known_issues=['intermittent-long-output-token0: unresolved'])
    target=BUILD/'integration/speed/consumer-contract-strataq2perf6.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(archived_files=len(files),speed_commit=head,scope=scope)))


if __name__=='__main__':main()
