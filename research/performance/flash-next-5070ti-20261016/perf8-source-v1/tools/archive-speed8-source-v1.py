"""Freeze actual Speed8 source acceptance, never substitute historical installed evidence."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
R=Path(__file__).parent
B=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
S=Path(r'C:\workspace\wt\siltok-speed\flash-perf')
sys.path.insert(0,str(B/'tools'))
from verify_performance_consumer import performance

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()
    head=subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()
    release=json.loads((S/'silitok_speed/qwen36/release.json').read_text())
    roles={};tested=None
    for mode in ('off','on'):
        for role in ('backend-quality','http-functional','http-performance'):
            name=f'speed8-source-backend-{mode}-quality-v2' if role=='backend-quality' else f'speed8-source-http-{mode}-{role.removeprefix("http-")}-v2'
            d=json.loads((R/(name+'.json')).read_text(encoding='utf-8'))
            x=json.loads((R/(name+'.resource.json')).read_text())
            assert x['returncode']==0 and x['stop_reason'] is None and not x['remaining_descendants']
            assert d['worker_pids_gone']
            if role=='backend-quality':
                assert d['status']=='passed' and (d['score'],d['thinking_score'],d['regressions'])==(15,3,[])
                inventory=d['identity']['speed_python']['files']
                if tested is None:tested=inventory
                else:assert inventory==tested
                key=f'backend-{mode}-quality'
            elif role=='http-functional':
                assert len(d['quality_probes'])==4 and all(x['passed'] for x in d['quality_probes'])
                key=f'http-{mode}-functional'
            else:performance(d);key=f'http-{mode}-performance'
            roles[key]=name+'.json'
    for name,digest in tested.items():assert sha(S/'silitok_speed'/name)==digest,name
    D=B/'research/performance/flash-next-5070ti-20261016/perf8-source-v1'
    assert not D.exists(),D
    D.mkdir(parents=True)
    def copy(p,rel):
        t=D/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    reports={}
    for role,name in roles.items():
        reports[role]='runs/'+name
        for suffix in ('.json','.resource.json','.stdout.log','.stderr.log'):
            p=(R/name).with_suffix(suffix);copy(p,'runs/'+p.name)
        if 'functional' in role:
            p=(R/name).with_suffix('.lifecycle.json');copy(p,'runs/'+p.name)
    for name in ('speed8-options-v2.log','speed8-map-ownership-v1.log','speed8-tools-v1.log','perf8-config-off-v1.log','perf8-config-on-v1.log','build-perf8-validator-v2.log','build-perf8-tools-v2.log'):
        copy(R/name,'support/'+name)
    for mode in ('off','on'):
        stem=f'speed8-source-backend-{mode}-quality-v1'
        for suffix in ('.json','.resource.json','.stdout.log','.stderr.log'):
            source=R/(stem+suffix);copy(source,'diagnostics/'+source.name)
    for mode in ('off','on'):
        root=Path(r'C:\workspace\cfg\siltok-speed\flash-perf')/f'perf8-{mode}-v1'
        for name in ('manifest.json','speed-inputs.json','speed-config/config.yaml'):copy(root/name,f'config/{mode}/'+name)
    for name in ('backend8-probe.py','run-identity.py','run-http-performance.py','run-http-smoke-v2.py','supervise-v2.py','frozen-prompts.json'):
        copy(R/'consumer-tools'/name,'tools/'+name)
    copy(Path(__file__),'tools/'+Path(__file__).name)
    files={p.relative_to(D).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file()}
    scope='speed_source_with_ordinary_freetoken_wheel_not_installer'
    index=D/'index.json';index.write_text(json.dumps(dict(format=1,scope=scope,speed_commit=head,reports=reports,files=files),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    native=B/'research/performance/flash-next-5070ti-20261016/perf8-wheel-v1/index.json'
    profile=B/'build/profiles/strataq2perf8.json'
    changed=subprocess.check_output(['git','-C',str(S),'diff','--name-only','f13ffb99','HEAD'],text=True).splitlines()
    source={name:hashlib.sha256((S/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in changed if '/kernel_cache_' not in name}
    contract=dict(format=1,kind='flash_next_5070ti_source_performance_consumer',runtime_scope=scope,
        version=release['version'],materials_commit=release['producer_commit'],speed_commit=head,
        materials='artifacts/strataq2perf8-cp312-cu130-sm120',profile=profile.relative_to(B).as_posix(),profile_sha256=sha(profile),
        wheel={k:release['wheel'][k] for k in ('path','sha256')},speed_release_wheel=release['wheel'],
        speed_source_files_normalization='lf',speed_source_files=source,speed_tested_python_files=tested,
        mtp_default_enabled=False,actual_install_validated=False,clean_os_validated=False,
        prompts_sha256=sha(R/'consumer-tools/frozen-prompts.json'),
        evidence=dict(index=index.relative_to(B).as_posix(),sha256=sha(index)),
        native_evidence=dict(index=native.relative_to(B).as_posix(),sha256=sha(native)),
        known_issues=['intermittent-long-output-token0: unresolved'])
    target=B/'integration/speed/consumer-contract-strataq2perf8.json';assert not target.exists()
    target.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(archived_files=len(files),speed_commit=head,scope=scope)))

if __name__=='__main__':main()
