"""Freeze real installed requests and their successful, owned uninstall evidence."""
from pathlib import Path
import hashlib,json,shutil,sys
R=Path(__file__).parent
B=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
S=Path(r'C:\workspace\wt\siltok-speed\rebase-dev-20261018')
E=R/'install-i08'
sys.path.insert(0,str(B/'tools'))
from verify_performance8_delivery import verify as verify_compiled

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    compiled_path=B/'integration/speed/consumer-contract-strataq2perf9-external-compiled.json'
    compiled=json.loads(compiled_path.read_text(encoding='utf-8'))
    verify_compiled(compiled_path,speed_root=S)
    progress=json.loads((E/'suite-progress.json').read_text())
    after=json.loads((E/'cleanup/after.json').read_text(encoding='utf-8-sig'))
    assert progress['status']=='passed' and progress['scope']=='actual_installed_payload_not_clean_os'
    assert after['ok'] and after['actual_install_validated'] and all(after['after_checks'].values())
    assert after['suite_progress_sha256']==sha(E/'suite-progress.json')
    D=B/'research/review/20261019-external/installed-v1'
    assert not D.exists(),D
    D.mkdir(parents=True)
    def copy(p,rel):
        t=D/rel;t.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,t)
    for p in sorted((E/'results-v1').rglob('*')):
        if p.is_file() and p.suffix in ('.json','.log','.jsonl'):
            copy(p,'runs/'+p.relative_to(E/'results-v1').as_posix())
    for name in ('before.json','installed.json','result.json','suite-progress.json','elevated.log','install.log','service-health.json','service-auth-rejection.json'):
        copy(E/name,'installation/'+name)
    for p in sorted((E/'cleanup').iterdir()):
        if p.is_file() and p.suffix in ('.json','.log','.txt'):
            copy(p,'installation/cleanup/'+p.relative_to(E/'cleanup').as_posix())
    for name in ('run-installed-v1.py','install-test-cleanup-i08-v1.ps1','archive-installed-v1.py'):
        copy(R/name,'tools/'+name)
    for name in ('run-installed-probe.py','installed8-runtime-identity.py','run-identity.py','run-http-performance.py','supervise-v2.py','frozen-prompts.json'):
        copy(R/'installed-tools'/name,'tools/'+name)
    copy(R/'run-http-compiled-alias-v1.py','tools/run-http-compiled-alias-v1.py')
    for name in ('record-flash-install.ps1','cleanup-flash-speed.ps1'):
        copy(Path(r'C:\workspace\tools\bin')/name,'tools/'+name)
    reports={'runtime-identity':'runs/runtime-identity-v1.json'}
    reports.update({f'http-{mode}-{suite}':f'runs/http-{mode}-{suite}-v1.json'
        for mode in ('off','on') for suite in ('functional','performance')})
    files={p.relative_to(D).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(D.rglob('*')) if p.is_file()}
    scope='actual_installed_payload_not_clean_os'
    index=D/'index.json';index.write_text(json.dumps(dict(format=1,scope=scope,speed_commit=compiled['speed_commit'],
        reports=reports,progress='installation/suite-progress.json',files=files),indent=2)+'\n',encoding='utf-8')
    contract=dict(format=1,kind='flash_next_5070ti_installed_performance_consumer',runtime_scope=scope,
        **{k:compiled[k] for k in ('version','materials','profile','speed_commit','known_issues')},
        compiled_contract=dict(path=compiled_path.relative_to(B).as_posix(),sha256=sha(compiled_path)),
        evidence=dict(index=index.relative_to(B).as_posix(),sha256=sha(index)),
        mtp_default_enabled=False,actual_install_validated=True,clean_os_validated=False)
    target=B/'integration/speed/consumer-contract-strataq2perf9-external-installed.json';assert not target.exists()
    target.write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from verify_performance8_installed import verify
    print(json.dumps(verify(target,S),ensure_ascii=False))

if __name__=='__main__':main()
