"""Freeze all source runs, including every ABBA observation, against the clean Speed commit."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

R=Path(__file__).parent
OLD=R.parent/'review-20261017'
B=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
S=Path(r'C:\workspace\wt\siltok-speed\flash-perf')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    assert read(R/'candidate-b-source-all-complete-v2.json')['status']=='passed'
    assert read(R/'candidate-b-release-env-ready-v2.json')['status']=='passed'
    head=read(R/'speed-b-source-committed-v2.json')['commit']
    assert subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()==head
    assert not subprocess.check_output(['git','-C',str(S),'status','--porcelain'],text=True).strip()
    E=B/'research/review/20261017-evening/candidate-b-source-v1'
    E.mkdir(parents=True,exist_ok=False)
    def copy(src,name):
        dst=E/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    for pattern in ('candidate-b-source-*.*','baseline-http-*.*','candidate-http-*.*'):
        for path in R.glob(pattern):copy(path,'runs/'+path.name)
    checks=('speed-candidate-b-related-v2.log','candidate-b-cpu-ready-v2.json','sse-idempotency-before-v2.log',
        'sse-duplicate-ownership-before-v1.log','sse-idempotency-after-v1.log',
        'sse-real-engine-before-v2.log','sse-real-engine-after-v2.log',
        'speed-kv-before-v1.log','speed-ordinary-wheel-inventory-v1.json',
        'http-abba-summary-v3.json','native-b-quality-summary-v1.json',
        'probe-provenance-name-repair.json','native-b-probe-provenance.json',
        'build-checks-v2-complete.json','build-validator-v2.log','build-tests-after-v2.log',
        'speed-b-source-committed-v2.json','task-state.json','build-tests-preset-v1.log',
        'build-validator-v3.log','build-checks-v3-complete.json',
        'test_sse_idempotency_review2_v2.py','test_sse_duplicate_ownership_before.py',
        'abba-output-audit-v1.json','audit-abba-outputs.py')
    for name in checks:copy(R/name,'checks/'+name)
    for directory in ('consumer-tools','inputs'):shutil.copytree(OLD/directory,E/directory)
    scripts=('run-source-acceptance-b-v2.py','run-source-followup-b-v2.py',
        'run-http-smoke-review2b-v2.py','run-source-acceptance-b.py','run-source-followup-b.py','run-http-smoke-review2b.py',
        'run-speed-tests-b.py','prepare-speed-release-b.py','bind-speed-candidate-b.py',
        'run-http-smoke-review2.py','run-source-followup.py','probe-factory-b.py')
    for name in scripts:
        if (R/name).exists():copy(R/name,'tools/'+name)
    copy(Path(__file__),'freeze-source-evidence-b.py')
    reports={f'backend-{mode}-quality':f'runs/candidate-b-source-backend-{mode}-quality-v2.json'
             for mode in ('off','on')}
    reports.update({f'http-{mode}-functional':f'runs/candidate-b-source-http-{mode}-functional-v2.json'
                    for mode in ('off','on')})
    reports.update({f'http-{mode}-performance':f'runs/candidate-http-{mode}-abba2-v3.json'
                    for mode in ('off','on')})
    files={path.relative_to(E).as_posix():dict(bytes=path.stat().st_size,sha256=sha(path))
           for path in sorted(E.rglob('*')) if path.is_file()}
    index=dict(format=1,scope='speed_source_with_ordinary_freetoken_wheel_not_installer',
               speed_commit=head,reports=reports,files=files)
    (E/'index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8',newline='\n')
    release=read(S/'silitok_speed/qwen36/release.json')
    identity=read(R/'candidate-b-source-backend-off-quality-v2.json')['identity']
    names=subprocess.check_output(['git','-C',str(S),'diff','--name-only',
        'd42d3bae7f307b4651173768f323e250855aec98',head],text=True).splitlines()
    names=sorted(set(names)|{'silitok_speed/'+name for name in identity['speed_python']['files']})
    source_files={name:hashlib.sha256((S/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        for name in names if not any(part.startswith('kernel_cache') for part in Path(name).parts)
        and '/history/' not in name and (Path(name).suffix in ('.py','.md','.txt','.json','.toml')
                                        or name=='.gitattributes')}
    profile='build/profiles/strataq2review2b.json'
    base='build/profiles/strataq2review2a.json'
    native='research/review/20261017-evening/candidate-b-native-v1/evidence.json'
    contract=dict(format=1,kind='flash_next_5070ti_source_performance_consumer',
        runtime_scope=index['scope'],version=release['version'],materials_commit=release['producer_commit'],
        speed_commit=head,materials='artifacts/strataq2review2b-cp312-cu130-sm120',
        profile=profile,profile_sha256=sha(B/profile),base_profile=base,base_profile_sha256=sha(B/base),
        wheel={key:release['wheel'][key] for key in ('path','sha256')},speed_release_wheel=release['wheel'],
        speed_source_files_normalization='lf',speed_source_files=source_files,
        speed_tested_python_files=identity['speed_python']['files'],mtp_default_enabled=False,
        actual_install_validated=False,clean_os_validated=False,
        prompts_sha256=read(R/'candidate-http-off-abba2-v3.json')['prompts_sha256'],
        evidence=dict(index=(E/'index.json').relative_to(B).as_posix(),sha256=sha(E/'index.json')),
        native_evidence=dict(index=native,sha256=sha(B/native)),
        native_quality_reports={mode:f'runtime/candidate-b-native-quality-{mode}-v1.json'
                                for mode in ('off','on')},
        known_issues=['intermittent-long-output-token0: unresolved'])
    contract['measurement_note']=('Final model quality, TCP and ABBA were measured from the clean final Speed commit; '
        '355 CPU tests ran before committing identical runtime bytes. Both final ABBA candidate runs and both baseline runs per mode are retained, '
        'six real 512-token observations per prompt/variant. Primary roles use the predeclared first final-source '
        'candidate run (ABBA2-v3); pooled ABBA v3 summary is also retained. The intermediate seven-fix ABBA v2 '
        'included small CPU reproduction probes during part of that round and is retained as exploratory data, '
        'separate from the final primary v3 comparison without CPU build/test overlap. An earlier A driver stopped '
        'after its successful baseline measurement due to expecting passed rather than measured; '
        'that original run and driver are retained, never counted as a product failure.')
    name=B/'integration/speed/consumer-contract-strataq2review2b.json'
    name.write_text(json.dumps(contract,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(contract=str(name),speed_commit=head,files=len(files))))

if __name__=='__main__':main()
