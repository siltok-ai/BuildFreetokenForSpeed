"""冻结普通wheel阶段证据；不创建尚未通过的Speed消费合同。"""
from pathlib import Path
import hashlib
import json
import shutil

RUN=Path(__file__).parent
BUILD=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')


def main():
    """逐报告核验完成状态、质量、生命周期与真实输出计数，再封存字节。"""
    summary=json.loads((RUN/'perf3-native-summary-v1.json').read_text(encoding='utf-8'))
    for mode in ('off','on'):
        data=json.loads((RUN/f'candidate3-wheel-quality-{mode}-v1.json').read_text(encoding='utf-8'))
        assert data['status']=='passed' and data['score']==15 and data['thinking_score']==3
        assert not data['regressions'] and data['shutdown_complete'] and len(data['rebuild_steps'])==2
        resource=json.loads((RUN/f'candidate3-wheel-quality-{mode}-v1.resource.json').read_text())
        assert resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants']
    assert json.loads((RUN/'candidate3-wheel-components-v1.json').read_text())['status']=='passed'
    assert 'Ran 37 tests' in (RUN/'build-all-tests-v4.log').read_text() and '\nOK\n' in (RUN/'build-all-tests-v4.log').read_text()
    last=json.loads((RUN/'build-full-validator-v4.log').read_text().splitlines()[-1])
    assert last['verified'] and last['performance_revision3']['verified']
    output=BUILD/'research/performance/flash-next-5070ti-20261016/perf3-native'
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    def copy(name,directory):
        """严格复制已存在证据，不用通配符混入未完成输出。"""
        target=output/directory/name
        target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(RUN/name,target)
    bases=['candidate3-wheel-quality-off-v1','candidate3-wheel-quality-on-v1',
           'candidate3-wheel-performance-off-v1','candidate3-wheel-performance-on-v1',
           'candidate3-wheel-components-v1','candidate3-source-quality-on-v1',
           'candidate3-source-quality-on-v2','candidate3-source-quality-on-v3',
           'mtp3-combined-perf-v1','off3-combined-perf-v1']
    for base in bases:
        for suffix in ('.json','.resource.json','.stdout.log','.stderr.log'):
            copy(base+suffix,'runs')
    for base in ('candidate3-source-components-v1','candidate3-source-components-v2'):
        for suffix in ('.resource.json','.stdout.log','.stderr.log'):
            copy(base+suffix,'runs')
    for name in ('perf3-native-summary-v1.json','mtp3-stage-comparison-v1.json',
                 'mtp-stage-perf2-v1.stages.json','mtp-stage-compact-v1.stages.json',
                 'derive-perf3-v1.log','perf3-material-validation-v1.log',
                 'build-full-validator-v4.log','build-all-tests-v4.log',
                 'candidate3-before-live-budget-engine.py','candidate3-before-cache-reclaim-engine.py'):
        copy(name,'support')
    for name in ('probe10.py','memory_inventory.py','test-candidate3-wheel.py','test-candidate3-source.py',
                 'freeze_candidate3.py','summarize-perf3-native.py','archive-perf3-native.py'):
        copy(name,'tools')
    copy('consumer-tools/supervise.py','tools')
    (output/'README.txt').write_text(
        'FreeToken perf3 native-wheel development checkpoint, RTX5070Ti.\n'
        '32 GPU components +4 subtests; off/on quality15/20 and thinking3/3 with no new failures;\n'
        'actual state-pool grow/shrink, graph recapture,15x512 deliveries per mode and supervised resource exits passed.\n'
        'Source v1/v2 failures retained before actual-budget/cache-reclaim fix; v3 and wheel pass.\n'
        'Full Build validator and37 tools tests passed. No perf3 Speed HTTP or installer acceptance yet.\n'
        'Raw absolute paths identify the original machine; frozen profile and patches reconstruct engine source.\n'
        'Long-output repetition/token0 remains unresolved. English MTP uplift remains insufficient; optimization continues.\n',
        encoding='utf-8')
    files={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(output.rglob('*')) if p.is_file()}
    index=dict(format=1,scope=summary['scope'],version=summary['version'],
        profile='build/profiles/strataq2perf3.json',
        profile_sha256=hashlib.sha256((BUILD/'build/profiles/strataq2perf3.json').read_bytes()).hexdigest(),
        performance_summary='support/perf3-native-summary-v1.json',files=files,
        speed_http_validated=False,installer_validated=False)
    (output/'index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,record in files.items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==record['sha256']
    print(json.dumps(dict(archived_files=len(files),scope=index['scope'])))


if __name__=='__main__':main()
