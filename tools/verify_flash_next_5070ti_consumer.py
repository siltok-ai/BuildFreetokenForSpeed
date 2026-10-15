"""离线核对5070 Ti真实安装验收；不重新授予其他设备或干净OS运行资格。"""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import statistics
import subprocess
import zipfile

from current_release import ROOT, load_current
from verify_flash_next_5070ti import require, verify as verify_source


def contained(root, name):
    """所有验收引用必须落在对应仓库或封存目录内。"""
    path=(root/name).resolve()
    require(path.is_relative_to(root.resolve()), f'Acceptance path escapes its root: {name}')
    return path


def read(path):
    """读取固定JSON，不导入产品或初始化GPU。"""
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify(contract_path=None, speed_root=None):
    """核对来源字节、安装身份、实际输出、性能统计和清场事实。"""
    current=load_current()
    path=contract_path or contained(ROOT,current['consumer_contract'])
    contract=read(path)
    require(contract['kind']=='flash_next_5070ti_installed_consumer','Not a 5070 Ti installed consumer')
    require(contract['acceptance']['full_installer'] is True and
            contract['acceptance']['clean_os'] is False,'Installed/clean-OS scope changed')
    verify_source()
    index_path=contained(ROOT,contract['evidence']['index'])
    require(hashlib.sha256(index_path.read_bytes()).hexdigest()==contract['evidence']['sha256'],
            'Installed acceptance index changed')
    index=read(index_path);directory=index_path.parent
    for name,entry in index['files'].items():
        content=contained(directory,name).read_bytes()
        require(len(content)==entry['bytes'] and hashlib.sha256(content).hexdigest()==entry['sha256'],
                f'Installed evidence changed: {name}')
    roles=index['reports']
    package=contract['speed_adapter']['package']
    commit=contract['speed_adapter']['source_commit']
    require(index['source_commit']==commit and
            index['evidence_commit']==contract['speed_adapter']['evidence_commit'],
            'Installed evidence commit binding changed')

    def evidence_read(path):
        """角色报告和来源/资源侧车也必须纳入逐文件冻结清单。"""
        require(path.relative_to(directory).as_posix() in index['files'],
                f'Unindexed installed evidence: {path.name}')
        return read(path)

    def role(name):
        """按封存角色定位报告，不依赖验收机器的绝对路径。"""
        return evidence_read(contained(directory,roles[name]))

    def run(name):
        """每个GPU任务必须来自同一干净提交的实际安装包，并完成进程树回收。"""
        report_path=contained(directory,roles[name])
        origin=evidence_read(report_path.with_suffix('.origin.json'))
        resource=evidence_read(report_path.with_suffix('.resource.json'))
        require(origin['status']=='passed' and origin['build_commit']==commit and not origin['build_dirty']
                and origin['bundle_sha256']==package['bundle_sha256']
                and origin['build_version']==package['version'],f'Wrong installed product: {name}')
        require(resource['returncode']==0 and resource['stop_reason'] is None
                and not resource['remaining_descendants'],f'Incomplete installed process cleanup: {name}')
        return evidence_read(report_path)

    materials=contained(ROOT,contract['materials'])
    require(hashlib.sha256(contained(ROOT,contract['profile']).read_bytes()).hexdigest()==contract['profile_sha256'],
            'Consumer profile changed')
    require(hashlib.sha256(contained(materials,contract['kernel_cache']['manifest']).read_bytes()).hexdigest()
            ==contract['kernel_cache']['manifest_sha256'],'Consumer AOT manifest changed')
    artifact=role('artifact')
    require(artifact==package['installer'] and artifact['version']==package['version']
            and artifact['source_commit']==commit and artifact['bundle_sha256']==package['bundle_sha256'],
            'Installer artifact identity differs from installed product')
    wheel=contained(materials,contract['wheel']['path'])
    require(hashlib.sha256(wheel.read_bytes()).hexdigest()==contract['wheel']['sha256'],'Wrong FreeToken wheel')
    with zipfile.ZipFile(wheel) as archive:
        expected_code={name:hashlib.sha256(archive.read(name)).hexdigest() for name in archive.namelist()
                       if name.startswith('freetoken/') and name.endswith(('.py','.pyd'))}
    expected_aot={p.relative_to(materials/'kernel-cache').as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (materials/'kernel-cache').rglob('*.dll')}
    require(len(expected_code)==459 and len(expected_aot)==27,'Frozen runtime inventory changed')
    baseline=read_quality_baseline()
    for mode in ('off','on'):
        report=run(f'backend-{mode}-quality-v1')
        identity=report['identity']
        require(identity['freetoken_installed_code']['files']==expected_code
                and identity['aot_dlls']['files']==expected_aot,'Installed runtime differs from paired materials')
        require(report['status']=='passed' and report['worker_pids_gone'] and report['cancel']['worker_reused'],
                'Installed backend lifecycle failed')
        cases=[row for row in report['runs'] if row['name'] in baseline]
        require(len(cases)==20 and sum(row['quality_pass'] for row in cases)==15,'Q2 quality scope changed')
        require(all(row['final']['tokens']==baseline[row['name']]['output']['token_ids'] for row in cases),
                'Installed Q2 token IDs differ')
        thinking=[row for row in report['runs'] if row['name'].startswith('thinking-')]
        require(len(thinking)==3 and all(row['quality_pass'] for row in thinking),'Installed thinking regression')

    performance={}
    for mode in ('off','on','off-matched'):
        report=run(f'http-{mode}-performance-v1')
        require(report['status']=='measured' and report['worker_pids_gone'] and len(report['runs'])==6,
                'Missing installed HTTP performance runs')
        grouped={}
        for row in report['runs']:
            times=row['arrival_offsets'];context=row['label'].rsplit('-',1)[0]
            require(row['output_tokens']==row['usage']['completion_tokens']==row['content_chunk_count']==len(times)==512
                    and row['done_events']==1 and row['finish_reason']=='length','Not 512 actual SSE outputs')
            require(row['usage']['prompt_tokens']=={'70-0':70,'diverse-4096-0':4096}[context],'Input length changed')
            require(all(b>=a for a,b in zip(times,times[1:])),'SSE times moved backwards')
            rate=511/(times[-1]-times[0])
            require(math.isclose(rate,row['decode_tps'],rel_tol=1e-7),'Decode timing mismatch')
            grouped.setdefault(context,[]).append(rate)
        for context,rates in grouped.items():
            require(len(rates)==3 and math.isclose(statistics.median(rates),
                    report['contexts'][context]['median_tps'],rel_tol=1e-7),'Installed median mismatch')
        performance[mode]=report
    on,matched=performance['on'],performance['off-matched']
    require(on['args']['protected_experts']==matched['args']['protected_experts']=='2304','Capacity comparison changed')
    for enabled,disabled in zip(on['runs'],matched['runs']):
        require(all(enabled[key]==disabled[key] for key in ('label','text','reasoning_content','usage','finish_reason')),
                'Matched-capacity installed outputs differ')
    for name in ('http-off-functional-v1','http-on-functional-v1','qwen36-http-v1'):
        report=run(name);lifecycle=evidence_read(contained(directory,roles[name]).with_suffix('.lifecycle.json'))
        require(report['worker_pids_gone'] and lifecycle['status']=='passed'
                and all(lifecycle[key] for key in ('same_worker_after_cancel','original_exited_before_reload',
                                                  'replacement_exited')),'Installed HTTP lifecycle failed')
        require(len(report['quality_probes'])==4 and all(row['passed'] and not row['truncated']
                for row in report['quality_probes']),'Installed HTTP meaningful-output gate failed')
    run('qwen36-backend-v1')
    quality=role('qwen36-quality')
    require(quality['status']=='passed' and all(quality['checks'].values()),'Old 35B regression')
    h3=run('h3-http-v1');media=role('h3-media-check')
    video=h3['media']['video'];audio=h3['media']['audio']
    require(h3['status']=='passed' and video['decoded_frames']==124 and video['width']==832 and video['height']==480
            and audio['decoded_samples']>0 and media['frames']==124 and media['audio_all_finite']
            and media['audio_peak']>0 and media['first_middle_mean_absolute_difference']>1,'H3 media regression')
    service=role('normal-service');cleanup=role('cleanup')
    require(service['status']=='passed' and service['test_bypass'] is False
            and service['health']['commit']==commit and not service['health']['dirty'],'Real service identity failed')
    require(cleanup['ok'] and all(cleanup['after_checks'].values()),'Owned installation cleanup incomplete')
    known_issues=contract.get('known_issues',[])
    require(len(known_issues)==1,'Recorded unresolved delivery issue must remain explicit')
    for issue in known_issues:
        require(issue['id']=='intermittent-long-output-token0' and issue['state']=='unresolved'
                and issue['delivery_exception_authorized_by_user'] is True,
                'Unrecognized or silently resolved delivery exception')
        failed=role(issue['first_failure_report'])
        logic=next(row for row in failed['quality_probes'] if row['id']=='constraint_logic')
        require(not logic['passed'] and logic['truncated'] and logic['finish_reason']=='length'
                and logic['usage']['completion_tokens']==2048 and '!'*64 in logic['actual'],
                'Original intermittent failure was lost or changed')
        for key in ('source_recheck_report','installed_recheck_report','numeric_stress_report',
                    'unobserved_stress_report'):
            repeated=role(issue[key])
            cases=repeated['quality_probes']
            if key=='unobserved_stress_report':
                # 原题重复与新增前缀是不同输入；新题的质量失败不能被抹去，也不能冒充原token0复现。
                original=[row for row in cases if row['unique_prefix'] is False]
                additional=[row for row in cases if row['unique_prefix'] is True and row['id']=='constraint_logic']
                require(len(original)>=48 and all(row['passed'] and not row['truncated'] for row in original),
                        'Original unobserved input did not pass the recorded repetitions')
                score=dict(tested=len(additional),passed=sum(row['passed'] for row in additional))
                require(score==issue['additional_prompt_variants'] and
                        all('!'*64 not in row['actual'] for row in additional),
                        'Additional prompt failures were omitted or misclassified')
            else:
                require(len(cases)>=4 and all(row['passed'] and not row['truncated'] for row in cases),
                        'Delivery exception lacks successful rechecks')
            require(repeated['worker_pids_gone'],'Known-issue recheck left workers')
        sampling_steps=[]
        for sample_role in issue['numeric_sample_reports']:
            sample=role(sample_role)
            require(sample['first_nonfinite'] is None,'Numeric diagnostic observed a nonfinite logit')
            sampling_steps.extend(sample['steps'])
        require(len(sampling_steps)>=13737 and all(row['nan_count']==row['inf_count']==0
                and row['token']!=0 for row in sampling_steps),'Numeric stress scope or result changed')
        native=role(issue['additional_variant_native_report'])
        native_variant=next(row for row in native['runs'] if row['id']=='constraint_logic_prefix23')
        stressed=role(issue['unobserved_stress_report'])
        failed_variants=[row for row in stressed['quality_probes'] if row['unique_prefix']
                         and row['id']=='constraint_logic' and not row['passed']]
        require(len(failed_variants)==1 and native['shutdown_complete']
                and native_variant['output']['text'].strip()==failed_variants[0]['actual'].strip()
                and len(native_variant['input_ids'])==failed_variants[0]['usage']['prompt_tokens']
                and len(native_variant['output']['token_ids'])==failed_variants[0]['usage']['completion_tokens']
                and not native_variant['repeated_token0'],
                'Additional prompt failure is not independently reproduced in native FreeToken')
        require(contract['acceptance']['status']=='accepted_with_known_issue',
                'Unresolved failure cannot be reported as unconditional acceptance')
    if speed_root is not None:
        git=['git','-C',str(speed_root)]
        head=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
        require(head in {commit,contract['speed_adapter']['evidence_commit']},'Unexpected Speed checkout')
        require(not subprocess.check_output(git+['status','--porcelain']),'Speed checkout is dirty')
        require(not subprocess.check_output(git+['diff',commit,'HEAD','--','silitok_speed','packaging/windows/build.ps1']),
                'Speed runtime/build differs from validated source')
    return dict(verified=True,scope='archived_actual_installed_5070ti_acceptance',speed_commit=commit,
                actual_http_512_token_runs=18,matched_mtp_outputs=6,full_installer=True,clean_os=False,
                unresolved_known_issues=[issue['id'] for issue in known_issues])


def read_quality_baseline():
    """沿用不可变Q2质量基线，历史五个失败不改判。"""
    path=ROOT/'research/acceptance/flash-next-5070ti-20261015/source/historical-quality.json.gz'
    return {row['name']:row for row in json.loads(gzip.decompress(path.read_bytes()))['runs']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path)
    parser.add_argument('--speed-root',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.contract,args.speed_root),ensure_ascii=False))
