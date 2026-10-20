"""复核 GGUF 独立资源包的冻结消费证据，不把离线检查算作新的实机运行。"""
from pathlib import Path
import argparse
import json
import subprocess

from prepare_source import ROOT, load_profile
from validate_python_revision import validate
from verify_performance8_consumer import frozen_index


def verify(contract_path, speed_root=None):
    """分别核对材料复用、各消费层 OFF/ON、删除与搬迁记录、以及性能对照范围。"""
    contract = json.loads(Path(contract_path).read_text(encoding='utf-8'))
    if speed_root is not None:
        commit = subprocess.check_output(['git','-C',str(speed_root),'rev-parse','HEAD'],text=True).strip()
        if commit != contract['speed_commit']:
            raise ValueError('Speed checkpoint commit changed')
    if (contract['kind'] != 'flash_next_portable_resource_consumer'
            or contract['license_mode'] != 'test_stub'
            or contract['device_license_validated'] is not False
            or contract['clean_os_validated'] is not False):
        raise ValueError('Wrong portable resource qualification scope')
    profile, _ = load_profile(ROOT/contract['profile'])
    if profile['custom_version'] != contract['version']:
        raise ValueError('Portable consumer version differs from profile')
    validate(ROOT/contract['profile'], ROOT/contract['materials'], ROOT/contract['base_profile'])
    directory, index = frozen_index(contract['evidence'])
    def report(role):
        return json.loads((directory/index['reports'][role]).read_text(encoding='utf-8'))
    stages = ['source', 'ordinary', 'protected']
    if contract['actual_install_validated']:
        stages.append('installed')
    for stage in stages:
        for mode in ('off', 'on'):
            item = report(stage+'-'+mode)
            if (item['status'] != 'passed' or not item['server_stopped'] or not item['workers_gone']
                    or item['mode'] != mode or not item['hf_offline']
                    or item['license_mode'] != 'test_stub'):
                raise ValueError('Incomplete portable HTTP/lifecycle proof: '+stage+'-'+mode)
            if stage != 'source' and item['mtp']['enabled'] != (mode == 'on'):
                raise ValueError('MTP ownership differs from selected mode')
    package = report('package')
    if (not package['radix_files_removed'] or package['remaining_legacy_metadata_files']
            or package['external_links'] or not package['all_weight_paths_relative']
            or not package['relocated_before_inference'] or not package['gguf_timestamps_changed']):
        raise ValueError('Package still depends on source-machine inputs')
    for mode in ('off','on'):
        candidate, control = report('ordinary-'+mode), report('control-'+mode)
        rows, old = candidate['benchmarks'], control['benchmarks']
        if len(rows) != 6 or len(old) != 6:
            raise ValueError('Incomplete English/Chinese performance comparison')
        for a,b in zip(rows,old):
            if (a['prompt_ids'] != b['prompt_ids'] or a['language'] != b['language']
                    or a['repetition'] != b['repetition'] or a['output_tokens'] != 512
                    or b['output_tokens'] != 512 or a['finish_reason'] != 'length'
                    or b['finish_reason'] != 'length'):
                raise ValueError('Performance inputs or actual delivered lengths differ')
    if contract.get('receiver_simulation_validated'):
        receiver = report('receiver')
        if (receiver['status'] != 'passed' or len(receiver['rounds']) != 8
                or len(receiver['restarts']) != 2
                or receiver['actual_install_validated'] != contract['actual_install_validated']
                or any(row['returncode'] != 0 or not row['workers_gone'] for row in receiver['restarts'])):
            raise ValueError('Incomplete receiver switching/restart proof')
        for mode in ('off','on'):
            rows = receiver['quality'][mode]
            measured = dict(no_thinking=sum(row['quality_pass'] for row in rows if not row['thinking']),
                thinking=sum(row['quality_pass'] for row in rows if row['thinking']))
            if len(rows) != 23 or measured != contract['receiver_quality_scores'][mode]:
                raise ValueError('Receiver quality summary differs from raw samples')
    return dict(status='verified', version=contract['version'],
        metadata_source='embedded_gguf', runtime_stages=stages,
        actual_install_validated=contract['actual_install_validated'],
        receiver_simulation_validated=contract.get('receiver_simulation_validated', False),
        device_license_validated=False, clean_os_validated=False)


def main():
    """兼容 Speed 的消费检查入口，实际复核合同后才返回成功。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,
        default=ROOT/'integration/speed/consumer-contract-strataq2portable1.json')
    parser.add_argument('--speed-root',type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.contract,args.speed_root)))


if __name__ == '__main__':
    main()
