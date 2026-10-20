"""复核 GGUF 独立资源包的冻结消费证据，不把离线检查算作新的实机运行。"""
from pathlib import Path
import json

from prepare_source import ROOT, load_profile
from validate_python_revision import validate
from verify_performance8_consumer import frozen_index


def verify(contract_path):
    """分别核对材料复用、各消费层 OFF/ON、删除与搬迁记录、以及性能对照范围。"""
    contract = json.loads(Path(contract_path).read_text(encoding='utf-8'))
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
    return dict(status='verified', version=contract['version'],
        metadata_source='embedded_gguf', runtime_stages=stages,
        actual_install_validated=contract['actual_install_validated'],
        device_license_validated=False, clean_os_validated=False)
