"""Validate the selected FreeToken materials and Flash Next resource inputs offline."""
import json

from current_release import ROOT, load_current
from prepare_source import load_profile
from flash_next_mtp import canonical_manifest
from verify_portable_consumer import verify as verify_consumer


def verify():
    current = load_current()
    profile, _ = load_profile(ROOT/current['profile'])
    consumer = json.loads((ROOT/current['consumer_contract']).read_text(encoding='utf-8'))
    _, mtp = canonical_manifest()
    source = json.loads((ROOT/current['weight_source_lock']).read_text(encoding='utf-8'))
    if (current['version'] != profile['custom_version'] or current['version'] != consumer['version']
            or current['materials'] != consumer['materials'] or current['profile'] != consumer['profile']
            or consumer['mtp']['tensor_count'] != len(mtp['tensors'])
            or consumer['mtp']['runtime_bytes'] != mtp['total_bytes']
            or len(source['gguf']['files']) != 2 or 'metadata' in source):
        raise ValueError('Current materials and model resource inputs disagree')
    result = verify_consumer(ROOT/current['consumer_contract'])
    return dict(status='verified', scope='offline_source_and_materials', version=current['version'],
                profile=current['profile'], mtp_default_enabled=current['mtp_default_enabled'],
                consumer=result, runtime_executed=False)


if __name__ == '__main__':
    print(json.dumps(verify()))
