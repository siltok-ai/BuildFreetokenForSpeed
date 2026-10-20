"""离线校验 main 的当前材料入口、权重锁和可选 MTP 发布清单。"""
import json
from pathlib import Path
import subprocess
import sys

from current_release import ROOT, load_current
from prepare_source import load_profile
from flash_next_mtp import canonical_manifest


def verify():
    """将默认 profile、消费合同和制品绑定在同一个清晰入口。"""
    current=load_current()
    profile,_=load_profile(ROOT/current['profile'])
    consumer=json.loads((ROOT/current['consumer_contract']).read_text(encoding='utf-8'))
    _,mtp=canonical_manifest()
    if (current['version']!=profile['custom_version'] or current['version']!=consumer['version']
            or current['materials']!=consumer['materials'] or current['profile']!=consumer['profile']
            or mtp['total_bytes']!=823887872):
        raise ValueError('Current producer/Speed consumer identities disagree')
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/verify_materials.py'),str(ROOT/current['materials'])],check=True)
    installed=None
    compiled=None
    source=None
    portable=None
    if consumer.get('kind')=='flash_next_5070ti_installed_consumer':
        from verify_flash_next_5070ti_consumer import verify as verify_installed
        installed=verify_installed(ROOT/current['consumer_contract'])
    elif consumer.get('kind')=='flash_next_5070ti_compiled_performance_consumer':
        from verify_performance8_delivery import verify as verify_compiled
        compiled=verify_compiled(ROOT/current['consumer_contract'])
    elif consumer.get('kind')=='flash_next_5070ti_installed_performance_consumer':
        from verify_performance8_installed import verify as verify_installed8
        installed=verify_installed8(ROOT/current['consumer_contract'])
    elif consumer.get('kind')=='flash_next_5070ti_source_performance_consumer':
        from verify_performance8_consumer import verify as verify_source
        source=verify_source(ROOT/current['consumer_contract'])
    elif consumer.get('kind')=='flash_next_portable_resource_consumer':
        from verify_portable_consumer import verify as verify_portable
        portable=verify_portable(ROOT/current['consumer_contract'])
    else:
        raise ValueError('Unsupported current consumer kind: '+str(consumer.get('kind')))
    return dict(status='verified',version=current['version'],profile=current['profile'],
                consumer_contract=current['consumer_contract'],mtp_default_enabled=False,
                installed_5070ti=installed,compiled_5070ti=compiled,source_5070ti=source,
                portable_resource_consumer=portable)


if __name__=='__main__':
    print(json.dumps(verify()))
