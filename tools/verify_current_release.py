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
    return dict(status='verified',version=current['version'],profile=current['profile'],
                consumer_contract=current['consumer_contract'],mtp_default_enabled=False)


if __name__=='__main__':
    print(json.dumps(verify()))
