"""验证 MTP2 消费合同、冻结材料和可选的 Speed 原始证据工作树。"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

from prepare_source import ROOT, load_profile, sha256
from validate_python_revision import validate as validate_materials


def verify(speed_checkout=None):
    """材料按完整补丁重放核对；外部证据只在实际读取 Speed 提交后给出通过。"""
    record=json.loads((ROOT/'integration/speed/consumer-contract-strataq2mtp2.json').read_text(encoding='utf-8'))
    profile=ROOT/record['profile'];materials=ROOT/record['materials'];adapter=record['speed_adapter']
    snapshot=ROOT/adapter['snapshot']
    if sha256(profile)!=record['profile_sha256'] or sha256(snapshot)!=adapter['evidence_sha256']:
        raise ValueError('Consumer profile or Speed checkpoint snapshot drift')
    summary=json.loads(snapshot.read_bytes())
    if (summary['source']['speed_commit']!=adapter['source_commit'] or summary['source']['producer_commit']!=record['producer_engine_commit']
            or summary['acceptance']!=record['acceptance'] or summary['real_http_performance']!=record['performance']
            or summary['package_identity']!=adapter['package']):
        raise ValueError('Consumer acceptance differs from frozen Speed evidence')
    if record['mtp']['default_enabled'] is not False or record['acceptance']['full_installer_clean_machine'] is not False:
        raise ValueError('Checkpoint must preserve default OFF and installer scope')
    loaded,_=load_profile(profile)
    if loaded['custom_version']!=record['version'] or sha256(materials/record['wheel']['path'])!=record['wheel']['sha256']:
        raise ValueError('Consumer FreeToken wheel/version drift')
    if sha256(materials/record['kernel_cache']['manifest'])!=record['kernel_cache']['manifest_sha256']:
        raise ValueError('Consumer AOT manifest drift')
    validate_materials(profile,materials)
    speed_raw=False
    if speed_checkout is not None:
        speed_checkout=Path(speed_checkout).resolve(strict=True)
        data=subprocess.run(['git','show',adapter['evidence_commit']+':'+adapter['evidence']],cwd=speed_checkout,check=True,capture_output=True).stdout
        if hashlib.sha256(data).hexdigest()!=adapter['evidence_sha256'] or (speed_checkout/adapter['evidence']).read_bytes()!=data:
            raise ValueError('Speed checkout is not the accepted evidence commit')
        subprocess.run([sys.executable,'-B',str(speed_checkout/'tools/verify_mtp_checkpoint.py')],cwd=speed_checkout,check=True)
        speed_raw=True
    return dict(status='verified',profile=record['profile'],version=record['version'],speed_evidence_commit=adapter['evidence_commit'],speed_raw_evidence_verified=speed_raw,full_installer_clean_machine=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--speed-checkout',type=Path)
    print(json.dumps(verify(parser.parse_args().speed_checkout)))
