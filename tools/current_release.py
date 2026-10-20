"""当前 Speed 材料、构建配置和模型来源的统一入口。"""
import json
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT/'integration/speed/current-release.json'


def load_current():
    """验证指针及其冻结引用，避免默认入口构建出另一套历史材料。"""
    record = json.loads(CURRENT.read_text(encoding='utf-8'))
    for name, digest in record['input_sha256'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Current release input changed: '+name)
    if record['mtp_default_enabled'] is not False:
        raise ValueError('Current release must preserve default-off MTP')
    return record
