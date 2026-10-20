"""固定 MTP 发布身份与真实 payload 核验；本机转换记录不进入发布 SHA。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flash_next_weights import LOCK, ROOT, layout, read, verify_file, write


def canonical_manifest():
    """原样保留当前 Speed 已批准的清单，避免改变其来源/哈希合同。"""
    lock = read(LOCK)
    path = LOCK.parent/lock['mtp']['manifest']
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != lock['mtp']['manifest_sha256']:
        raise ValueError('Frozen MTP manifest changed')
    manifest = json.loads(data)
    if (manifest['source']['revision'] != lock['mtp']['revision']
            or len(manifest['tensors']) != lock['mtp']['tensor_count']
            or manifest['total_bytes'] != lock['mtp']['runtime_bytes']):
        raise ValueError('Frozen MTP manifest geometry changed')
    return data, manifest


def verify_runtime(root, require_manifest=True):
    """核对31份真实数据，拒绝多余/缺失/重复清单或同长度损坏。"""
    root = Path(root).resolve()
    data, manifest = canonical_manifest()
    if require_manifest and (root/'runtime.json').read_bytes() != data:
        raise ValueError('Runtime manifest differs from the frozen release')
    names = [row['path'] for row in manifest['tensors']]
    if len(set(names)) != 31 or {p.name for p in root.glob('*.bin')} != set(names):
        raise ValueError('MTP requires exactly the 31 certified payloads')
    for row in manifest['tensors']:
        path = root/row['path']
        if path.resolve().parent != root:
            raise ValueError('MTP payload escaped its runtime directory')
        verify_file(path, row)
    return dict(status='verified', manifest_sha256=hashlib.sha256(data).hexdigest(),
                tensors=31, bytes=manifest['total_bytes'])


def finalize_runtime(root, generated_report):
    """数据全部匹配才写入批准清单；历史转换时间属于原清单来源，不代表本次耗时。"""
    root = Path(root)
    if (root/'runtime.json').exists():
        raise FileExistsError('Existing runtime manifest is preserved')
    data, frozen = canonical_manifest()
    actual = {r['name']:r for r in generated_report['tensors']}
    if len(actual) != 31 or len(generated_report['tensors']) != 31:
        raise ValueError('Incomplete generated MTP tensor catalog')
    for expected in frozen['tensors']:
        row = actual.get(expected['name'], {})
        if any(row.get(k) != expected[k] for k in ('shape','kind','type_id','path','bytes','sha256','source_sha256')):
            raise ValueError('Converted payload differs from the approved recipe: '+expected['name'])
    verify_runtime(root, require_manifest=False)
    write(root/'conversion-report.json', dict(generated_report,
        release_manifest_sha256=hashlib.sha256(data).hexdigest(),
        note='Local conversion diagnostics; runtime.json is the immutable approved release record.'))
    (root/'runtime.json').write_bytes(data)
    return verify_runtime(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','fetch','build','verify'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    paths = layout(args.root, read(LOCK))
    source = args.source or paths['mtp_source']
    if args.command == 'plan':
        result = dict(source=str(source), runtime=str(paths['mtp']), **read(LOCK)['mtp'])
    elif args.command == 'verify':
        result = verify_runtime(paths['mtp'])
    else:
        import subprocess
        import sys
        if args.command == 'fetch':
            script = ROOT/'tools/mtp/fetch_source.py'
            if not (source/'mtp-inventory.json').exists():
                subprocess.run([sys.executable,'-B',str(script),'inventory','--output',str(source)],check=True)
            subprocess.run([sys.executable,'-B',str(script),'fetch','--output',str(source)],check=True)
            result = dict(status='source_verified', source=str(source))
        else:
            subprocess.run([sys.executable,'-B',str(ROOT/'tools/mtp/build_runtime.py'),
                            '--source',str(source),'--output',str(paths['mtp'])],check=True)
            result = verify_runtime(paths['mtp'])
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
