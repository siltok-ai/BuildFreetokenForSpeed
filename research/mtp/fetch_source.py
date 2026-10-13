"""只读取固定revision的MTP张量，独立核验完整集合，禁止回退到main。"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVISION = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
OUT = Path('C:/workspace/models/Qwen/Qwen3.8-Flash-Next') / REVISION / 'mtp-source-bf16'
STRATA_COMMIT = '6674a0065fb96bacde33e3eb10f91a1df86f95f2'


def main():
    """限定新权重目录、来源与完整张量集合；下载后独立核验每个文件。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'fetch', 'verify'])
    args = parser.parse_args()
    origins = json.loads((HERE / 'references/origins.json').read_text(encoding='utf-8'))
    assert origins['commit'] == STRATA_COMMIT
    code = HERE / 'references/mtp_fetch.py'
    canonical = code.read_bytes()
    identity = next(r for r in origins['files'] if r['path'] == 'mtp_fetch.py')
    assert hashlib.sha256(canonical).hexdigest() == identity['sha256']
    spec = importlib.util.spec_from_file_location('strata_mtp_fetch', code)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.REVISION = REVISION
    module.REPO = module.PINNED = f'https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/{REVISION}/'
    assert len(module.SHA256) == 31
    marker = OUT / 'source-lock.json'
    lock = dict(format=1, repository='Qwen/Qwen3.8-Flash-Next', revision=REVISION,
        repo_url=module.REPO, strata_commit=STRATA_COMMIT,
        strata_fetch_sha256=hashlib.sha256(canonical).hexdigest(), tensor_sha256=module.SHA256,
        complete_tensor_count=31, scope='MTP-only BF16 source; not a standalone main checkpoint')
    if marker.exists():
        assert json.loads(marker.read_text(encoding='utf-8')) == lock
    else:
        if OUT.exists() and any(OUT.iterdir()):
            raise FileExistsError('Refuse to adopt an unowned tensor directory')
        OUT.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(lock, indent=2), encoding='utf-8')
    if args.command == 'inventory':
        target = OUT / 'mtp-inventory.json'
        if target.exists():
            raise FileExistsError(target)
        rows = module.inventory(str(OUT))
        assert {r['name'] for r in rows} == set(module.SHA256)
        assert all(r['dtype']=='BF16' for r in rows)
        print(json.dumps(dict(output=str(OUT), tensors=len(rows),
            bytes=sum(r['bytes'] for r in rows), shards=len({r['shard'] for r in rows}))))
        return
    inv = json.loads((OUT/'mtp-inventory.json').read_text(encoding='utf-8'))
    assert inv['repo'] == module.REPO
    rows = inv['tensors']
    assert len(rows) == 31 and {r['name'] for r in rows} == set(module.SHA256)
    if args.command == 'fetch':
        module.fetch(str(OUT), None)
    verified = []
    for row in rows:
        tensor = OUT / 'tensors' / (row['name']+'.bin')
        assert tensor.stat().st_size == row['bytes'], row['name']
        digest = module.sha256_of(tensor)
        assert digest == module.SHA256[row['name']], row['name']
        verified.append(dict(row, sha256=digest, path=str(tensor), mtime_ns=tensor.stat().st_mtime_ns))
    result = dict(passed=True, revision=REVISION, complete_tensor_count=31,
        total_bytes=sum(r['bytes'] for r in rows), tensors=verified)
    (OUT/'complete-source-verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='tensors'}))


if __name__ == '__main__':
    main()
